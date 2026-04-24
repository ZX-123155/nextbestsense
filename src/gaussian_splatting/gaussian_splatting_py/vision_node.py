#!/usr/bin/env python3

import os
import datetime
import os.path as osp
import subprocess
import sys
import shutil
import pickle
import threading
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
from collections import deque
from typing import Deque, List, Optional, Tuple, Union


import tf2_ros as tf2
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as sciR

import rospy

from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo

from gaussian_splatting_py.splatfacto3d import Splatfacto3D as splatfacto

from gaussian_splatting_py.vision_utils.vision_utils import convert_intrinsics, learn_scale_and_offset_raw, warp_image
from gaussian_splatting_py.load_yaml import load_config
from gaussian_splatting_py.monocular_depth import MonocularDepth

from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest, NBVPoses, NBVPosesResponse, NBVPosesRequest, NBVResult, NBVResultRequest, NBVResultResponse, SaveModel, SaveModelRequest, SaveModelResponse

class VisionNode(object):

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _repo_root() -> str:
        return osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "../../.."))

    @staticmethod
    def _resolve_python_cmd(configured_cmd: str) -> str:
        if configured_cmd:
            configured_cmd = str(configured_cmd).strip()
            if configured_cmd and shutil.which(configured_cmd):
                return configured_cmd
        candidate_paths = [
            "/home/ras/miniconda3/envs/sam2/bin/python",
            osp.expanduser("~/miniconda3/envs/sam2/bin/python"),
        ]
        for cmd in candidate_paths + ["python3.11", "python3", sys.executable]:
            if cmd and (osp.isabs(cmd) and osp.isfile(cmd) or shutil.which(cmd)):
                return cmd
        return "python3"

    def _get_sam2_env_name(self) -> str:
        """Get conda environment name for SAM2"""
        return "sam2"

    def _sam2_subprocess_env(self) -> dict:
        env = os.environ.copy()
        env["SAM2_REPO_DIR"] = self.sam2_repo_dir
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [self.sam2_repo_dir] + ([existing_pythonpath] if existing_pythonpath else [])
        env["PYTHONPATH"] = ":".join([p for p in pythonpath_parts if p])
        return env

    @staticmethod
    def _depth_to_uint16(depth_m: np.ndarray) -> np.ndarray:
        safe_depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        safe_depth[safe_depth < 0] = 0
        safe_depth[safe_depth > 65.535] = 0
        return np.round(safe_depth * 1000.0).astype(np.uint16)

    @staticmethod
    def _depth_msg_to_meters(depth_img: Image, depth_np: np.ndarray) -> np.ndarray:
        encoding = str(depth_img.encoding).lower()
        depth_np = np.array(depth_np, copy=False)
        if encoding in {"16uc1", "mono16"}:
            depth_m = depth_np.astype(np.float32) / 1000.0
        elif encoding in {"32fc1", "32fc"}:
            depth_m = depth_np.astype(np.float32)
        else:
            depth_m = depth_np.astype(np.float32)
            if np.nanmax(depth_m) > 100.0:
                depth_m = depth_m / 1000.0
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m[depth_m < 0] = 0
        return depth_m

    @staticmethod
    def _color_msg_to_rgb(img_msg: Image, img_np: np.ndarray) -> np.ndarray:
        encoding = str(img_msg.encoding).lower()
        if encoding in {"rgb8", "rgb16"}:
            return img_np
        if encoding in {"bgr8", "bgr16"}:
            return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        if encoding in {"mono8", "mono16"}:
            return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        # Fallback: assume OpenCV default BGR-like channel order.
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return img_np

    @staticmethod
    def _get_param_compat(private_name: str, global_name: str, default):
        private_key = f"~{private_name}"
        if rospy.has_param(private_key):
            return rospy.get_param(private_key)
        global_candidates = [global_name, f"/{global_name}"]
        for key in global_candidates:
            if rospy.has_param(key):
                return rospy.get_param(key)
        return default

    @staticmethod
    def _msg_stamp_to_sec(msg: Image) -> Optional[float]:
        if msg is None or not hasattr(msg, "header"):
            return None
        stamp = msg.header.stamp
        if stamp is None:
            return None
        sec = stamp.to_sec()
        if sec <= 0.0:
            return None
        return float(sec)

    def _rgb_msg_cb(self, msg: Image) -> None:
        with self._msg_buffer_lock:
            self._rgb_msg_buffer.append(msg)

    def _depth_msg_cb(self, msg: Image) -> None:
        with self._msg_buffer_lock:
            self._depth_msg_buffer.append(msg)

    def _pick_latest_rgb_and_nearest_depth(self) -> Tuple[Optional[Image], Optional[Image], Optional[float]]:
        with self._msg_buffer_lock:
            rgb_msgs = list(self._rgb_msg_buffer)
            depth_msgs = list(self._depth_msg_buffer)

        if len(rgb_msgs) == 0:
            return None, None, None

        rgb_msg = rgb_msgs[-1]
        if len(depth_msgs) == 0:
            return rgb_msg, None, None

        rgb_stamp = self._msg_stamp_to_sec(rgb_msg)
        if rgb_stamp is None:
            return rgb_msg, depth_msgs[-1], None

        best_depth = None
        best_delta = None
        for depth_msg in reversed(depth_msgs):
            depth_stamp = self._msg_stamp_to_sec(depth_msg)
            if depth_stamp is None:
                continue
            cur_delta = abs(depth_stamp - rgb_stamp)
            if best_delta is None or cur_delta < best_delta:
                best_delta = cur_delta
                best_depth = depth_msg
            if best_delta is not None and best_delta <= self.rgb_depth_sync_tolerance_s:
                break

        if best_depth is None:
            best_depth = depth_msgs[-1]
        return rgb_msg, best_depth, best_delta

    def _wait_for_synced_rgb_depth(self) -> Tuple[Image, Image, Optional[float]]:
        end_time = rospy.Time.now().to_sec() + max(0.0, self.rgb_depth_sync_wait_s)
        best_rgb = None
        best_depth = None
        best_delta = None

        while rospy.Time.now().to_sec() <= end_time and not rospy.is_shutdown():
            rgb_msg, depth_msg, delta = self._pick_latest_rgb_and_nearest_depth()
            if rgb_msg is not None:
                best_rgb = rgb_msg
            if depth_msg is not None:
                best_depth = depth_msg
            if delta is not None and (best_delta is None or delta < best_delta):
                best_delta = delta
            if best_rgb is not None and best_depth is not None:
                if best_delta is None or best_delta <= self.rgb_depth_sync_tolerance_s:
                    break
            rospy.sleep(0.01)

        if best_rgb is None:
            best_rgb = rospy.wait_for_message(self.CAMERA_TOPIC, Image)
        if best_depth is None:
            best_depth = rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)

        if best_delta is None:
            rgb_stamp = self._msg_stamp_to_sec(best_rgb)
            depth_stamp = self._msg_stamp_to_sec(best_depth)
            if rgb_stamp is not None and depth_stamp is not None:
                best_delta = abs(depth_stamp - rgb_stamp)

        return best_rgb, best_depth, best_delta

    def __init__(self) -> None:
        rospy.init_node("vision_node")
        rospy.loginfo("Vision Node Initializing")

        self.script_dir = osp.dirname(osp.abspath(__file__))
        self.repo_root = self._repo_root()
        self.sam2_repo_dir = osp.abspath(osp.expanduser(rospy.get_param("~sam2_repo_dir", "~/sam2")))
        configured_py_cmd = rospy.get_param("~python_cmd", "")
        self.python_cmd = self._resolve_python_cmd(configured_py_cmd)

        # fetch parameter
        self.CAMERA_TOPIC = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera/color/camera_info")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_link = rospy.get_param("~ee_link", "Link6")
        self.camera_link = rospy.get_param("~camera_link", "ee_camera_optical_frame")
        self.touch_link = rospy.get_param("~touch_link", "touch")
        self.camera_color_frame = rospy.get_param("~camera_color_frame", "ee_camera_optical_frame")
        self.camera_depth_frame = rospy.get_param("~camera_depth_frame", "ee_camera_optical_frame")
        
        # depth cam topic and cam info
        self.DEPTH_CAMERA_TOPIC = rospy.get_param("~depth_image_topic", "/camera/depth/image_rect_raw")
        self.depth_cam_info_topic = rospy.get_param("~depth_cam_info_topic", "/camera/depth/camera_info")
        
        self.save_data = self._parse_bool(self._get_param_compat("save_data", "save_data", True))
        self.should_collect_test_views = self._parse_bool(
            self._get_param_compat("should_collect_test_views", "should_collect_test_views", False)
        )
        self.wait_for_offline_sam2 = self._parse_bool(
            self._get_param_compat("wait_for_offline_sam2", "wait_for_offline_sam2", False)
        )
        self.offline_sam2_done_flag = self._get_param_compat(
            "offline_sam2_done_flag", "offline_sam2_done_flag", ".offline_sam2_done"
        )
        self.reuse_existing_dataset = self._parse_bool(
            self._get_param_compat("reuse_existing_dataset", "reuse_existing_dataset", False)
        )
        self.existing_dataset_dir = osp.abspath(
            str(self._get_param_compat("existing_dataset_dir", "existing_dataset_dir", ""))
        )
        default_save_data_dir = osp.join(self.repo_root, "data")
        default_gs_data_dir = osp.join(self.repo_root, "outputs", "bunny_blender_data")
        self.save_data_dir = osp.abspath(rospy.get_param("~save_data_dir", default_save_data_dir))
        self.gs_data_dir = osp.abspath(rospy.get_param("~gs_data_dir", default_gs_data_dir))
        os.makedirs(self.save_data_dir, exist_ok=True)
        os.makedirs(self.gs_data_dir, exist_ok=True)
        
        self.is_challenge_object = self._parse_bool(
            rospy.get_param("~is_challenge_object", rospy.get_param("is_challenge_object", False))
        )

        self.train_split_fraction = float(rospy.get_param("~train_split_fraction", "0.85"))
        self.camera_optimizer_mode = rospy.get_param("~camera_optimizer_mode", "off")
        self.quats_lr = float(rospy.get_param("~quats_lr", "5e-4"))
        self.scales_lr = float(rospy.get_param("~scales_lr", "2e-3"))
        self.opacities_lr = float(rospy.get_param("~opacities_lr", "2e-2"))
        self.uncertainty_object_mask_weight = float(rospy.get_param("~uncertainty_object_mask_weight", "0.0"))
        self.uncertainty_mask_gamma = float(rospy.get_param("~uncertainty_mask_gamma", "4.0"))
        self.uncertainty_bg_floor = float(rospy.get_param("~uncertainty_bg_floor", "0.005"))
        self.uncertainty_fruit_weight = float(rospy.get_param("~uncertainty_fruit_weight", "1.0"))
        self.uncertainty_leaf_weight = float(rospy.get_param("~uncertainty_leaf_weight", "0.4"))
        self.uncertainty_bg_weight = float(rospy.get_param("~uncertainty_bg_weight", "0.02"))
        self.gs_max_iterations = int(rospy.get_param("~gs_max_iterations", "11000"))
        
        # GS model
        self.gs_training = False
        self.gs_model = splatfacto(
            data_dir=self.gs_data_dir,
            train_split_fraction=self.train_split_fraction,
            camera_optimizer_mode=self.camera_optimizer_mode,
            quats_lr=self.quats_lr,
            scales_lr=self.scales_lr,
            opacities_lr=self.opacities_lr,
            uncertainty_object_mask_weight=self.uncertainty_object_mask_weight,
            uncertainty_mask_gamma=self.uncertainty_mask_gamma,
            uncertainty_bg_floor=self.uncertainty_bg_floor,
            uncertainty_fruit_weight=self.uncertainty_fruit_weight,
            uncertainty_leaf_weight=self.uncertainty_leaf_weight,
            uncertainty_bg_weight=self.uncertainty_bg_weight,
        )
        
        # construct monocular depth model
        param_filename = osp.join(osp.dirname(osp.abspath(__file__)), "config.yml") 
        self.monocular_depth = MonocularDepth(load_config(param_filename))
        
        # whether to use SAM2 for depth alignment.
        self.use_sam = self._parse_bool(self._get_param_compat("use_sam", "use_sam", True))
        
        rospy.loginfo("Camera Topic: {}".format(self.CAMERA_TOPIC))
        rospy.loginfo("Depth Camera Topic: {}".format(self.DEPTH_CAMERA_TOPIC))
        rospy.loginfo("Save Data: {}".format(self.save_data))
        rospy.loginfo("Save Data Dir: {}".format(self.save_data_dir))
        rospy.loginfo("Use SAM2 alignment: {}".format(self.use_sam))
        rospy.loginfo("Wait for offline SAM2: {}".format(self.wait_for_offline_sam2))
        rospy.loginfo("Reuse existing dataset: {}".format(self.reuse_existing_dataset))
        if self.reuse_existing_dataset:
            rospy.loginfo("Existing dataset dir: {}".format(self.existing_dataset_dir))
        rospy.loginfo("Train split fraction: {}".format(self.train_split_fraction))
        rospy.loginfo("GS max iterations: {}".format(self.gs_max_iterations))
        rospy.loginfo("Camera optimizer mode: {}".format(self.camera_optimizer_mode))
        rospy.loginfo("Quat/Scale/Opacity LR: {}/{}/{}".format(self.quats_lr, self.scales_lr, self.opacities_lr))
        rospy.loginfo(
            "Uncertainty mask params (object_weight/gamma/bg_floor): {}/{}/{}".format(
                self.uncertainty_object_mask_weight, self.uncertainty_mask_gamma, self.uncertainty_bg_floor
            )
        )
        rospy.loginfo(
            "Uncertainty class weights (fruit/leaf/bg): {}/{}/{}".format(
                self.uncertainty_fruit_weight, self.uncertainty_leaf_weight, self.uncertainty_bg_weight
            )
        )
        rospy.loginfo("SAM2 Repo Dir: {}".format(self.sam2_repo_dir))
        rospy.loginfo("Python command for helper scripts: {}".format(self.python_cmd))

        # wait for image topic
        rospy.loginfo("Waiting for camera topic")
        rospy.wait_for_message(self.CAMERA_TOPIC, Image)
        rospy.loginfo("Camera topic found")
        
        # wait for depth image topic
        rospy.loginfo("Waiting for depth camera topic")
        rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)
        rospy.loginfo("Depth Camera topic found")
        self.bridge = CvBridge()
        
        # get camera infos 
        rospy.loginfo("Waiting for camera infos")
        self.color_cam_info: CameraInfo = rospy.wait_for_message(self.cam_info_topic, CameraInfo)
        self.depth_cam_info: CameraInfo = rospy.wait_for_message(self.depth_cam_info_topic, CameraInfo)

        # store images in (H, W, 3) [0 - 255]
        self.images: List[np.ndarray] = []
        # store depth images in (H, W) [0 - max val of uint16]
        self.depths: List[np.ndarray] = []
        # store monocular depth images in (H, W) [0 - max val of uint16]
        self.monocular_depths: List[np.ndarray] = []
        # store realsense depth images in (H, W) [0 - max val of uint16]
        self.realsense_depths: List[np.ndarray] = []
        # store original camera to world pose 
        self.poses: List[np.ndarray] = []
        # store poses for next best view to send to GS
        self.poses_for_nbv: List[np.ndarray] = []
        self.depth_aligned_complete = []
        for _ in range(100):
            self.depth_aligned_complete.append(False)
        
        self.scores: List[float] = []
        self.nbv_wait_timeout_s = float(rospy.get_param("~nbv_wait_timeout_s", "300.0"))
        self.depth_retry_count = int(rospy.get_param("~depth_retry_count", "4"))
        self.rgb_depth_sync_tolerance_s = float(rospy.get_param("~rgb_depth_sync_tolerance_s", "0.05"))
        self.rgb_depth_sync_wait_s = float(rospy.get_param("~rgb_depth_sync_wait_s", "0.4"))
        self._rgb_msg_buffer: Deque[Image] = deque(maxlen=30)
        self._depth_msg_buffer: Deque[Image] = deque(maxlen=120)
        self._msg_buffer_lock = threading.Lock()

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        self.rgb_sub = rospy.Subscriber(self.CAMERA_TOPIC, Image, self._rgb_msg_cb, queue_size=5)
        self.depth_sub = rospy.Subscriber(self.DEPTH_CAMERA_TOPIC, Image, self._depth_msg_cb, queue_size=20)
        self.idx = 0
        self.data_base_dir = None
        self.gs_training_dir = None
        self.reuse_dataset_working_dir = None
        session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.online_sam2_dir = osp.join(self.save_data_dir, f".sam2_online_{session_ts}")
        self.online_sam2_images_dir = osp.join(self.online_sam2_dir, "images")
        self.online_sam2_masks_dir = osp.join(self.online_sam2_dir, "masks")
        os.makedirs(self.online_sam2_images_dir, exist_ok=True)
        os.makedirs(self.online_sam2_masks_dir, exist_ok=True)

        if self.reuse_existing_dataset:
            working_dataset_dir = self._prepare_reuse_dataset_working_copy(self.existing_dataset_dir)
            if working_dataset_dir:
                self.reuse_dataset_working_dir = working_dataset_dir
            loaded_ok = self._load_existing_dataset_into_memory(
                self.reuse_dataset_working_dir if self.reuse_dataset_working_dir else self.existing_dataset_dir
            )
            if loaded_ok:
                self._bootstrap_online_sam2_from_existing_dataset(self.gs_training_dir)
        
        # Register shutdown handler to clean up child processes
        rospy.on_shutdown(self._on_shutdown)

        # add service only after all runtime dependencies are ready
        rospy.loginfo("Adding services")
        self.addview_srv = rospy.Service("add_view", Trigger, self.addVisionCb)
        self.nbv_srv = rospy.Service("next_best_view", NBV, self.NextBestView)
        self.nbv_get_poses_srv = rospy.Service("get_poses", NBVPoses, self.getNBVPoses)
        self.nbv_get_poses_srv = rospy.Service("receive_nbv_scores", NBVResult, self.receiveNBVScoresGS)
        self.savemodel_srv = rospy.Service("save_model", SaveModel, self.saveModelCb)
        self.get_gs_data_dir_srv = rospy.Service("get_gs_data_dir", Trigger, self.getGSDataDirCb)

        rospy.loginfo("Vision Node Initialized")

    def _load_existing_dataset_into_memory(self, dataset_dir: str) -> bool:
        if len(dataset_dir.strip()) == 0:
            rospy.logwarn("reuse_existing_dataset=True but existing_dataset_dir is empty")
            return False

        dataset_dir = osp.abspath(osp.expanduser(dataset_dir))
        transforms_path = osp.join(dataset_dir, "transforms.json")
        if not osp.isfile(transforms_path):
            rospy.logwarn("Existing dataset missing transforms.json: %s", transforms_path)
            return False

        try:
            with open(transforms_path, "r") as file:
                data = json.load(file)
        except Exception as exc:
            rospy.logwarn("Failed to read existing transforms (%s): %s", transforms_path, str(exc))
            return False

        frames = data.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            rospy.logwarn("Existing dataset has no frames: %s", transforms_path)
            return False

        self.images = []
        self.depths = []
        self.realsense_depths = []
        self.monocular_depths = []
        self.poses = []

        loaded_count = 0
        for frame in frames:
            image_rel = frame.get("file_path")
            depth_rel = frame.get("depth_file_path")
            mde_rel = frame.get("mde_depth_file_path")
            transform_matrix = frame.get("transform_matrix")

            if image_rel is None or depth_rel is None or transform_matrix is None:
                continue

            image_path = osp.join(dataset_dir, image_rel)
            depth_path = osp.join(dataset_dir, depth_rel)
            mde_path = osp.join(dataset_dir, mde_rel) if mde_rel is not None else None

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            mde_u16 = cv2.imread(mde_path, cv2.IMREAD_UNCHANGED) if mde_path is not None else None

            if image is None or depth_u16 is None:
                continue

            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            depth = depth_u16.astype(np.float32) / 1000.0
            if mde_u16 is not None:
                monocular_depth = mde_u16.astype(np.float32) / 1000.0
            else:
                monocular_depth = depth.copy()

            pose = np.array(transform_matrix, dtype=np.float64)
            if pose.shape != (4, 4):
                continue

            self.images.append(image)
            self.depths.append(depth)
            self.realsense_depths.append(depth.copy())
            self.monocular_depths.append(monocular_depth)
            self.poses.append(pose)
            loaded_count += 1

        if loaded_count == 0:
            rospy.logwarn("Failed to load any frame from existing dataset: %s", dataset_dir)
            return False

        self.gs_training_dir = dataset_dir
        self.idx = loaded_count
        rospy.loginfo(
            "Loaded existing dataset for training bootstrap: %d frames from %s",
            loaded_count,
            dataset_dir,
        )
        return True

    def _prepare_reuse_dataset_working_copy(self, source_dataset_dir: str) -> str:
        if len(str(source_dataset_dir).strip()) == 0:
            rospy.logwarn("reuse_existing_dataset=True but existing_dataset_dir is empty")
            return ""

        src_dir = osp.abspath(osp.expanduser(source_dataset_dir))
        transforms_src = osp.join(src_dir, "transforms.json")
        if not osp.isfile(transforms_src):
            rospy.logwarn("Existing dataset missing transforms.json: %s", transforms_src)
            return ""

        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        dst_dir = osp.join(self.save_data_dir, date_str)
        suffix = 1
        while osp.exists(dst_dir):
            dst_dir = osp.join(self.save_data_dir, f"{date_str}-reuse-{suffix:02d}")
            suffix += 1

        try:
            os.makedirs(dst_dir, exist_ok=False)
            src_images_dir = osp.join(src_dir, "images")
            dst_images_dir = osp.join(dst_dir, "images")
            if osp.isdir(src_images_dir):
                shutil.copytree(src_images_dir, dst_images_dir)
            else:
                os.makedirs(dst_images_dir, exist_ok=True)

            src_masks_dir = osp.join(src_dir, "masks")
            dst_masks_dir = osp.join(dst_dir, "masks")
            if osp.isdir(src_masks_dir):
                shutil.copytree(src_masks_dir, dst_masks_dir)
            else:
                os.makedirs(dst_masks_dir, exist_ok=True)

            shutil.copy2(transforms_src, osp.join(dst_dir, "transforms.json"))
        except Exception as exc:
            rospy.logwarn(
                "Failed to create working copy for reused dataset (%s -> %s): %s",
                src_dir,
                dst_dir,
                str(exc),
            )
            return ""

        rospy.loginfo(
            "Prepared working dataset copy for reuse: %s -> %s",
            src_dir,
            dst_dir,
        )
        return dst_dir

    def _build_sam2_points_from_mask(self, mask_path: str):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        fruit_fg = mask >= 224
        leaf_fg = (mask >= 96) & (mask < 224)
        fg = fruit_fg | leaf_fg | (mask > 0)
        if not np.any(fg):
            return None

        ys, xs = np.where(fruit_fg if np.any(fruit_fg) else fg)
        center_idx = len(xs) // 2
        fruit_pos_pt = [float(xs[center_idx]), float(ys[center_idx])]

        all_ys, all_xs = np.indices(mask.shape)
        dist = (all_xs - fruit_pos_pt[0]) ** 2 + (all_ys - fruit_pos_pt[1]) ** 2
        bg = ~fg
        points = [fruit_pos_pt]
        labels = [1]
        class_ids = [1]

        if np.any(leaf_fg):
            l_ys, l_xs = np.where(leaf_fg)
            leaf_center_idx = len(l_xs) // 2
            points.append([float(l_xs[leaf_center_idx]), float(l_ys[leaf_center_idx])])
            labels.append(1)
            class_ids.append(2)

        if np.any(bg):
            bg_dist = np.where(bg, dist, -1.0)
            bg_flat_idx = int(np.argmax(bg_dist))
            bg_y, bg_x = np.unravel_index(bg_flat_idx, mask.shape)
            neg_pt = [float(bg_x), float(bg_y)]
            points.append(neg_pt)
            labels.append(0)
            class_ids.append(1)
        return {
            "points": np.array(points, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32),
            "class_ids": np.array(class_ids, dtype=np.int32),
        }

    def _bootstrap_online_sam2_from_existing_dataset(self, dataset_dir: str) -> None:
        dataset_dir = osp.abspath(osp.expanduser(dataset_dir))
        transforms_path = osp.join(dataset_dir, "transforms.json")
        if not osp.isfile(transforms_path):
            rospy.logwarn("Skip SAM2 bootstrap: transforms.json missing in %s", dataset_dir)
            return

        try:
            with open(transforms_path, "r") as file:
                data = json.load(file)
        except Exception as exc:
            rospy.logwarn("Skip SAM2 bootstrap: failed to parse %s (%s)", transforms_path, str(exc))
            return

        frames = data.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            rospy.logwarn("Skip SAM2 bootstrap: no frames in %s", transforms_path)
            return

        copied_frames = 0
        first_valid_mask = None
        for idx, frame in enumerate(frames):
            image_rel = frame.get("file_path")
            rs_depth_rel = frame.get("depth_file_path")
            mde_depth_rel = frame.get("mde_depth_file_path")
            mask_rel = frame.get("mask_path")

            if not image_rel:
                continue

            src_img = osp.join(dataset_dir, image_rel)
            dst_img = osp.join(self.online_sam2_images_dir, "{:04d}.png".format(idx))
            if osp.isfile(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                continue

            if rs_depth_rel:
                src_rs = osp.join(dataset_dir, rs_depth_rel)
                dst_rs = osp.join(self.online_sam2_images_dir, "{:04d}_realsense_depth.png".format(idx))
                if osp.isfile(src_rs):
                    shutil.copy2(src_rs, dst_rs)
            if mde_depth_rel:
                src_mde = osp.join(dataset_dir, mde_depth_rel)
                dst_mde = osp.join(self.online_sam2_images_dir, "{:04d}_mde_depth.png".format(idx))
                if osp.isfile(src_mde):
                    shutil.copy2(src_mde, dst_mde)
            if mask_rel:
                src_mask = osp.join(dataset_dir, mask_rel)
                dst_mask = osp.join(self.online_sam2_masks_dir, "{:04d}.png".format(idx))
                if osp.isfile(src_mask):
                    shutil.copy2(src_mask, dst_mask)
                    if first_valid_mask is None:
                        first_valid_mask = dst_mask
            copied_frames += 1

        points_labels_path = osp.join(self.online_sam2_dir, "sam2_points_labels.pkl")
        if first_valid_mask is not None and not osp.isfile(points_labels_path):
            points_obj = self._build_sam2_points_from_mask(first_valid_mask)
            if points_obj is not None:
                try:
                    with open(points_labels_path, "wb") as file:
                        pickle.dump(points_obj, file)
                    rospy.loginfo("Initialized SAM2 point prompts from existing mask: %s", first_valid_mask)
                except Exception as exc:
                    rospy.logwarn("Failed to save bootstrap SAM2 prompts (%s)", str(exc))

        rospy.loginfo(
            "Bootstrapped online SAM2 workspace from existing dataset: %d frames (%s)",
            copied_frames,
            dataset_dir,
        )

    def _on_shutdown(self):
        """Clean up child processes on ROS shutdown."""
        rospy.loginfo("Shutting down Vision Node...")
        if self.gs_model and hasattr(self.gs_model, 'training_process'):
            if self.gs_model.training_process is not None:
                rospy.loginfo(f"Terminating GS training process {self.gs_model.training_process.pid}...")
                try:
                    self.gs_model.training_process.terminate()
                    self.gs_model.training_process.wait(timeout=5)
                except Exception as e:
                    rospy.logwarn(f"Error terminating GS process: {e}")
                    try:
                        self.gs_model.training_process.kill()
                    except Exception as kill_err:
                        rospy.logerr(f"Failed to kill GS process: {kill_err}")
        rospy.loginfo("Vision Node shutdown complete")

    def _run_frames_sam2_annotation(self):
        frames_sam2_script = osp.join(self.script_dir, "frames_sam2.py")
        sam2_env_name = self._get_sam2_env_name()
        # Check if we already have points from first frame
        points_labels_path = osp.join(self.online_sam2_dir, "sam2_points_labels.pkl")
        is_first_frame = not osp.isfile(points_labels_path)
        
        if not is_first_frame:
            rospy.loginfo(f"Points already exist; auto-propagating without UI annotation")
        else:
            rospy.loginfo(f"First frame detected; showing image-click UI in {self.online_sam2_dir}")
        
        subprocess.run(
            [
                "conda",
                "run",
                "-n",
                sam2_env_name,
                "python",
                frames_sam2_script,
                "--data_dir",
                self.online_sam2_dir,
            ],
            check=True,
            env=self._sam2_subprocess_env(),
        )

    def _prepare_online_sam2_inputs(self, frame_idx: int, rgb: np.ndarray, rs_depth: np.ndarray, predicted_depth: np.ndarray):
        img_path = osp.join(self.online_sam2_images_dir, "{:04d}.png".format(frame_idx))
        rs_depth_path = osp.join(self.online_sam2_images_dir, "{:04d}_realsense_depth.png".format(frame_idx))
        mde_depth_path = osp.join(self.online_sam2_images_dir, "{:04d}_mde_depth.png".format(frame_idx))

        if rgb.ndim == 3:
            cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(img_path, rgb)
        cv2.imwrite(rs_depth_path, self._depth_to_uint16(rs_depth))
        cv2.imwrite(mde_depth_path, self._depth_to_uint16(predicted_depth))

        return img_path, rs_depth_path, mde_depth_path
        

    def convertPose2Numpy(pose:Union[PoseStamped, Pose]) -> np.ndarray:
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        c2w = np.eye(4)
        quat = sciR.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        return c2w
    
    def convertTransform2Numpy(transform:Union[TransformStamped, Transform]) -> np.ndarray:
        if isinstance(transform, TransformStamped):
            transform = transform.transform

        c2w = np.eye(4)
        quat = sciR.from_quat([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]

        return c2w
    
    def convertNumpy2PoseStamped(c2w:np.ndarray, frame_id="base_link") -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = c2w[0, 3]
        pose.pose.position.y = c2w[1, 3]
        pose.pose.position.z = c2w[2, 3]
        
        quat = sciR.from_matrix(c2w[:3, :3])
        pose.pose.orientation.x = quat.as_quat()[0]
        pose.pose.orientation.y = quat.as_quat()[1]
        pose.pose.orientation.z = quat.as_quat()[2]
        pose.pose.orientation.w = quat.as_quat()[3]
        
        return pose
    
    def saveModelCb(self, req) -> SaveModelResponse:
        """ Save Model Callback """
        res = SaveModelResponse()
        res.success = req.success
        gs_data_dir = self.gs_training_dir

        if self.reuse_existing_dataset and self.gs_training_dir is None:
            dataset_dir = self.reuse_dataset_working_dir
            if not dataset_dir:
                dataset_dir = self._prepare_reuse_dataset_working_copy(self.existing_dataset_dir)
                if dataset_dir:
                    self.reuse_dataset_working_dir = dataset_dir
            self._load_existing_dataset_into_memory(dataset_dir if dataset_dir else self.existing_dataset_dir)
            gs_data_dir = self.gs_training_dir
        
        if self.save_data:
            gs_data_dir = self.save_images()
        if gs_data_dir is None:
            res.success = False
            res.message = "No dataset directory available for training"
            return res

        if self.wait_for_offline_sam2:
            done_flag_path = osp.join(gs_data_dir, self.offline_sam2_done_flag)
            rospy.logwarn("Offline SAM2 wait is ENABLED; training is paused until done flag appears.")
            rospy.logwarn(
                f"Waiting for offline SAM2 completion flag: {done_flag_path}. "
                "Run offline script, then create this file to continue."
            )
            rate = rospy.Rate(1)
            while not rospy.is_shutdown() and not osp.isfile(done_flag_path):
                rate.sleep()
            if rospy.is_shutdown():
                res.success = False
                res.message = "ROS shutdown while waiting for offline SAM2 completion"
                return res
            rospy.loginfo(f"Detected offline SAM2 completion flag: {done_flag_path}")
            
        if self.should_collect_test_views:
            res.message = "Test views done."
            return res
        
        if self.gs_training and hasattr(self.gs_model, "is_training_alive"):
            try:
                if not self.gs_model.is_training_alive():
                    return_code = None
                    if hasattr(self.gs_model, "training_return_code"):
                        return_code = self.gs_model.training_return_code()
                    rospy.logwarn(
                        "GS process is no longer alive (return_code=%s). Will restart training.",
                        str(return_code),
                    )
                    self.gs_training = False
            except Exception as exc:
                rospy.logwarn(f"Failed to query GS training process state: {exc}")

        if self.gs_training:
            try:
                rospy.wait_for_service('continue_training', timeout=10.0)
                rospy.loginfo("Calling continue_training service")
                continue_training_srv = rospy.ServiceProxy('continue_training', Trigger)
                request = TriggerRequest()
                response = continue_training_srv(request)
                if response.success:
                    rospy.loginfo("Successfully called continue_training service")
                else:
                    rospy.logwarn("continue_training service returned failure; will restart training.")
                    self.gs_training = False
            except rospy.ROSException:
                rospy.logwarn("continue_training service unavailable after timeout; will restart training.")
                self.gs_training = False
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed with error {e}; will restart training.")
                self.gs_training = False
            
        rospy.loginfo("Saving Model...")
        
        # start training model
        if not self.gs_training:
            if not self.saveModel(gs_data_dir):
                res.success = False
                res.message = "Failed to start GS training process"
                return res
            
        
        res.message = "Success"
        return res
    
    def align_depth(self, depth: np.ndarray, predicted_depth: np.ndarray, 
                    rgb: np.ndarray, use_sam: bool = False, object_mask_path: str = None) -> np.ndarray:
        rs_depth = np.array(depth, copy=True)
        scale, offset = learn_scale_and_offset_raw(predicted_depth, rs_depth)
        depth_np = (scale * predicted_depth) + offset
        depth_np[depth_np < 0] = 0
        first_stage_depth = depth_np
        
        
        # perform SAM2 semantic alignment if use_sam is True
        if use_sam and self.use_sam:
            # Save the image, depth, and predicted depth
            img_path = osp.join(self.save_data_dir, "sam2_img.png")
            depth_path = osp.join(self.save_data_dir, "sam2_depth.png")
            mde_depth_path = osp.join(self.save_data_dir, "sam2_mde_depth.png")
            original_mde_depth_path = osp.join(self.save_data_dir, "sam2_original_mde_depth.png")
            
            if rgb.ndim == 3:
                cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(img_path, rgb)
            depth = self._depth_to_uint16(rs_depth)
            predicted_depth_uint16 = self._depth_to_uint16(predicted_depth)
            cv2.imwrite(depth_path, depth)
            cv2.imwrite(mde_depth_path, predicted_depth_uint16)
            if object_mask_path:
                cv2.imwrite(original_mde_depth_path, predicted_depth_uint16)
            
            run_sam2_script = osp.join(self.script_dir, "run_sam2.py")
            try:
                # Use conda run to properly activate the sam2 environment
                sam2_env_name = self._get_sam2_env_name()
                cmd = [
                    "conda",
                    "run",
                    "-n",
                    sam2_env_name,
                    "python",
                    run_sam2_script,
                    "--img_path",
                    img_path,
                    "--real_depth",
                    depth_path,
                    "--mde_depth_path",
                    mde_depth_path,
                ]
                # Add object mask path if provided
                if object_mask_path and osp.isfile(object_mask_path):
                    cmd.extend(["--object_mask_path", object_mask_path])
                    cmd.extend(["--original_mde_depth_path", original_mde_depth_path])
                    rospy.loginfo(f"Using object mask from: {object_mask_path}")
                
                rospy.loginfo(f"Running SAM2 with command: {' '.join(cmd)}")
                subprocess.run(
                    cmd,
                    check=True,
                    env=self._sam2_subprocess_env(),
                )
                aligned_path = osp.join(self.save_data_dir, "mde_depth_aligned.png")
                aligned_depth = cv2.imread(aligned_path, cv2.IMREAD_UNCHANGED)
                if aligned_depth is not None:
                    depth_np = aligned_depth / 1000.0
                    rospy.loginfo("SAM2 depth alignment completed successfully")
                else:
                    rospy.logwarn(f"SAM2 output not found at {aligned_path}; using first-stage aligned depth.")
                    depth_np = first_stage_depth
            except subprocess.CalledProcessError as e:
                rospy.logwarn(f"SAM2 alignment failed ({e}); using first-stage aligned depth.")
                depth_np = first_stage_depth
            except FileNotFoundError as e:
                rospy.logwarn(f"SAM2 runner not found ({e}); using first-stage aligned depth.")
                depth_np = first_stage_depth
        
        # remove bad values
        depth_np[depth_np < 0] = 0
        
        rospy.loginfo(f"Scale: {scale}, Offset: {offset}")
        return depth_np, rs_depth, first_stage_depth

    def addVisionCb(self, req) -> TriggerResponse:
        """ AddVision Cb 
        
        return TriggerResponse:
            bool success
            message:
                form: Error Code X: [Error Message]
                X=1 -> Training thread is still running
                X=2 -> Unsupported image encoding
                X=3 -> Failed to lookup transform from camera to base_link
        """
        
        res = TriggerResponse()
        res.success = True

        if not hasattr(self, "tfBuffer") or self.tfBuffer is None:
            res.success = False
            res.message = "Error Code 4: Vision node is not fully initialized"
            rospy.logwarn("add_view requested before VisionNode initialization completed")
            return res

        rospy.sleep(3)

        # grab synchronized RGB/depth messages (best-effort)
        img, depth_img, sync_delta = self._wait_for_synced_rgb_depth()
        img_np = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
        img_np = self._color_msg_to_rgb(img, img_np)
        if sync_delta is not None:
            if sync_delta > self.rgb_depth_sync_tolerance_s:
                rospy.logwarn(
                    "RGB/depth timestamp gap is %.4fs (> %.4fs tolerance).",
                    float(sync_delta),
                    float(self.rgb_depth_sync_tolerance_s),
                )
            else:
                rospy.loginfo("RGB/depth timestamp gap: %.4fs", float(sync_delta))
        
        try:
            lookup_time = img.header.stamp if hasattr(img, "header") else rospy.Time(0)
            transform: TransformStamped = self.tfBuffer.lookup_transform(
                self.base_frame,
                self.camera_link,
                lookup_time,
                rospy.Duration(2.0),
            )
        except Exception as e:
            rospy.logwarn(f"Time-stamped transform lookup failed ({e}); fallback to latest transform.")
            try:
                transform = self.tfBuffer.lookup_transform(
                    self.base_frame,
                    self.camera_link,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
            except Exception as e2:
                rospy.logerr(f"Failed to lookup transform from camera to base_link: {e2}")
                res.success = False
                res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
                return res

        # grab the depth image message (retry a few times to avoid transient empty depth frames)
        depth = None
        nonzero_depth = np.array([])
        for attempt_idx in range(max(1, self.depth_retry_count)):
            if attempt_idx > 0:
                depth_img = rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)
            raw_depth = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
            candidate_depth = self._depth_msg_to_meters(depth_img, raw_depth)
            candidate_depth[candidate_depth > 20] = 0
            candidate_nonzero = candidate_depth[candidate_depth > 0]
            if candidate_nonzero.size > 0:
                depth = candidate_depth
                nonzero_depth = candidate_nonzero
                if attempt_idx > 0:
                    rospy.loginfo(
                        f"Recovered valid depth on retry {attempt_idx + 1}/{self.depth_retry_count}"
                    )
                break
            rospy.logwarn(
                f"Depth frame retry {attempt_idx + 1}/{self.depth_retry_count}: no valid (>0m) pixels"
            )

        if depth is None:
            depth = candidate_depth

        if nonzero_depth.size > 0:
            rospy.loginfo(
                f"Depth topic={self.DEPTH_CAMERA_TOPIC} encoding={depth_img.encoding} "
                f"range_m=[{float(nonzero_depth.min()):.4f}, {float(nonzero_depth.max()):.4f}]"
            )
        else:
            rospy.logwarn(
                f"Depth topic={self.DEPTH_CAMERA_TOPIC} encoding={depth_img.encoding} has no valid (>0m) pixels"
            )
        
        if self.camera_color_frame == self.camera_depth_frame:
            realsense_depth = depth
            rospy.loginfo("Depth and color frames are identical; skipping depth reprojection")
        else:
            new_intrinsics = np.array(self.color_cam_info.K)
            new_intrinsics_tup = (new_intrinsics[0], new_intrinsics[4], new_intrinsics[2], new_intrinsics[5])
            
            # convert instrinsics
            depth = convert_intrinsics(depth, new_size=(self.color_cam_info.width, self.color_cam_info.height), new_intrinsics=new_intrinsics_tup)
            
            try:
                cam2cam_transform: TransformStamped = self.tfBuffer.lookup_transform(
                    self.camera_color_frame,
                    self.camera_depth_frame,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
            except Exception as e:
                rospy.logerr(f"Failed to lookup transform from camera_color_frame to camera_depth_frame: {e}")
                res.success = False
                res.message = "Error Code 3: Failed to lookup transform between camera frames"
                return res
            cam2cam_transform = VisionNode.convertTransform2Numpy(cam2cam_transform)
            K = np.array(self.depth_cam_info.K).reshape(3, 3)
            realsense_depth = warp_image(depth, K, cam2cam_transform[:3, :3], cam2cam_transform[:3, 3])

        realsense_nonzero = realsense_depth[realsense_depth > 0]
        if realsense_nonzero.size > 0:
            rospy.loginfo(
                f"Realsense processed depth range_m=[{float(realsense_nonzero.min()):.4f}, {float(realsense_nonzero.max()):.4f}]"
            )
        else:
            rospy.logwarn("Realsense processed depth has no valid (>0m) pixels")
        has_valid_realsense_depth = realsense_nonzero.size > 0
        # run MDE to get the depth image
        output = self.monocular_depth(img_np)
        predicted_depth = output['depth']
        predicted_nonzero = predicted_depth[predicted_depth > 0]
        if predicted_nonzero.size > 0:
            rospy.loginfo(
                f"MDE({self.monocular_depth.model_name}) range_m="
                f"[{float(predicted_nonzero.min()):.4f}, {float(predicted_nonzero.max()):.4f}]"
            )
        else:
            rospy.logwarn(f"MDE({self.monocular_depth.model_name}) produced all-zero depth")
        
        frame_idx = len(self.images)
        full_mde_path = osp.join(self.save_data_dir, f"mde_depth_img{self.idx}.png")
        full_rs_path = osp.join(self.save_data_dir, f"rs_depth_img{self.idx}.png")
        self.idx += 1
        
        cv2.imwrite(full_mde_path, self._depth_to_uint16(predicted_depth))
        
        # 先写入在线 SAM2 数据目录，再做 mask 标注/传播
        _, _, _ = self._prepare_online_sam2_inputs(frame_idx, img_np, realsense_depth, predicted_depth)

        object_mask_path = osp.join(self.online_sam2_masks_dir, "{:04d}.png".format(frame_idx))
        try:
            self._run_frames_sam2_annotation()
        except subprocess.CalledProcessError as e:
            rospy.logwarn(f"frames_sam2 failed ({e}); fallback to first-stage depth alignment.")

        if osp.isfile(object_mask_path) and has_valid_realsense_depth:
            # 每个视角采集时即完成 SAM2 深度对齐，避免 save_images() 阶段重复跑
            depth_np, rs_depth, first_stage_depth = self.align_depth(
                realsense_depth,
                predicted_depth,
                img_np,
                use_sam=True,
                object_mask_path=object_mask_path,
            )
        else:
            if not has_valid_realsense_depth:
                rospy.logwarn("Skipping SAM2 alignment for this view because depth has no valid pixels.")
            rospy.logwarn(f"Mask not found for frame {frame_idx}: {object_mask_path}; using first-stage alignment only.")
            depth_np, rs_depth, first_stage_depth = self.align_depth(realsense_depth, predicted_depth, img_np, use_sam=False)
        
        cv2.imwrite(full_rs_path, self._depth_to_uint16(rs_depth))
        

        if res.success:
            c2w = VisionNode.convertTransform2Numpy(transform)
            self.poses.append(c2w)
            self.images.append(img_np)
            self.depths.append(depth_np)
            self.realsense_depths.append(rs_depth)
            self.monocular_depths.append(predicted_depth)
            
            res.message = "Success"
            rospy.loginfo(f"Added view to the dataset with {len(self.images)} images")
        return res
    
    def getGSDataDirCb(self, req: TriggerRequest) -> TriggerResponse:
        """ Get GS Data Dir Callback """
        res = TriggerResponse()
        res.success = True
        res.message = self.gs_data_dir
        return res

    def getNBVPoses(self, req: NBVPosesRequest) -> NBVPosesResponse:
        """ Get Next Best View Poses to send to GS

        Args:
            req (NBVPosesRequest): Request to get the next best view poses
        Returns:
            NBVPosesResponse: _description_
        """
        response = NBVPosesResponse()
        
        if len(self.poses_for_nbv) == 0:
            # if there are no poses for next best view, return false so that GS continues training as is.
            response.success = False
            response.message = "No poses for Next Best View"
            return response
         
        response.success = True
        response.message = f"Sent {len(self.poses_for_nbv)} poses for Next Best View to GS."
        response.poses = self.poses_for_nbv
        
        return response
    
    def receiveNBVScoresGS(self, req: NBVResultRequest) -> NBV:
        """ 
        Receive the NBV Scores from GS 
        
        """
        response = NBVResultResponse()
        if len(req.scores) == 0:
            response.success = False
            response.message = "No scores provided. GS will continue training."
            return response
        
        response.success = True
        response.message = "Successfully received scores from GS"
        self.scores = req.scores
        self.done = True
        return response

    def NextBestView(self, req:NBVRequest) -> NBVResponse:
        """ Next Best View Service Callback 
            Waits for GS to be trained, then proceeds
        """
        poses = req.poses
        sensor = req.sensor
        res = NBVResponse()

        if len(poses) == 0:
            res.success = True
            res.message = "No-Op -- No poses provided"
            res.scores = []
            return res
        
        scores = self.EvaluatePoses(poses, sensor)
        if scores is None:
            scores = np.zeros((len(poses),), dtype=np.float64)

        # return response
        res.success = True
        res.message = "Success"
        res.scores = list(scores)
        return res
    
    def save_images(self):
        """ Save the captured images as a NeRF Synthetic Dataset format
            When new views exist, """
        # get camera info
        # get the date format in Year-Month-Day-Hour-Minute-Second
        
        if self.gs_training_dir is None:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d-%H-%M-%S")

            data_base_dir = osp.join(self.save_data_dir, date_str)
            os.makedirs(data_base_dir, exist_ok=True)
            os.makedirs(osp.join(data_base_dir, "images"), exist_ok=True)
            self.gs_training_dir = data_base_dir
            os.makedirs(osp.join(data_base_dir, "masks"), exist_ok=True)
        

        cam_info:CameraInfo = rospy.wait_for_message(self.cam_info_topic, CameraInfo)

        cam_K = np.array(cam_info.K).reshape(3, 3)
        fovx = 2 * np.arctan(cam_K[0, 2] / cam_K[0, 0])
        fovy = 2 * np.arctan(cam_K[1, 2] / cam_K[1, 1])
        focal_x = cam_K[0, 0]
        focal_y = cam_K[1, 1]

        cam_height = cam_info.height
        cam_width = cam_info.width

        json_txt = {
            "w": cam_width,
            "h": cam_height,
            "fl_x": focal_x,
            "fl_y": focal_y,
            "cx": cam_K[0, 2],
            "cy": cam_K[1, 2],
            "camera_angle_x": fovx,
            "camera_angle_y": fovy,
            "frames": [],
        }

        for img_idx, (pose, image, depth, rs_depth, monocular_depth) in enumerate(zip(self.poses, self.images, self.depths, self.realsense_depths, self.monocular_depths)):
            # save the image
            image_path = osp.join(self.gs_training_dir, "images", "{:04d}.png".format(img_idx))
            depth_path = osp.join(self.gs_training_dir, "images", "{:04d}_depth.png".format(img_idx))
            realsense_depth_path = osp.join(self.gs_training_dir, "images", "{:04d}_realsense_depth.png".format(img_idx))
            mde_depth_path = osp.join(self.gs_training_dir, "images", "{:04d}_mde_depth.png".format(img_idx))
            
            depth = self._depth_to_uint16(depth)
            realsense_depth = self._depth_to_uint16(rs_depth)
            monocular_depth = self._depth_to_uint16(monocular_depth)

            if image.ndim == 3:
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(image_path, image)
            cv2.imwrite(depth_path, depth)
            cv2.imwrite(realsense_depth_path, realsense_depth)
            cv2.imwrite(mde_depth_path, monocular_depth)

            # save the pose
            pose_list = pose.tolist()
            pose_info = {
                "file_path": osp.join("images", "{:04d}.png".format(img_idx)),
                "depth_file_path": osp.join("images", "{:04d}_depth.png".format(img_idx)),
                "mde_depth_file_path": osp.join("images", "{:04d}_mde_depth.png".format(img_idx)),
                "mask_path": osp.join("masks", "{:04d}.png".format(img_idx)),
                "transform_matrix": pose_list,
            }
            json_txt["frames"].append(pose_info)

        # dump to json file
        json_file = osp.join(self.gs_training_dir, "transforms.json")
        with open(json_file, "w") as f:
            json.dump(json_txt, f, indent=4)
            
        rospy.loginfo(f"Saved all images to {self.gs_training_dir}.")

        masks_dir = osp.join(self.gs_training_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        for img_idx in range(len(self.images)):
            mask_path = osp.join(masks_dir, "{:04d}.png".format(img_idx))
            online_mask_path = osp.join(self.online_sam2_masks_dir, "{:04d}.png".format(img_idx))
            if osp.isfile(online_mask_path):
                shutil.copy2(online_mask_path, mask_path)
            elif not osp.isfile(mask_path):
                fallback_mask = np.ones((cam_height, cam_width), dtype=np.uint8) * 255
                cv2.imwrite(mask_path, fallback_mask)
                rospy.logwarn(f"Missing mask {mask_path}; wrote fallback all-valid mask.")
        
        return self.gs_training_dir
    
    def invertTransform(self, transform:np.ndarray) -> np.ndarray:
        """ Invert the transformation matrix """
        inv_transform = np.eye(4)
        inv_transform[:3, :3] = transform[:3, :3].T
        inv_transform[:3, 3] = -inv_transform[:3, :3] @ transform[:3, 3]
        return inv_transform

    def EvaluatePoses(self, poses:List[PoseStamped], sensor='rgb') -> np.ndarray:
        """ 
        Evaluate poses. Waits a few minutes for GS to reach 2k steps, then requests a pose from GS with FisherRF
        """
        self.done = False
        sensor_poses: List[PoseStamped] = []
        link_name = self.camera_link if sensor == 'rgb' else self.touch_link
        
        try:
            transform: TransformStamped = self.tfBuffer.lookup_transform(self.ee_link, link_name, rospy.Time())
        except Exception as e:
            rospy.logerr(f"Failed to lookup transform from camera to end_effector_link: {e}")
            return None
        
        for pose in poses:
            try:
                ee_pose = VisionNode.convertPose2Numpy(pose)
                ee_to_sensor = VisionNode.convertTransform2Numpy(transform)
                
                # get the camera pose
                sensor_pose = ee_pose @ ee_to_sensor
                
                # convert to pose
                new_pose = VisionNode.convertNumpy2PoseStamped(sensor_pose, frame_id=self.base_frame)
                sensor_poses.append(new_pose)
            except Exception as e:
                rospy.logerr(f"Failed to transform pose: {e}")
                return None
        
        self.poses_for_nbv = sensor_poses
        self.scores = []
        
        rate = rospy.Rate(1)  # 1 Hz
        wait_start = rospy.Time.now().to_sec()
        try:
            # loop until GS hits 2k steps and requests a pose
            while not rospy.is_shutdown() and not self.done:
                if self.gs_training and hasattr(self.gs_model, "is_training_alive"):
                    if not self.gs_model.is_training_alive():
                        return_code = None
                        if hasattr(self.gs_model, "training_return_code"):
                            return_code = self.gs_model.training_return_code()
                        rospy.logerr(f"GS training process exited early with code {return_code}; stopping NBV wait.")
                        self.done = True
                        break
                rospy.loginfo("GS running...")
                if rospy.Time.now().to_sec() - wait_start > self.nbv_wait_timeout_s:
                    rospy.logwarn(f"NBV score wait timed out after {self.nbv_wait_timeout_s}s; returning zeros.")
                    self.done = True
                    break
                rate.sleep()
        except rospy.ROSInterruptException:
            rospy.logwarn("ROS interrupt while waiting NBV scores; returning zeros.")
            self.done = True
        except Exception as e:
            rospy.logerr(f"Unexpected error while waiting NBV scores: {e}; returning zeros.")
            self.done = True
        rospy.loginfo("GS Done")

        if len(self.scores) == 0:
            return np.zeros((len(poses),), dtype=np.float64)
        if len(self.scores) != len(poses):
            rospy.logwarn("NBV score length mismatch; padding/truncating to candidate count.")
            scores = np.zeros((len(poses),), dtype=np.float64)
            copy_n = min(len(self.scores), len(poses))
            scores[:copy_n] = np.array(self.scores[:copy_n], dtype=np.float64)
            return scores
        
        # result is obtained, return the scores, which should be populated
        return np.array(self.scores)

    def saveModel(self, gs_data_dir: str):
        """ Save the model  """
        # send request to NS to continue training
        self.gs_training = True
        self.gs_data_dir
        
        # call training in data_dir
        return self.gs_model.start_training(gs_data_dir, steps=self.gs_max_iterations)
        

if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()