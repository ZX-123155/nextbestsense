#!/usr/bin/env python3

import json
import os
import threading
from typing import Dict, List

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as SciR
from sensor_msgs.msg import CameraInfo, Image


class SimpleCaptureNode(object):
    def __init__(self) -> None:
        rospy.init_node("simple_capture_node")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.camera_link = rospy.get_param("~camera_link", "camera_link")
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw")
        self.rgb_info_topic = rospy.get_param("~rgb_info_topic", "/camera/color/camera_info")
        self.save_root = os.path.expanduser(rospy.get_param("~save_root", "~/gazebopicture"))
        self.captures_per_run = int(rospy.get_param("~captures_per_run", "5"))

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._image_lock = threading.Lock()
        self._latest_rgb_msg = None
        self._latest_depth_msg = None

        os.makedirs(self.save_root, exist_ok=True)

        self.depth_topic = self._resolve_topic(
            preferred_topic=self.depth_topic,
            fallback_topics=["/camera/depth/image_rect_raw", "/camera/depth/image_raw"],
            topic_type="sensor_msgs/Image",
        )

        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)

    def _next_run_dir(self) -> str:
        run_ids: List[int] = []
        for name in os.listdir(self.save_root):
            path = os.path.join(self.save_root, name)
            if os.path.isdir(path) and name.isdigit():
                run_ids.append(int(name))
        next_id = 1 if len(run_ids) == 0 else (max(run_ids) + 1)
        return os.path.join(self.save_root, str(next_id))

    def _lookup_camera_pose_matrix(self, source_frame: str) -> List[List[float]]:
        candidates: List[str] = []
        for frame in [source_frame, self.camera_link, "ee_camera_optical_frame"]:
            frame_name = frame.strip() if isinstance(frame, str) else ""
            if frame_name and frame_name not in candidates:
                candidates.append(frame_name)

        last_exc = None
        transform = None
        for frame in candidates:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    frame,
                    rospy.Time(0),
                    rospy.Duration(1.0),
                )
                if frame != source_frame:
                    rospy.logwarn("TF frame %s unavailable, using %s", source_frame, frame)
                break
            except Exception as exc:
                last_exc = exc

        if transform is None:
            raise last_exc

        pose = np.eye(4, dtype=np.float64)
        t = transform.transform.translation
        q = transform.transform.rotation
        pose[:3, 3] = np.array([float(t.x), float(t.y), float(t.z)], dtype=np.float64)
        pose[:3, :3] = SciR.from_quat([float(q.x), float(q.y), float(q.z), float(q.w)]).as_matrix()
        return pose.tolist()

    @staticmethod
    def _safe_cam_distortion(cam_info: CameraInfo, idx: int, default: float = 0.0) -> float:
        if idx < len(cam_info.D):
            return float(cam_info.D[idx])
        return default

    def _build_transforms_header(self, cam_info: CameraInfo) -> Dict:
        return {
            "w": int(cam_info.width),
            "h": int(cam_info.height),
            "fl_x": float(cam_info.K[0]),
            "fl_y": float(cam_info.K[4]),
            "cx": float(cam_info.K[2]),
            "cy": float(cam_info.K[5]),
            "k1": self._safe_cam_distortion(cam_info, 0),
            "k2": self._safe_cam_distortion(cam_info, 1),
            "p1": self._safe_cam_distortion(cam_info, 2),
            "p2": self._safe_cam_distortion(cam_info, 3),
            "camera_model": "OPENCV",
            "frames": [],
            "applied_transform": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }

    def _rgb_cb(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_rgb_msg = msg

    def _depth_cb(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_depth_msg = msg

    def _wait_latest_pair(self, timeout_sec: float = 5.0):
        deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
        rate = rospy.Rate(100)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self._image_lock:
                rgb_msg = self._latest_rgb_msg
                depth_msg = self._latest_depth_msg
            if rgb_msg is not None and depth_msg is not None:
                return rgb_msg, depth_msg
            rate.sleep()
        raise rospy.ROSException(
            "timeout exceeded while waiting for cached messages on topics {} and {}".format(
                self.rgb_topic, self.depth_topic
            )
        )

    def _resolve_topic(self, preferred_topic: str, fallback_topics: List[str], topic_type: str) -> str:
        requested = preferred_topic.strip()
        candidates: List[str] = []
        for topic in [requested] + fallback_topics:
            if topic and topic not in candidates:
                candidates.append(topic)

        try:
            published_topics = dict(rospy.get_published_topics())
        except Exception:
            published_topics = {}

        for topic in candidates:
            if published_topics.get(topic) == topic_type:
                if topic != requested:
                    rospy.logwarn("Requested topic %s not available, using %s", requested, topic)
                else:
                    rospy.loginfo("Using topic %s", topic)
                return topic

        rospy.logwarn("No published topic found among %s, keeping requested topic %s", candidates, requested)
        return requested

    def _capture_once(self, idx: int, rgb_dir: str, depth_dir: str) -> Dict:
        rgb_msg, depth_msg = self._wait_latest_pair(timeout_sec=5.0)

        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        if depth_img.dtype != np.uint16:
            if np.issubdtype(depth_img.dtype, np.floating):
                depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
                depth_img = np.clip(depth_img * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
            else:
                depth_img = np.clip(depth_img, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        rgb_path = os.path.join(rgb_dir, f"{idx:04d}.png")
        depth_path = os.path.join(depth_dir, f"{idx:04d}.png")
        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)

        frame_entry = {
            "file_path": f"rgb_imgs/{idx:04d}.png",
            "transform_matrix": self._lookup_camera_pose_matrix(rgb_msg.header.frame_id),
            "colmap_im_id": int(idx),
            "depth_file_path": f"depth_imgs/{idx:04d}.png",
        }
        return frame_entry

    def run(self) -> None:
        run_dir = self._next_run_dir()
        rgb_dir = os.path.join(run_dir, "rgb_imgs")
        depth_dir = os.path.join(run_dir, "depth_imgs")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        rospy.loginfo("Capture run directory: %s", run_dir)
        rospy.loginfo("Press 'c' + Enter to capture, 'q' + Enter to quit early")

        cam_info: CameraInfo = rospy.wait_for_message(self.rgb_info_topic, CameraInfo, timeout=5.0)
        self._wait_latest_pair(timeout_sec=5.0)
        transforms = self._build_transforms_header(cam_info)

        for i in range(1, self.captures_per_run + 1):
            if rospy.is_shutdown():
                break

            while True:
                key = input(f"[{i}/{self.captures_per_run}] input c to capture, q to quit: ").strip().lower()
                if key in ["c", "q"]:
                    break

            if key == "q":
                rospy.logwarn("User requested exit before completing captures")
                break

            try:
                frame_entry = self._capture_once(i, rgb_dir, depth_dir)
                transforms["frames"].append(frame_entry)
                rospy.loginfo("Captured %d/%d", i, self.captures_per_run)
            except Exception as exc:
                rospy.logerr("Capture %d failed: %s", i, str(exc))

        transforms_path = os.path.join(run_dir, "transforms.json")
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)

        rospy.loginfo("Saved %d captures to %s", len(transforms["frames"]), run_dir)


if __name__ == "__main__":
    node = SimpleCaptureNode()
    node.run()
