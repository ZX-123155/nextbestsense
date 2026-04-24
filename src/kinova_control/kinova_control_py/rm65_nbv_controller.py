#!/usr/bin/env python3

import json
import os.path as osp
import sys
from typing import List, Optional, Tuple

import moveit_commander
import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as sciR

from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerRequest
from gaussian_splatting.srv import NBV, NBVRequest, NBVResponse, SaveModel, SaveModelRequest

from kinova_control_py.pose_util import RandomPoseGenerator


class RM65NBVController(object):
    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def __init__(self) -> None:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("rm65_nbv_controller")
        rospy.loginfo("Initializing RM65 NBV controller")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.pose_solver_ee_link = rospy.get_param("~pose_solver_ee_link", "Link6")
        self.home_named_target = rospy.get_param("~home_named_target", "zero")
        self.arm_group_name = rospy.get_param("~arm_group_name", "arm")

        self.starting_views = int(rospy.get_param("~starting_views", "8"))
        self.candidate_views_per_round = int(rospy.get_param("~candidate_views_per_round", "8"))
        self.nbv_max_rounds = int(rospy.get_param("~nbv_max_rounds", "5"))
        self.nbv_start_step = int(rospy.get_param("~nbv_start_step", "5000"))
        self.nbv_interval_steps = int(rospy.get_param("~nbv_interval_steps", "1000"))

        self.object_center = np.array(rospy.get_param("~object_center", [0.6, 0.0, 0.3]), dtype=float)
        self.sample_radius_min = float(rospy.get_param("~sample_radius_min", "0.12"))
        self.sample_radius_max = float(rospy.get_param("~sample_radius_max", "0.22"))
        self.max_pose_generation_attempts = int(rospy.get_param("~max_pose_generation_attempts", "500"))
        self.num_poses = int(rospy.get_param("~num_candidate_poses", self.candidate_views_per_round))
        self.sensor_type = rospy.get_param("~sensor_type", "rgb")
        self.max_reach_m = float(rospy.get_param("~max_reach_m", "0.72"))
        self.starting_views_mode = rospy.get_param("~starting_views_mode", "semicircle")
        self.starting_semicircle_radius = float(
            rospy.get_param("~starting_semicircle_radius", (self.sample_radius_min + self.sample_radius_max) * 0.5)
        )
        self.starting_semicircle_phi = float(rospy.get_param("~starting_semicircle_phi", str(np.pi / 6.0)))
        self.starting_semicircle_theta_min = float(
            rospy.get_param("~starting_semicircle_theta_min", str(-np.pi / 2.0))
        )
        self.starting_semicircle_theta_max = float(
            rospy.get_param("~starting_semicircle_theta_max", str(np.pi / 2.0))
        )
        self.require_back_start_view = bool(rospy.get_param("~require_back_start_view", True))
        self.require_low_side_start_view = bool(rospy.get_param("~require_low_side_start_view", True))
        self.starting_low_side_phi = float(rospy.get_param("~starting_low_side_phi", "1.95"))
        self.strict_required_start_views = bool(rospy.get_param("~strict_required_start_views", False))
        self.required_anchor_radius_min = float(
            rospy.get_param("~required_anchor_radius_min", self.sample_radius_min)
        )
        self.required_anchor_radius_max = float(
            rospy.get_param("~required_anchor_radius_max", self.sample_radius_max)
        )
        self.move_max_velocity_scaling = float(rospy.get_param("~move_max_velocity_scaling", "0.3"))
        self.move_max_acceleration_scaling = float(rospy.get_param("~move_max_acceleration_scaling", "0.3"))
        self.use_random_nbv = bool(rospy.get_param("~use_random_nbv", False))
        self.nbv_filtered_score = float(rospy.get_param("~nbv_filtered_score", -1e12))
        self.nbv_filtered_eps = float(rospy.get_param("~nbv_filtered_eps", 1e-6))
        self.nbv_candidate_growth = int(rospy.get_param("~nbv_candidate_growth", "5"))
        self.nbv_candidate_retry_max = int(rospy.get_param("~nbv_candidate_retry_max", "200"))
        self.nbv_min_valid_candidates = int(rospy.get_param("~nbv_min_valid_candidates", "2"))
        self.ik_cache_size = int(rospy.get_param("~ik_cache_size", "200"))
        self.use_fixed_starting_views = bool(rospy.get_param("~use_fixed_starting_views", False))
        self.fixed_starting_views_path = str(rospy.get_param("~fixed_starting_views_path", ""))
        self.fixed_starting_views_count = int(rospy.get_param("~fixed_starting_views_count", self.starting_views))
        self.fixed_starting_sensor_link = str(rospy.get_param("~fixed_starting_sensor_link", "ee_camera_optical_frame"))
        self.fixed_starting_views_strict = bool(rospy.get_param("~fixed_starting_views_strict", True))
        self.reuse_existing_dataset = self._parse_bool(rospy.get_param("~reuse_existing_dataset", True))
        self.existing_dataset_dir = str(rospy.get_param("~existing_dataset_dir", "/home/ras/NextBestSense/data/2026-04-22-11-36-13"))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot = moveit_commander.RobotCommander("robot_description")
        self.arm_group = moveit_commander.MoveGroupCommander(self.arm_group_name, ns=rospy.get_namespace())
        self.arm_group.set_max_acceleration_scaling_factor(self.move_max_acceleration_scaling)
        self.arm_group.set_max_velocity_scaling_factor(self.move_max_velocity_scaling)

        self.pose_generator = RandomPoseGenerator(
            base_name=self.base_frame,
            ee_name=self.pose_solver_ee_link,
            cache_size=self.ik_cache_size,
        )

        rospy.loginfo("Waiting for services: /add_view /next_best_view /save_model")
        rospy.wait_for_service("/add_view")
        rospy.wait_for_service("/next_best_view")
        rospy.wait_for_service("/save_model")

        self.add_view_client = rospy.ServiceProxy("/add_view", Trigger)
        self.nbv_client = rospy.ServiceProxy("/next_best_view", NBV)
        self.save_model_client = rospy.ServiceProxy("/save_model", SaveModel)

        self._log_workspace_reachability()

    def _log_workspace_reachability(self) -> None:
        center_dist = float(np.linalg.norm(self.object_center))
        min_candidate_dist = max(0.0, center_dist - self.sample_radius_max)
        max_candidate_dist = center_dist + self.sample_radius_max

        rospy.loginfo(
            "Object center distance=%.3fm, candidate shell=[%.3f, %.3f]m, configured max_reach=%.3fm",
            center_dist,
            min_candidate_dist,
            max_candidate_dist,
            self.max_reach_m,
        )

        if min_candidate_dist > self.max_reach_m:
            rospy.logerr(
                "Candidate shell is outside likely RM65 reach. Reduce object_center distance or radius. "
                "Example: object_center_x in [0.40, 0.60], sample_radius in [0.10, 0.20]."
            )

    def _pose_looks_at_center(self, pose: np.ndarray, max_lateral_offset: float = 0.04) -> bool:
        vec = self.object_center - np.asarray(pose[:3], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return False
        vec = vec / norm
        quat = np.asarray(pose[3:7], dtype=float)
        if np.linalg.norm(quat) < 1e-6:
            return False
        forward = sciR.from_quat(quat).as_matrix()[:, 2]
        cos_angle = float(np.clip(np.dot(vec, forward), -1.0, 1.0))
        lateral_error = float(np.linalg.norm(np.cross(vec, forward)))
        return cos_angle > 0.90 and lateral_error < max_lateral_offset

    def _enforce_look_at_center(self, pose: np.ndarray) -> np.ndarray:
        fixed_pose = np.asarray(pose, dtype=float).copy()
        if fixed_pose.shape[0] != 7:
            return fixed_pose

        position = fixed_pose[:3]
        to_center = self.object_center - position
        norm = float(np.linalg.norm(to_center))
        if norm < 1e-6:
            return fixed_pose

        z_axis = to_center / norm
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(z_axis, up_hint))) > 0.98:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=float)

        x_axis = np.cross(up_hint, z_axis)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-6:
            return fixed_pose
        x_axis = x_axis / x_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)

        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        fixed_pose[3:7] = sciR.from_matrix(rotation).as_quat()
        return fixed_pose

    def _send_req(self, client, req, name: str):
        try:
            return client(req)
        except rospy.ServiceException as exc:
            rospy.logerr("Service %s failed: %s", name, str(exc))
            return None

    def _plan_joint_target(self, joints) -> bool:
        try:
            plan_result = self.arm_group.plan(joints)
        except Exception:
            return False

        if isinstance(plan_result, tuple):
            if len(plan_result) > 0 and isinstance(plan_result[0], bool):
                return bool(plan_result[0])
            return False

        traj = getattr(plan_result, "joint_trajectory", None)
        return traj is not None and len(traj.points) > 0

    def reach_named_position(self, target_name: str) -> bool:
        self.arm_group.set_named_target(target_name)
        plan_result = self.arm_group.plan()
        if isinstance(plan_result, tuple):
            success = bool(plan_result[0])
            trajectory = plan_result[1] if len(plan_result) > 1 else None
            if not success or trajectory is None:
                return False
            return bool(self.arm_group.execute(trajectory, wait=True))

        return bool(self.arm_group.go(wait=True))

    def reach_joint_angles(self, joint_positions, tolerance: float = 1e-3) -> bool:
        self.arm_group.set_goal_joint_tolerance(tolerance)
        self.arm_group.set_joint_value_target(joint_positions)
        success = bool(self.arm_group.go(wait=True))
        self.arm_group.stop()
        return success

    def convert_numpy_to_pose_stamped(self, pose: np.ndarray) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.header.stamp = rospy.Time.now()

        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = pose[2]
        pose_msg.pose.orientation.x = pose[3]
        pose_msg.pose.orientation.y = pose[4]
        pose_msg.pose.orientation.z = pose[5]
        pose_msg.pose.orientation.w = pose[6]
        return pose_msg

    def generate_candidate_set(self, target_count: int) -> Tuple[List[np.ndarray], List[PoseStamped]]:
        candidate_joints: List[np.ndarray] = []
        candidate_poses: List[PoseStamped] = []
        attempts = 0

        while len(candidate_joints) < target_count and attempts < self.max_pose_generation_attempts:
            attempts += 1
            pose = self.pose_generator.sampleInSphere(
                self.object_center,
                self.sample_radius_min,
                self.sample_radius_max,
            )
            pose = self._enforce_look_at_center(pose)
            if not self._pose_looks_at_center(pose):
                continue
            joints = self.pose_generator.calcIK(pose)
            if joints is None:
                continue

            if not self._plan_joint_target(joints):
                continue

            candidate_joints.append(joints)
            candidate_poses.append(self.convert_numpy_to_pose_stamped(pose))

        if len(candidate_joints) < target_count:
            rospy.logwarn(
                "Generated only %d/%d feasible candidate poses after %d attempts",
                len(candidate_joints),
                target_count,
                attempts,
            )

        return candidate_joints, candidate_poses

    def request_nbv_scores(self, candidate_poses: List[PoseStamped]) -> Optional[np.ndarray]:
        if len(candidate_poses) == 0:
            return None

        req = NBVRequest()
        req.poses = candidate_poses
        req.sensor = self.sensor_type

        rospy.loginfo("Requesting NBV scores for %d candidates", len(candidate_poses))
        res: Optional[NBVResponse] = self._send_req(self.nbv_client, req, "/next_best_view")
        if res is None:
            rospy.logwarn("NBV request failed; falling back to zero scores")
            return np.zeros((len(candidate_poses),), dtype=np.float64)
        if not res.success:
            rospy.logwarn("NBV response unsuccessful (%s); falling back to zero scores", res.message)
            return np.zeros((len(candidate_poses),), dtype=np.float64)

        scores = np.array(res.scores, dtype=np.float64)
        if scores.shape[0] != len(candidate_poses):
            rospy.logwarn("NBV scores length mismatch, applying safe pad/truncate")
            fixed = np.zeros((len(candidate_poses),), dtype=np.float64)
            copy_n = min(scores.shape[0], len(candidate_poses))
            fixed[:copy_n] = scores[:copy_n]
            scores = fixed

        return scores

    def call_add_view(self) -> bool:
        req = TriggerRequest()
        res = self._send_req(self.add_view_client, req, "/add_view")
        return res is not None and bool(res.success)

    def call_save_model(self, success: bool) -> bool:
        req = SaveModelRequest()
        req.success = success
        res = self._send_req(self.save_model_client, req, "/save_model")
        return res is not None and bool(res.success)

    def _find_semicircle_start_joints(self, theta: float, phi: float, radius: Optional[float] = None) -> Optional[np.ndarray]:
        view_radius = float(self.starting_semicircle_radius if radius is None else radius)
        pose = self.pose_generator.poseOnSphere(
            theta,
            phi,
            view_radius,
            self.object_center,
        )
        pose = self._enforce_look_at_center(pose)
        if not self._pose_looks_at_center(pose):
            return None
        joints = self.pose_generator.calcIK(pose)
        if joints is None:
            return None
        if not self._plan_joint_target(joints):
            return None
        return joints

    def _anchor_radius_candidates(self) -> List[float]:
        center_dist = float(np.linalg.norm(self.object_center))
        reach_limited = max(0.12, self.max_reach_m - center_dist - 0.03)
        anchor_min = min(float(self.required_anchor_radius_min), float(self.required_anchor_radius_max))
        anchor_max = max(float(self.required_anchor_radius_min), float(self.required_anchor_radius_max))
        anchor_max = min(anchor_max, reach_limited)
        if anchor_max < anchor_min:
            anchor_min = max(0.12, anchor_max)
        if anchor_max < 0.12:
            anchor_max = 0.12
            anchor_min = 0.12
        candidates = list(np.linspace(anchor_max, anchor_min, num=5))
        # also try the configured starting radius if it is in a feasible range
        start_radius = min(float(self.starting_semicircle_radius), reach_limited)
        candidates.insert(0, max(0.12, start_radius))
        # keep unique order
        uniq = []
        for radius in candidates:
            if all(abs(radius - existing) > 1e-6 for existing in uniq):
                uniq.append(radius)
        return uniq

    def _capture_start_view_from_joints(self, joints: np.ndarray, view_label: str, idx: int) -> bool:
        if not self.reach_joint_angles(joints):
            rospy.logwarn("Failed to execute starting pose %d (%s)", idx + 1, view_label)
            return False
        if not self.call_add_view():
            rospy.logwarn("add_view failed at starting pose %d (%s)", idx + 1, view_label)
            return False
        rospy.loginfo("Captured starting view %d/%d (%s)", idx + 1, self.starting_views, view_label)
        return True

    @staticmethod
    def _invert_transform(transform: np.ndarray) -> np.ndarray:
        inv = np.eye(4, dtype=float)
        inv[:3, :3] = transform[:3, :3].T
        inv[:3, 3] = -inv[:3, :3] @ transform[:3, 3]
        return inv

    @staticmethod
    def _transform_to_matrix(transform) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        quat = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        matrix[:3, :3] = sciR.from_quat(quat).as_matrix()
        matrix[:3, 3] = [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ]
        return matrix

    @staticmethod
    def _matrix_to_pose7(matrix: np.ndarray) -> np.ndarray:
        quat = sciR.from_matrix(matrix[:3, :3]).as_quat()
        pose = np.zeros((7,), dtype=float)
        pose[:3] = matrix[:3, 3]
        pose[3:] = quat
        return pose

    def _load_fixed_starting_pose_candidates(self) -> List[np.ndarray]:
        if not self.use_fixed_starting_views:
            return []
        if len(self.fixed_starting_views_path.strip()) == 0:
            rospy.logwarn("use_fixed_starting_views=True but fixed_starting_views_path is empty")
            return []

        json_path = osp.expanduser(self.fixed_starting_views_path)
        if not osp.isfile(json_path):
            rospy.logwarn("Fixed starting views file not found: %s", json_path)
            return []

        try:
            with open(json_path, "r") as file:
                data = json.load(file)
        except Exception as exc:
            rospy.logwarn("Failed to read fixed starting views json (%s): %s", json_path, str(exc))
            return []

        frames = data.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            rospy.logwarn("No frames found in fixed starting views file: %s", json_path)
            return []

        target_count = max(1, min(self.starting_views, self.fixed_starting_views_count, len(frames)))

        candidate_sensor_links = [
            self.fixed_starting_sensor_link,
            "ee_camera_optical_frame",
            "ee_camera_link",
            "camera_link",
        ]
        unique_sensor_links: List[str] = []
        for sensor_link in candidate_sensor_links:
            if sensor_link not in unique_sensor_links:
                unique_sensor_links.append(sensor_link)

        sensor_to_ee = None
        selected_sensor_link = None
        last_exc = None
        for sensor_link in unique_sensor_links:
            try:
                ee_to_sensor_tf = self.tf_buffer.lookup_transform(
                    self.pose_solver_ee_link,
                    sensor_link,
                    rospy.Time(0),
                    rospy.Duration(2.0),
                )
                ee_to_sensor = self._transform_to_matrix(ee_to_sensor_tf.transform)
                sensor_to_ee = self._invert_transform(ee_to_sensor)
                selected_sensor_link = sensor_link
                break
            except Exception as exc:
                last_exc = exc

        if sensor_to_ee is None:
            rospy.logwarn(
                "Failed TF lookup for fixed starting views (%s <- %s): %s",
                self.pose_solver_ee_link,
                self.fixed_starting_sensor_link,
                str(last_exc),
            )
            return []

        if selected_sensor_link != self.fixed_starting_sensor_link:
            rospy.logwarn(
                "fixed_starting_sensor_link '%s' unavailable; fallback to '%s'",
                self.fixed_starting_sensor_link,
                selected_sensor_link,
            )

        pose_candidates: List[np.ndarray] = []
        for frame in frames[:target_count]:
            transform_matrix = frame.get("transform_matrix")
            if transform_matrix is None:
                continue
            sensor_pose = np.array(transform_matrix, dtype=float)
            if sensor_pose.shape != (4, 4):
                continue
            ee_pose = sensor_pose @ sensor_to_ee
            pose_candidates.append(self._matrix_to_pose7(ee_pose))

        return pose_candidates

    def _capture_fixed_starting_views(self) -> Optional[int]:
        if not self.use_fixed_starting_views:
            return None

        pose_candidates = self._load_fixed_starting_pose_candidates()
        if len(pose_candidates) == 0:
            if self.fixed_starting_views_strict:
                rospy.logerr("Fixed starting views enabled but no valid fixed poses were loaded")
                return -1
            rospy.logwarn("Fixed starting views enabled but unavailable; fallback to generated starting views")
            return 0

        captured_count = 0
        for idx, pose in enumerate(pose_candidates):
            joints = self.pose_generator.calcIK(pose)
            if joints is None or not self._plan_joint_target(joints):
                message = f"Fixed start pose {idx + 1} IK/plan failed"
                if self.fixed_starting_views_strict:
                    rospy.logerr(message)
                    return -1
                rospy.logwarn(message)
                continue

            if not self._capture_start_view_from_joints(joints, "fixed", captured_count):
                if self.fixed_starting_views_strict:
                    return -1
                continue

            captured_count += 1
            if captured_count >= self.starting_views:
                break

        if captured_count < self.starting_views and self.fixed_starting_views_strict:
            rospy.logerr(
                "Captured only %d/%d fixed starting views in strict mode",
                captured_count,
                self.starting_views,
            )
            return -1

        if captured_count > 0:
            rospy.loginfo("Captured %d fixed starting views from %s", captured_count, self.fixed_starting_views_path)

        return captured_count

    def capture_starting_views(self) -> bool:
        rospy.loginfo("Collecting %d starting views", self.starting_views)
        use_semicircle = str(self.starting_views_mode).lower() == "semicircle"
        if use_semicircle:
            rospy.loginfo(
                "Starting views mode=semicircle center=%s radius=%.3f phi=%.3f theta=[%.3f, %.3f]",
                self.object_center.tolist(),
                self.starting_semicircle_radius,
                self.starting_semicircle_phi,
                self.starting_semicircle_theta_min,
                self.starting_semicircle_theta_max,
            )

        fixed_captured = self._capture_fixed_starting_views()
        if fixed_captured is not None:
            if fixed_captured < 0:
                return False
            if fixed_captured >= self.starting_views:
                return True
            captured_count = fixed_captured
        else:
            captured_count = 0

        # enforce key anchors first: one back view + one low-side upward view
        if use_semicircle and self.starting_views > 0 and captured_count == 0:
            forced_specs = []
            if self.require_back_start_view:
                back_candidates = [
                    (np.pi, self.starting_semicircle_phi),
                    (3.0 * np.pi / 4.0, self.starting_semicircle_phi),
                    (-3.0 * np.pi / 4.0, self.starting_semicircle_phi),
                ]
                forced_specs.append(("back", back_candidates))

            if self.require_low_side_start_view and len(forced_specs) < self.starting_views:
                low_phi = float(np.clip(self.starting_low_side_phi, np.pi / 2.0 + 0.05, np.pi - 0.05))
                low_candidates = [
                    (np.pi / 2.0, low_phi),
                    (-np.pi / 2.0, low_phi),
                    (2.0 * np.pi / 3.0, low_phi),
                    (-2.0 * np.pi / 3.0, low_phi),
                    (np.pi / 3.0, low_phi),
                    (-np.pi / 3.0, low_phi),
                    (np.pi / 2.0, max(np.pi / 2.0 + 0.05, low_phi - 0.20)),
                    (-np.pi / 2.0, max(np.pi / 2.0 + 0.05, low_phi - 0.20)),
                    (2.0 * np.pi / 3.0, max(np.pi / 2.0 + 0.05, low_phi - 0.20)),
                    (-2.0 * np.pi / 3.0, max(np.pi / 2.0 + 0.05, low_phi - 0.20)),
                    (np.pi / 2.0, max(np.pi / 2.0 + 0.05, low_phi - 0.35)),
                    (-np.pi / 2.0, max(np.pi / 2.0 + 0.05, low_phi - 0.35)),
                ]
                forced_specs.append(("low-side", low_candidates))

            for label, angle_candidates in forced_specs:
                if captured_count >= self.starting_views:
                    break
                selected_joints = None
                selected_theta = None
                selected_phi = None
                selected_radius = None
                for theta, phi in angle_candidates:
                    for radius in self._anchor_radius_candidates():
                        selected_joints = self._find_semicircle_start_joints(theta, phi, radius=radius)
                        if selected_joints is not None:
                            selected_theta = theta
                            selected_phi = phi
                            selected_radius = radius
                            rospy.loginfo(
                                "Selected required %s start-view candidate theta=%.3f phi=%.3f radius=%.3f",
                                label,
                                theta,
                                phi,
                                radius,
                            )
                            break
                    if selected_joints is not None:
                        break

                if selected_joints is None:
                    msg = (
                        f"Failed to find required '{label}' starting view "
                        f"(IK/plan infeasible across angle+radius candidates)."
                    )
                    if self.strict_required_start_views:
                        rospy.logerr(msg)
                        return False
                    rospy.logwarn(msg + " Continue with remaining starting views because strict mode is off.")
                    continue

                if not self._capture_start_view_from_joints(selected_joints, f"required-{label}", captured_count):
                    return False
                captured_count += 1

        for view_idx in range(captured_count, self.starting_views):
            candidate_joints = []
            if use_semicircle:
                if self.starting_views == 1:
                    theta = 0.5 * (self.starting_semicircle_theta_min + self.starting_semicircle_theta_max)
                else:
                    ratio = float(view_idx) / float(self.starting_views - 1)
                    theta = self.starting_semicircle_theta_min + ratio * (
                        self.starting_semicircle_theta_max - self.starting_semicircle_theta_min
                    )
                pose = self.pose_generator.poseOnSphere(
                    theta,
                    self.starting_semicircle_phi,
                    self.starting_semicircle_radius,
                    self.object_center,
                )
                pose = self._enforce_look_at_center(pose)
                if not self._pose_looks_at_center(pose):
                    rospy.logwarn(
                        "Semicircle start pose %d does not sufficiently face object center; skipping.",
                        view_idx + 1,
                    )
                    candidate_joints = []
                    continue
                joints = self.pose_generator.calcIK(pose)
                if joints is not None and self._plan_joint_target(joints):
                    candidate_joints = [joints]
                else:
                    rospy.logwarn(
                        "Semicircle start pose %d IK/plan failed at theta=%.3f; fallback to random feasible pose.",
                        view_idx + 1,
                        theta,
                    )

            if len(candidate_joints) == 0:
                candidate_joints, _ = self.generate_candidate_set(1)
                if len(candidate_joints) == 0:
                    rospy.logerr("Failed to generate feasible starting pose at index %d", view_idx)
                    rospy.logerr(
                        "IK produced 0 feasible poses. Check object_center=%s, sample_radius=[%.3f, %.3f], max_reach_m=%.3f",
                        self.object_center.tolist(),
                        self.sample_radius_min,
                        self.sample_radius_max,
                        self.max_reach_m,
                    )
                    return False

            if not self._capture_start_view_from_joints(candidate_joints[0], "default", view_idx):
                return False

        return True

    def run_nbv_round(self, round_idx: int) -> bool:
        expected_step = self.nbv_start_step + (round_idx * self.nbv_interval_steps)
        rospy.loginfo(
            "NBV round %d/%d (expected training step >= %d)",
            round_idx + 1,
            self.nbv_max_rounds,
            expected_step,
        )

        candidate_count = int(self.num_poses)
        candidate_joints: List[np.ndarray] = []
        candidate_poses: List[PoseStamped] = []

        if self.use_random_nbv:
            candidate_joints, candidate_poses = self.generate_candidate_set(candidate_count)
            if len(candidate_joints) == 0:
                rospy.logwarn("No candidate views generated for NBV round %d", round_idx + 1)
                return False
            best_idx = int(np.random.randint(0, len(candidate_joints)))
            rospy.loginfo("NBV best candidate index=%d (random selection)", best_idx)
        else:
            retry_idx = 0
            scores: Optional[np.ndarray] = None
            while True:
                candidate_joints, candidate_poses = self.generate_candidate_set(candidate_count)
                if len(candidate_joints) == 0:
                    rospy.logwarn(
                        "No candidate views generated for NBV round %d (retry=%d)",
                        round_idx + 1,
                        retry_idx,
                    )
                    return False

                scores = self.request_nbv_scores(candidate_poses)
                if scores is None:
                    return False

                valid_mask = scores > (self.nbv_filtered_score + self.nbv_filtered_eps)
                valid_count = int(np.sum(valid_mask))
                if valid_count >= self.nbv_min_valid_candidates:
                    break

                retry_idx += 1
                if self.nbv_candidate_retry_max >= 0 and retry_idx > self.nbv_candidate_retry_max:
                    rospy.logerr(
                        "Only %d valid candidates after %d retries (< %d required). Aborting this NBV round.",
                        valid_count,
                        self.nbv_candidate_retry_max,
                        self.nbv_min_valid_candidates,
                    )
                    return False

                candidate_count += max(1, int(self.nbv_candidate_growth))
                rospy.logwarn(
                    "%d/%d candidates valid (< %d). Retrying with %d candidates (retry %d).",
                    valid_count,
                    len(candidate_poses),
                    self.nbv_min_valid_candidates,
                    candidate_count,
                    retry_idx,
                )

            assert scores is not None
            valid_mask = scores > (self.nbv_filtered_score + self.nbv_filtered_eps)
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.shape[0] < self.nbv_min_valid_candidates:
                rospy.logerr(
                    "Internal check failed: valid candidates=%d (< %d). Aborting NBV round.",
                    valid_indices.shape[0],
                    self.nbv_min_valid_candidates,
                )
                return False
            best_local = int(np.argmax(scores[valid_indices]))
            best_idx = int(valid_indices[best_local])
            rospy.loginfo("NBV best candidate index=%d score=%.6f (valid=%d)", best_idx, float(scores[best_idx]), int(np.sum(valid_mask)))

        if not self.reach_joint_angles(candidate_joints[best_idx]):
            rospy.logwarn("Failed to move to best NBV pose")
            return False

        if not self.call_add_view():
            rospy.logwarn("add_view failed after moving to best NBV pose")
            return False

        if not self.call_save_model(True):
            rospy.logwarn("save_model failed after adding NBV view")
            return False

        rospy.loginfo("NBV round completed: moved to best pose, captured new view, and requested GS continue training")
        return True

    def run(self) -> bool:
        if not self.reach_named_position(self.home_named_target):
            rospy.logerr("Failed to go to home target: %s", self.home_named_target)
            return False

        if self.reuse_existing_dataset:
            rospy.loginfo(
                "reuse_existing_dataset=True, skip starting-view capture and use existing dataset: %s",
                self.existing_dataset_dir,
            )
        else:
            if not self.capture_starting_views():
                return False

        rospy.loginfo("Starting 3DGS training after initial view collection")
        if not self.call_save_model(True):
            rospy.logerr("Initial save_model/start_training call failed")
            return False

        for round_idx in range(self.nbv_max_rounds):
            if not self.run_nbv_round(round_idx):
                rospy.logwarn("NBV round %d failed, stopping workflow", round_idx + 1)
                return False

        rospy.loginfo("RM65 NBV workflow finished successfully")
        self.reach_named_position(self.home_named_target)
        return True


def main():
    try:
        controller = RM65NBVController()
        success = controller.run()
        if success:
            rospy.loginfo("RM65 NBV workflow completed successfully")
        else:
            rospy.logerr("RM65 NBV workflow failed")
    except KeyboardInterrupt:
        rospy.loginfo("RM65 NBV interrupted by user")
    except Exception as e:
        rospy.logerr(f"RM65 NBV error: {e}")
    finally:
        rospy.signal_shutdown("NBV workflow finished")
        sys.exit(0)


if __name__ == "__main__":
    main()
