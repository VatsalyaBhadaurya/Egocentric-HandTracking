# Single-arm inverse-kinematics retargeter: maps a tracked right hand (in
# camera-space, MediaPipe-named landmarks) to SO-101 right-arm joint angles
# using the vendored Vision-Based-Hand-Shadowing (vbhs) PyBullet IK solver.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_VBHS_SRC = Path(__file__).resolve().parent.parent / "vbhs" / "src"
if str(_VBHS_SRC) not in sys.path:
    sys.path.insert(0, str(_VBHS_SRC))

import pybullet as p  # noqa: E402
import pybullet_data  # noqa: E402

from vbhs.pipeline import types as vbhs_types  # noqa: E402
from vbhs.pipeline import inverse_kinematics  # noqa: E402
from vbhs.pipeline import joint_transformations  # noqa: E402
from vbhs.pipeline import robot_gripper_estimation  # noqa: E402
from vbhs.pipeline.hands import target_pose  # noqa: E402

from .one_euro_filter import OneEuroFilter  # noqa: E402

_URDF_PATH = Path(__file__).resolve().parent.parent / "vbhs" / "robot" / "Dual_S101_Assembly.urdf"

# Joint/link layout for Dual_S101_Assembly.urdf, per
# vbhs/src/vbhs/simulation/simulator.py's docstring.
RIGHT_ARM_JOINTS = [8, 9, 10, 11, 12]
RIGHT_END_EFFECTOR = 12
RIGHT_GRIPPER_JOINT = 13
MIN_Z_HEIGHT = 0.05

# Largest plausible frame-to-frame motion of the IK target (metres). A jump
# bigger than this is almost always a depth glitch (a single bad pixel throws
# the deprojected point metres away), so we clamp the step instead of chasing
# it — this is the main source of the arm "lunging" between frames.
MAX_TARGET_STEP_M = 0.12


@dataclass
class IKArmResult:
    joint_angles: Optional[list[float]]  # [j1..j6] radians for joints 1-6, or None if no valid target
    target_pos: Optional[tuple[float, float, float]]
    target_quat: Optional[tuple[float, float, float, float]]


class SingleArmIKRetargeter:
    """Retargets a right-hand pose (camera-space landmarks) to SO-101 right-arm joint angles."""

    def __init__(self, gui: bool = False):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        self._robot_id = p.loadURDF(str(_URDF_PATH), useFixedBase=True,
                                    physicsClientId=self._client)

        self._space_transformer = joint_transformations.RobotSpaceHandsFromCameraSpace()
        self._ik_solver = inverse_kinematics.IKSolver(
            robot_id=self._robot_id,
            end_effector_idx=RIGHT_END_EFFECTOR,
            joint_indices=RIGHT_ARM_JOINTS)
        # Tighten grip by ~10 degrees, matching vbhs's control_command_estimation default.
        self._gripper_estimator = robot_gripper_estimation.RobotGripperCommandsFromHandLandmarks(
            offset=-0.175)

        # --- Temporal smoothing / continuity state -------------------------
        # The raw pipeline only 1-Euro-filters the 2D keypoints; depth (and
        # therefore the IK target) and the solved joint angles are unfiltered.
        # We smooth the robot-space target position and the final joint command
        # here so the servos see a continuous stream instead of per-frame noise.
        self._target_filter = OneEuroFilter(freq=25.0, min_cutoff=0.8, beta=0.02)
        self._joint_filter = OneEuroFilter(freq=25.0, min_cutoff=1.0, beta=0.03)
        self._last_target_pos: Optional[np.ndarray] = None
        self._last_quat: Optional[np.ndarray] = None
        # Seed pose the IK solver continues from frame-to-frame. PyBullet uses
        # the robot's *current* joint configuration as the IK seed, so we reset
        # the arm joints to the previous solution before each solve. This keeps
        # the redundant arm on one branch instead of flipping the elbow/wrist.
        self._seed_angles = [0.0] * len(RIGHT_ARM_JOINTS)
        self._reset_arm_to_seed()

    def _reset_arm_to_seed(self) -> None:
        for joint_idx, angle in zip(RIGHT_ARM_JOINTS, self._seed_angles):
            p.resetJointState(self._robot_id, joint_idx, angle,
                              physicsClientId=self._client)

    def reset(self) -> None:
        """Clears temporal state after tracking is lost, so re-acquisition
        snaps to the new pose instead of lerping across the gap. The IK seed is
        kept so the arm holds its last solved pose rather than jumping."""
        self._target_filter.reset()
        self._joint_filter.reset()
        self._last_target_pos = None
        self._last_quat = None

    def step(self, hand_pose_camera_space: vbhs_types.HandPose3D,
             fps: Optional[float] = None) -> IKArmResult:
        # Converts camera-space landmarks to robot space, solves IK for joints
        # 1-5, and estimates the gripper angle (joint 6). Applies outlier
        # rejection, temporal smoothing and IK-seed continuity so the physical
        # arm receives a stable, jitter-free joint stream.
        robot_landmarks = self._space_transformer(
            vbhs_types.HandLandmarksCameraSpace(
                left_hand_landmarks=None,
                right_hand_landmarks=hand_pose_camera_space))
        hand_pose_robot = robot_landmarks.right_hand_landmarks

        target_pos = target_pose.calculate_target_position(hand_pose_robot, target_pose.Hand.RIGHT)
        if target_pos is not None and target_pos[2] < MIN_Z_HEIGHT:
            target_pos = None

        if target_pos is None:
            # Tracking lost — drop temporal state so re-acquisition snaps cleanly.
            self.reset()
            return IKArmResult(joint_angles=None, target_pos=None, target_quat=None)

        target_pos = np.asarray(target_pos, dtype=np.float64)

        # Reject depth-glitch jumps: clamp the step toward the new target so a
        # single bad frame nudges rather than launches the arm.
        if self._last_target_pos is not None:
            step_vec = target_pos - self._last_target_pos
            dist = float(np.linalg.norm(step_vec))
            if dist > MAX_TARGET_STEP_M:
                target_pos = self._last_target_pos + step_vec * (MAX_TARGET_STEP_M / dist)

        # 1-Euro smoothing on the (now de-spiked) target position.
        target_pos = self._target_filter(target_pos, freq=fps)
        self._last_target_pos = target_pos.copy()

        target_quat = target_pose.calculate_target_orientation(hand_pose_robot, target_pose.Hand.RIGHT)
        if target_quat is None:
            target_quat = target_pose.calculate_fallback_orientation(hand_pose_robot, target_pose.Hand.RIGHT)
        target_quat = self._stabilize_quat(target_quat)

        # Seed the solver from the previous solution so the redundant arm stays
        # on one IK branch (no elbow/wrist flips between frames).
        self._reset_arm_to_seed()
        arm_angles = self._ik_solver.solve(tuple(target_pos), target_quat)
        if arm_angles is None:
            return IKArmResult(joint_angles=None, target_pos=tuple(target_pos), target_quat=target_quat)
        self._seed_angles = list(arm_angles)

        # Smooth the final arm joints; the gripper is left responsive (its own
        # estimator is already distance-based and low-noise).
        arm_angles = self._joint_filter(np.asarray(arm_angles, dtype=np.float64),
                                        freq=fps).tolist()

        gripper_commands = self._gripper_estimator(
            vbhs_types.HandLandmarksRobotSpace(
                left_hand_landmarks=None, right_hand_landmarks=hand_pose_robot))
        gripper_angle = gripper_commands.right_gripper_angle

        return IKArmResult(
            joint_angles=[*arm_angles, gripper_angle],
            target_pos=tuple(target_pos),
            target_quat=target_quat)

    def _stabilize_quat(self, quat: Optional[tuple]) -> Optional[tuple]:
        # Removes the quaternion double-cover sign flips (q and -q are the same
        # rotation) that make the wrist spin 180 between frames, then lightly
        # smooths the orientation by normalized lerp toward the new quaternion.
        if quat is None:
            return None
        q = np.asarray(quat, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-9:
            return None
        q = q / n
        if self._last_quat is not None:
            if float(np.dot(q, self._last_quat)) < 0.0:
                q = -q
            q = 0.5 * self._last_quat + 0.5 * q
            q = q / np.linalg.norm(q)
        self._last_quat = q
        return tuple(q.tolist())

    def close(self):
        p.disconnect(physicsClientId=self._client)
