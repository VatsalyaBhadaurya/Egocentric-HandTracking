# Single-arm inverse-kinematics retargeter: maps a tracked right hand (in
# camera-space, MediaPipe-named landmarks) to SO-101 right-arm joint angles
# using the vendored Vision-Based-Hand-Shadowing (vbhs) PyBullet IK solver.

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

_URDF_PATH = Path(__file__).resolve().parent.parent / "vbhs" / "robot" / "Dual_S101_Assembly.urdf"

# Joint/link layout for Dual_S101_Assembly.urdf, per
# vbhs/src/vbhs/simulation/simulator.py's docstring.
RIGHT_ARM_JOINTS = [8, 9, 10, 11, 12]
RIGHT_END_EFFECTOR = 12
RIGHT_GRIPPER_JOINT = 13
MIN_Z_HEIGHT = 0.05


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

    def step(self, hand_pose_camera_space: vbhs_types.HandPose3D) -> IKArmResult:
        # Converts camera-space landmarks to robot space, solves IK for joints
        # 1-5, and estimates the gripper angle (joint 6).
        robot_landmarks = self._space_transformer(
            vbhs_types.HandLandmarksCameraSpace(
                left_hand_landmarks=None,
                right_hand_landmarks=hand_pose_camera_space))
        hand_pose_robot = robot_landmarks.right_hand_landmarks

        target_pos = target_pose.calculate_target_position(hand_pose_robot, target_pose.Hand.RIGHT)
        if target_pos is not None and target_pos[2] < MIN_Z_HEIGHT:
            target_pos = None

        target_quat = None
        if target_pos is not None:
            target_quat = target_pose.calculate_target_orientation(hand_pose_robot, target_pose.Hand.RIGHT)
            if target_quat is None:
                target_quat = target_pose.calculate_fallback_orientation(hand_pose_robot, target_pose.Hand.RIGHT)

        if target_pos is None:
            return IKArmResult(joint_angles=None, target_pos=None, target_quat=None)

        arm_angles = self._ik_solver.solve(tuple(target_pos), target_quat)
        if arm_angles is None:
            return IKArmResult(joint_angles=None, target_pos=tuple(target_pos), target_quat=target_quat)

        gripper_commands = self._gripper_estimator(
            vbhs_types.HandLandmarksRobotSpace(
                left_hand_landmarks=None, right_hand_landmarks=hand_pose_robot))
        gripper_angle = gripper_commands.right_gripper_angle

        return IKArmResult(
            joint_angles=[*arm_angles, gripper_angle],
            target_pos=tuple(target_pos),
            target_quat=target_quat)

    def close(self):
        p.disconnect(physicsClientId=self._client)
