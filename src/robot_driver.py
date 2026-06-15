# Drives a physical LeRobot SO-101 follower arm (right arm) from the joint
# angles produced by SingleArmIKRetargeter, using the same joint ranges and
# normalization as vbhs/src/vbhs/scripts/action_playback.py.

from __future__ import annotations

import time

import numpy as np

# [j1..j5, gripper] joint limits in radians, matching action_playback.py's
# JOINT_RANGES_RADS / Dual_S101_Assembly.urdf.
JOINT_RANGES_RADS = np.array([
    [-1.91986, 1.91986],
    [-1.74533, 1.74533],
    [-1.74533, 1.5708],
    [-1.65806, 1.65806],
    [-2.79253, 2.79253],
    [-0.174533, 1.74533],
])

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ACTION_NAMES = [f"{m}.pos" for m in MOTOR_NAMES]

# Matches action_playback.send_standby's STANDBY_POSITION.
STANDBY_ACTION = {
    "shoulder_pan.pos": -1.20,
    "shoulder_lift.pos": -98.01,
    "elbow_flex.pos": 99.73,
    "wrist_flex.pos": 4.62,
    "wrist_roll.pos": -2.59,
    "gripper.pos": 0.0,
}


def joint_angles_to_action(joint_angles_rad) -> dict[str, float]:
    """Converts [j1..j5, gripper] radians (vbhs IK convention) into a LeRobot
    SO-101 send_action dict, normalized the same way as
    vbhs.scripts.action_playback.prepare_action."""
    angles = np.clip(np.asarray(joint_angles_rad, dtype=np.float64),
                      JOINT_RANGES_RADS[:, 0], JOINT_RANGES_RADS[:, 1])
    normalized = (angles - JOINT_RANGES_RADS[:, 0]) / (JOINT_RANGES_RADS[:, 1] - JOINT_RANGES_RADS[:, 0])
    return {
        name: float(n * 100) if "gripper" in name else float((n - 0.5) * 200)
        for n, name in zip(normalized, ACTION_NAMES)
    }


class SO101FollowerDriver:
    """Connects to and drives a physical LeRobot SO-101 follower arm (right arm)."""

    def __init__(self, port: str, robot_id: str = "right_follower",
                 p_coefficient: int = 12, i_coefficient: int = 0, d_coefficient: int = 24,
                 acceleration_limit: int = 20, gripper_acceleration_limit: int = 50,
                 velocity_limit: int = 500, gripper_velocity_limit: int = 1000):
        from lerobot.robots.so101_follower import SO101FollowerConfig
        from lerobot.robots.utils import make_robot_from_config

        self.robot = make_robot_from_config(SO101FollowerConfig(port=port, id=robot_id))
        self.robot.connect()

        # Lower the position-control gains for smoother, gentler tracking
        # before commanding any motion.
        self.robot.bus.disable_torque()
        for motor in self.robot.bus.motors:
            self.robot.bus.write("P_Coefficient", motor, p_coefficient)
            self.robot.bus.write("I_Coefficient", motor, i_coefficient)
            self.robot.bus.write("D_Coefficient", motor, d_coefficient)
        self.robot.bus.enable_torque()

        acceleration = {motor: acceleration_limit for motor in self.robot.bus.motors}
        acceleration["gripper"] = gripper_acceleration_limit
        velocity = {motor: velocity_limit for motor in self.robot.bus.motors}
        velocity["gripper"] = gripper_velocity_limit
        self.robot.bus.sync_write("Acceleration", acceleration)
        self.robot.bus.sync_write("Goal_Velocity", velocity)

        self.send_standby()

    def send(self, joint_angles_rad) -> None:
        # Commands the arm towards the given [j1..j5, gripper] radian targets.
        self.robot.send_action(joint_angles_to_action(joint_angles_rad))

    def send_standby(self) -> None:
        self.robot.send_action(dict(STANDBY_ACTION))
        time.sleep(1.0)

    def close(self) -> None:
        try:
            self.send_standby()
        finally:
            self.robot.disconnect()
