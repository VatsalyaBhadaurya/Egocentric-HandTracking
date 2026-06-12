from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import HandAction, HandState


@dataclass
class ArmCommand:
    joint_targets: np.ndarray
    gripper: float
    action_label: str


class SimpleArmRetargeter:
    """Example retargeter from egocentric hand state to a simple 6-DoF arm command."""

    def __init__(self, image_size: tuple[int, int] = (960, 720)):
        self.image_width = float(image_size[0])
        self.image_height = float(image_size[1])

    def map(self, hand: HandState, action: HandAction) -> ArmCommand:
        px, py = hand.palm_center_2d
        z = hand.palm_depth if hand.palm_depth is not None else 0.4

        norm_x = np.clip((px / self.image_width - 0.5) * 2.0, -1.0, 1.0)
        norm_y = np.clip((py / self.image_height - 0.5) * 2.0, -1.0, 1.0)
        norm_z = np.clip((z - 0.4) / 0.5, -1.0, 1.0)

        roll = float(np.clip(hand.palm_normal[1], -1.0, 1.0))
        pitch = float(np.clip(-hand.palm_normal[0], -1.0, 1.0))
        yaw = float(np.clip(hand.palm_normal[2], -1.0, 1.0))

        joint_targets = np.array(
            [
                0.9 * norm_x,
                -0.7 * norm_y,
                0.6 * norm_z,
                0.8 * roll,
                0.8 * pitch,
                0.6 * yaw,
            ],
            dtype=np.float32,
        )

        return ArmCommand(
            joint_targets=joint_targets,
            gripper=float(np.clip(action.gripper_command, 0.0, 1.0)),
            action_label=action.label,
        )
