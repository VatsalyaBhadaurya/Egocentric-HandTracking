from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot_mapper import ArmCommand


@dataclass(eq=False)
class HandState:
    slot: int
    side: str
    keypoints: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)
    palm_depth: float | None = None
    gesture: str = "GRASP"
    gesture_confidence: float = 0.0
    palm_center_2d: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    palm_center_3d: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    palm_normal: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @property
    def visible(self) -> bool:
        return bool(self.scores.size and float(np.mean(self.scores)) > 0.0)


@dataclass(eq=False)
class HandAction:
    slot: int
    side: str
    label: str
    confidence: float
    palm_velocity_2d: np.ndarray
    palm_velocity_3d: np.ndarray
    gripper_command: float
    target_pose_hint: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class FrameState:
    timestamp: float
    fps: float
    hands: list[HandState]
    actions: list[HandAction] = field(default_factory=list)
    arm_commands: list[ArmCommand] = field(default_factory=list)

