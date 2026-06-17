from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .robot_mapper import ArmCommand


# Hand-visibility thresholds — the single source of truth shared by the live
# overlay and HandState.visible (which gates the IK/robot path). A detection is
# only "visible" if its mean confidence clears MEAN_SCORE_THR *and* enough of
# its joints individually clear JOINT_SCORE_THR. The two-part test rejects the
# low-confidence phantom poses the detector-less top-down model emits when no
# real hand is in frame.
MEAN_SCORE_THR = 0.18
JOINT_SCORE_THR = 0.12
VISIBLE_FRAC_THR = 0.55


def is_hand_visible(scores: np.ndarray) -> bool:
    """True when a hand's keypoint scores indicate a real detection, not a
    spurious top-down pose. Mirrors the live-overlay acceptance gate."""
    if not getattr(scores, "size", 0):
        return False
    return (float(np.mean(scores)) >= MEAN_SCORE_THR and
            float(np.mean(scores > JOINT_SCORE_THR)) >= VISIBLE_FRAC_THR)


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
        return is_hand_visible(self.scores)


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

