from __future__ import annotations

import numpy as np

from .gesture_abstraction import FINGER_MCPS, WRIST
from .state import HandAction, HandState


class HandActionEncoder:
    """Convert filtered hand state into robot-learning friendly action primitives."""

    PLANAR_VEL_THR = 45.0
    DEPTH_VEL_THR = 0.12
    WRIST_ROT_THR = 0.22
    MIN_DT = 1e-3

    # Mean finger-extension ratio range mapped to a graded OPEN PALM gripper command.
    OPEN_RATIO_LO = 1.5
    OPEN_RATIO_HI = 2.2

    def __init__(self, planar_vel_thr: float = PLANAR_VEL_THR,
                 depth_vel_thr: float = DEPTH_VEL_THR,
                 wrist_rot_thr: float = WRIST_ROT_THR,
                 min_dt: float = MIN_DT):
        self.planar_vel_thr = planar_vel_thr
        self.depth_vel_thr = depth_vel_thr
        self.wrist_rot_thr = wrist_rot_thr
        self.min_dt = min_dt

        self._prev_palm_2d = [None, None]
        self._prev_palm_3d = [None, None]
        self._prev_t = [None, None]

    def encode(self, hands: list[HandState], timestamp: float) -> list[HandAction]:
        actions: list[HandAction] = []
        for hand in hands:
            actions.append(self._encode_hand(hand, timestamp))
        return actions

    def _encode_hand(self, hand: HandState, timestamp: float) -> HandAction:
        slot = hand.slot
        prev_t = self._prev_t[slot]
        dt = max(timestamp - prev_t, self.min_dt) if prev_t is not None else None

        palm_2d = hand.palm_center_2d.astype(np.float32)
        palm_3d = hand.palm_center_3d.astype(np.float32)

        if dt is None or self._prev_palm_2d[slot] is None:
            vel_2d = np.zeros(2, dtype=np.float32)
            vel_3d = np.zeros(3, dtype=np.float32)
        else:
            vel_2d = (palm_2d - self._prev_palm_2d[slot]) / dt
            vel_3d = (palm_3d - self._prev_palm_3d[slot]) / dt

        self._prev_palm_2d[slot] = palm_2d.copy()
        self._prev_palm_3d[slot] = palm_3d.copy()
        self._prev_t[slot] = timestamp

        label = self._label_action(hand, vel_2d, vel_3d)
        conf = float(max(0.05, hand.gesture_confidence))
        gripper = self._gripper_command(
            hand.gesture, hand.meta.get("pinch_norm"), hand.meta.get("mean_ratio", 0.0)
        )
        pose_hint = self._target_pose_hint(hand)

        return HandAction(
            slot=slot,
            side=hand.side,
            label=label,
            confidence=conf,
            palm_velocity_2d=vel_2d,
            palm_velocity_3d=vel_3d,
            gripper_command=gripper,
            target_pose_hint=pose_hint,
            meta={
                "gesture": hand.gesture,
                "pinch_norm": hand.meta.get("pinch_norm"),
                "mean_ratio": hand.meta.get("mean_ratio"),
            },
        )

    def _label_action(
        self, hand: HandState, vel_2d: np.ndarray, vel_3d: np.ndarray
    ) -> str:
        speed_xy = float(np.linalg.norm(vel_2d))
        speed_z = float(abs(vel_3d[2]))
        wrist_yaw = float(hand.palm_normal[0]) if hand.palm_normal.size else 0.0

        if hand.gesture == "PINCH":
            return "PINCH_PICK"
        if hand.gesture == "OPEN PALM" and speed_xy < 20.0:
            return "RELEASE"
        if speed_z > self.depth_vel_thr:
            return "REACH_DEPTH"
        if speed_xy > self.planar_vel_thr:
            return "MOVE_XY"
        if abs(wrist_yaw) > self.wrist_rot_thr:
            return "ROTATE_WRIST"
        return "HOLD"

    def _gripper_command(self, gesture: str, pinch_norm: float | None,
                          mean_ratio: float = 0.0) -> float:
        if gesture == "PINCH":
            if pinch_norm is None:
                return 0.2
            return float(np.clip(pinch_norm / 0.55, 0.0, 0.5))
        if gesture == "OPEN PALM":
            # Grade the open command by how extended the fingers are, rather than
            # always reporting a fully-open gripper.
            span = self.OPEN_RATIO_HI - self.OPEN_RATIO_LO
            t = np.clip((mean_ratio - self.OPEN_RATIO_LO) / span, 0.0, 1.0)
            return float(0.6 + 0.4 * t)
        return 0.0

    @staticmethod
    def _target_pose_hint(hand: HandState) -> np.ndarray:
        center = hand.palm_center_3d.astype(np.float32).copy()
        normal = hand.palm_normal.astype(np.float32).copy()
        return np.concatenate([center, normal], axis=0)


def build_hand_state(
    *,
    slot: int,
    side: str,
    keypoints: np.ndarray,
    scores: np.ndarray,
    gesture: str,
    meta: dict,
    timestamp: float,
    palm_depth: float | None,
) -> HandState:
    palm_ids = [WRIST, *FINGER_MCPS]
    valid = [j for j in palm_ids if scores[j] > 0.12]
    if valid:
        palm_center_2d = keypoints[valid, :2].mean(axis=0).astype(np.float32)
    else:
        palm_center_2d = keypoints[:, :2].mean(axis=0).astype(np.float32)

    palm_center_3d = np.zeros(3, dtype=np.float32)
    palm_center_3d[:2] = palm_center_2d
    if palm_depth is not None:
        palm_center_3d[2] = float(palm_depth)

    return HandState(
        slot=slot,
        side=side,
        keypoints=keypoints.copy(),
        scores=scores.copy(),
        palm_depth=palm_depth,
        gesture=gesture,
        gesture_confidence=float(meta.get("confidence", 0.0)),
        palm_center_2d=palm_center_2d,
        palm_center_3d=palm_center_3d,
        palm_normal=np.asarray(meta.get("palm_normal", np.zeros(3)), dtype=np.float32),
        meta=dict(meta),
        timestamp=timestamp,
    )

