# Adapter mapping our InterHand2.6M HandState onto the MediaPipe-named hand
# landmark types expected by the vendored Vision-Based-Hand-Shadowing (vbhs)
# IK pipeline in vbhs/src/vbhs.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_VBHS_SRC = Path(__file__).resolve().parent.parent / "vbhs" / "src"
if str(_VBHS_SRC) not in sys.path:
    sys.path.insert(0, str(_VBHS_SRC))

from vbhs.pipeline import types as vbhs_types  # noqa: E402
from vbhs.pipeline.hands import deprojection  # noqa: E402

# Maps our InterHand2.6M joint indices (see src/gesture_abstraction.py
# FINGER_CHAINS_IDX) onto vbhs.config.config.MEDIAPIPE_HAND_LANDMARKS names.
# Each finger chain is [WRIST, base, mid, tip-1, tip]; for the thumb the
# base-most joint is the CMC, the next is the MCP (matches MediaPipe's
# THUMB_CMC/THUMB_MCP split).
INTERHAND_TO_MEDIAPIPE: dict[int, str] = {
    20: "WRIST",
    3: "THUMB_CMC", 2: "THUMB_MCP", 1: "THUMB_IP", 0: "THUMB_TIP",
    7: "INDEX_FINGER_MCP", 6: "INDEX_FINGER_PIP", 5: "INDEX_FINGER_DIP", 4: "INDEX_FINGER_TIP",
    11: "MIDDLE_FINGER_MCP", 10: "MIDDLE_FINGER_PIP", 9: "MIDDLE_FINGER_DIP", 8: "MIDDLE_FINGER_TIP",
    15: "RING_FINGER_MCP", 14: "RING_FINGER_PIP", 13: "RING_FINGER_DIP", 12: "RING_FINGER_TIP",
    19: "PINKY_MCP", 18: "PINKY_PIP", 17: "PINKY_DIP", 16: "PINKY_TIP",
}


def hand_state_to_pose2d(keypoints: np.ndarray) -> vbhs_types.HandPose2D:
    # Builds a MediaPipe-keyed {name: (u, v)} dict from our 21x{2,3} keypoint array.
    return {name: (float(keypoints[idx, 0]), float(keypoints[idx, 1]))
            for idx, name in INTERHAND_TO_MEDIAPIPE.items()}


def hand_state_to_pose3d(keypoints: np.ndarray, scores: np.ndarray,
                          depth_map: np.ndarray,
                          intrinsics: vbhs_types.CameraIntrinsics,
                          joint_thr: float = 0.12,
                          min_depth_m: float = 0.05,
                          max_depth_m: float = 3.0) -> vbhs_types.HandPose3D:
    # Deprojects each scored-confident InterHand2.6M joint to camera-space 3D
    # metres using the metric depth map (already in metres -> depth_scale=1.0)
    # and the given camera intrinsics. Low-confidence joints map to None.
    depth_scale = 1.0
    H, W = depth_map.shape[:2]
    pose2d = hand_state_to_pose2d(keypoints)

    pose3d: vbhs_types.HandPose3D = {}
    for idx, name in INTERHAND_TO_MEDIAPIPE.items():
        if scores[idx] <= joint_thr:
            pose3d[name] = None
            continue
        u, v = pose2d[name]
        x_px, y_px = int(round(u)), int(round(v))
        if not (0 <= x_px < W and 0 <= y_px < H):
            pose3d[name] = None
            continue
        depth_m = float(depth_map[y_px, x_px])
        try:
            pose3d[name] = deprojection.deproject_point(
                (x_px, y_px), depth_m, intrinsics,
                depth_scale, min_depth_m, max_depth_m)
        except (IndexError, deprojection.DepthOutOfRange):
            pose3d[name] = None

    deprojection._apply_depth_fallback(  # noqa: SLF001 — shared helper, no public API
        pose3d, pose2d, intrinsics, depth_scale, min_depth_m, max_depth_m)
    return pose3d
