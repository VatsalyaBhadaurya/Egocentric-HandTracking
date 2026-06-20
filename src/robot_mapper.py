# from __future__ import annotations

# from dataclasses import dataclass

# import numpy as np

# from .state import HandAction, HandState


# @dataclass
# class ArmCommand:
#     joint_targets: np.ndarray
#     gripper: float
#     action_label: str


# class SimpleArmRetargeter:
#     """Example retargeter from egocentric hand state to a simple 6-DoF arm command."""

#     def __init__(self, image_size: tuple[int, int] = (960, 720)):
#         self.image_width = float(image_size[0])
#         self.image_height = float(image_size[1])

#     def map(self, hand: HandState, action: HandAction) -> ArmCommand:
#         px, py = hand.palm_center_2d
#         z = hand.palm_depth if hand.palm_depth is not None else 0.4

#         norm_x = np.clip((px / self.image_width - 0.5) * 2.0, -1.0, 1.0)
#         norm_y = np.clip((py / self.image_height - 0.5) * 2.0, -1.0, 1.0)
#         norm_z = np.clip((z - 0.4) / 0.5, -1.0, 1.0)

#         roll = float(np.clip(hand.palm_normal[1], -1.0, 1.0))
#         pitch = float(np.clip(-hand.palm_normal[0], -1.0, 1.0))
#         yaw = float(np.clip(hand.palm_normal[2], -1.0, 1.0))

#         joint_targets = np.array(
#             [
#                 0.9 * norm_x,
#                 -0.7 * norm_y,
#                 0.6 * norm_z,
#                 0.8 * roll,
#                 0.8 * pitch,
#                 0.6 * yaw,
#             ],
#             dtype=np.float32,
#         )

#         return ArmCommand(
#             joint_targets=joint_targets,
#             gripper=float(np.clip(action.gripper_command, 0.0, 1.0)),
#             action_label=action.label,
#         )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import HandAction, HandState
from .gesture_abstraction import WRIST as _WRIST_IDX, FINGER_MCPS as _FINGER_MCPS

# Middle-finger MCP — the wrist→this-point vector gives the hand's in-plane
# rotation, which we map to the gripper's wrist_roll.
_MIDDLE_MCP_IDX = _FINGER_MCPS[2]


@dataclass
class ArmCommand:
    """A Cartesian target for the arm, in the robot base frame (metres).

    `target_xyz` is the position you hand to your IK solver (PyIK).
    `target_rpy` is an approximate orientation (roll, pitch, yaw, radians)
    read from the palm normal.

    `joint_targets` is kept only so older drawing/logging code that reads it
    does not break. It is NOT joint angles — it is just
    [x, y, z, roll, pitch, yaw] concatenated. Do not send it straight to motors.
    """
    target_xyz: np.ndarray
    target_rpy: np.ndarray
    gripper: float
    action_label: str

    @property
    def joint_targets(self) -> np.ndarray:
        return np.concatenate([self.target_xyz, self.target_rpy]).astype(np.float32)


class SimpleArmRetargeter:
    """Map an egocentric hand state to a metric Cartesian arm target.

    What map() does, step by step:
      1. Deproject the palm pixel (px, py) + depth z into a REAL 3D point in
         metres, in the camera frame, using the camera lens numbers (intrinsics).
         This is the fix: x and y stop being pixels and become metres, like z.
      2. Measure that point from a fixed zero — the image centre, at `ref_depth`
         metres. Now all three axes share one unit: metres.
      3. Rotate the camera-frame offset into robot directions with `R_map`
         (a simple sign / axis-swap you verify one axis at a time).
      4. Scale it, clamp it to a safe reach, and add it to the arm's `home_xyz`.
         The result is the position you feed to IK.
    """

    # Camera frame:  +X = right,  +Y = down,  +Z = forward (into the scene).
    # Robot frame:   +X = forward (reach),  +Y = left,  +Z = up.
    # Default mapping: forward <- camera depth, side <- -camera X, up <- -camera Y.
    DEFAULT_R_MAP = np.array(
        [
            [0.0,  0.0, 1.0],   # robot forward  <-  +dz  (hand far / near)
            [-1.0, 0.0, 0.0],   # robot side     <-  -dx  (hand left / right)
            [0.0, -1.0, 0.0],   # robot up       <-  -dy  (hand up / down)
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        intrinsics=None,
        *,
        image_size: tuple[int, int] = (848, 480),
        ref_depth_m: float = 0.40,
        scale: float = 1.0,
        home_xyz: tuple[float, float, float] = (0.20, 0.0, 0.20),
        max_reach_m: float = 0.30,
        r_map: np.ndarray | None = None,
        hfov_deg: float = 60.0,
        freeze_depth_axis: bool = False,
    ):
        # image_size: width, height of the frame the pixels actually come from.
        self.work_w = float(image_size[0])
        self.work_h = float(image_size[1])

        self.ref_depth = float(ref_depth_m)   # the "40 cm" zero plane, in metres
        self.scale = float(scale)             # robot metres per hand metre
        self.home_xyz = np.asarray(home_xyz, dtype=np.float32)  # arm resting pose
        self.max_reach = float(max_reach_m)   # max offset from home, per axis
        self.R_map = (
            self.DEFAULT_R_MAP if r_map is None else np.asarray(r_map, np.float32)
        )
        # When the depth source is unreliable (monocular DA2 on a plain webcam),
        # freeze the camera Z (depth) component so only the image plane drives
        # the arm. RealSense/Orbbec give true metric depth, so leave it False.
        self.freeze_depth_axis = bool(freeze_depth_axis)

        self.fx, self.fy, self.cx, self.cy = self._resolve_intrinsics(
            intrinsics, hfov_deg
        )

    def _resolve_intrinsics(self, intrinsics, hfov_deg):
        # Prefer the camera's real lens numbers (RealSense / Orbbec provide them).
        # A plain webcam reports none, so we estimate them from the field of view;
        # x/y in metres are then approximate but still consistent with z.
        if intrinsics is not None:
            fx, fy = float(intrinsics.fx), float(intrinsics.fy)
            cx, cy = float(intrinsics.ppx), float(intrinsics.ppy)

            # If the intrinsics were measured at a different resolution than the
            # image our pixels come from, scale them to match.
            iw = float(getattr(intrinsics, "width", self.work_w))
            ih = float(getattr(intrinsics, "height", self.work_h))
            if iw > 0 and ih > 0 and (iw != self.work_w or ih != self.work_h):
                sx, sy = self.work_w / iw, self.work_h / ih
                fx, cx = fx * sx, cx * sx
                fy, cy = fy * sy, cy * sy
            return fx, fy, cx, cy

        # Fallback: webcam with unknown lens numbers.
        cx, cy = self.work_w / 2.0, self.work_h / 2.0
        fx = self.work_w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        fy = fx  # assume square pixels
        print(
            "[retargeter] no camera intrinsics given — estimating from "
            f"hfov={hfov_deg} deg. x/y in metres will be approximate."
        )
        return float(fx), float(fy), float(cx), float(cy)

    def map(self, hand: HandState, action: HandAction) -> ArmCommand:
        px, py = hand.palm_center_2d
        z = hand.palm_depth if hand.palm_depth is not None else self.ref_depth

        # 1) pixels + depth  ->  real metres (camera frame). Standard pinhole
        #    back-projection: Xc = (u - cx) * Z / fx,  Yc = (v - cy) * Z / fy.
        #    All three are now metres. z appears on purpose: a pixel's real
        #    lateral size grows with distance.
        Xc = (float(px) - self.cx) * z / self.fx   # +right, metres
        Yc = (float(py) - self.cy) * z / self.fy   # +down,  metres
        Zc = float(z)                              # +forward (depth), metres

        # 2) measure from the zero point (optical axis at ref_depth) — metres.
        #    Hand at image centre and at ref_depth => delta = 0 => arm at home.
        dz = 0.0 if self.freeze_depth_axis else (Zc - self.ref_depth)
        delta_cam = np.array([Xc, Yc, dz], dtype=np.float32)

        # 3) camera directions -> robot directions.
        #    If one direction moves the wrong way, flip its sign in R_map.
        delta_robot = self.R_map @ delta_cam

        # 4) scale, clamp to a safe reach, add to home -> IK position target.
        delta_robot = np.clip(delta_robot * self.scale, -self.max_reach, self.max_reach)
        target_xyz = (self.home_xyz + delta_robot).astype(np.float32)

        # Orientation. A 5-DOF arm can't do position + full orientation, so we
        # only produce ONE rotation that the driver sends straight to wrist_roll
        # (decoupled from the position IK — that is what stops the flailing).
        #
        # roll = the hand's IN-PLANE rotation, from 2D keypoints (robust): the
        # angle of the wrist -> middle-MCP vector. Fingers pointing up is 0.
        kp = np.asarray(hand.keypoints, dtype=np.float32)
        v  = kp[_MIDDLE_MCP_IDX, :2] - kp[_WRIST_IDX, :2]
        roll = float(np.arctan2(float(v[1]), float(v[0])) + np.pi / 2.0)
        roll = float(np.arctan2(np.sin(roll), np.cos(roll)))   # wrap to [-pi, pi]

        # pitch/yaw from the palm normal are kept for the overlay/logging only —
        # they are NOT sent to the 5-DOF IK.
        nf, ns, nu = self.R_map @ np.asarray(hand.palm_normal, dtype=np.float32)
        yaw = float(np.arctan2(ns, nf))                    # turn left / right
        pitch = float(np.arcsin(np.clip(nu, -1.0, 1.0)))   # tilt up / down
        target_rpy = np.array([roll, pitch, yaw], dtype=np.float32)

        return ArmCommand(
            target_xyz=target_xyz,
            target_rpy=target_rpy,
            gripper=float(np.clip(action.gripper_command, 0.0, 1.0)),
            action_label=action.label,
        )