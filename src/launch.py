# # testing for whole body control with ik solver 
# """
# joint_commands.py  —  SO-101 arm full control via hand tracking pipeline

# Publishes all 6 joints to /joint_commands as a JointState message.
# Joints 1-5 (arm) are solved via IKPy from hand 3D pose.
# Joint 6 (gripper) is controlled by hand gesture.

# Gripper logic:
#     OPEN PALM  -> gripper = 1.0  (open)
#     PINCH      -> gripper = 0.0  (closed)
#     GRASP      -> gripper = 0.0  (closed)
#     No hand    -> gripper = 1.0  (default open)

# SO-101 joint names (from URDF):
#     shoulder_pan   bounds=(-1.91986,  1.91986)   [index 1]
#     shoulder_lift  bounds=(-1.74533,  1.74533)   [index 2]
#     elbow_flex     bounds=(-1.69,     1.69)       [index 3]
#     wrist_flex     bounds=(-1.65806,  1.65806)   [index 4]
#     wrist_roll     bounds=(-2.74385,  2.84121)   [index 5]
#     gripper        bounds=(-0.174533, 1.74533)   [separate]
# """

# import argparse
# import threading
# import sys
# import os
# import warnings
# from pathlib import Path

# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from ikpy.chain import Chain

# # ── resolve project root (ego-centric/) from src/joint_commands.py ───────────
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # ── src/ uses relative imports so we must treat it as a package ──────────────
# sys.path.insert(0, str(PROJECT_ROOT))
# from src.pipeline import RobotLearningHandPipeline, CAM_W, CAM_H
# from src.state import HandAction
# from src.one_euro_filter import OneEuroFilter

# # ── config ────────────────────────────────────────────────────────────────────
# URDF_PATH  = Path(__file__).resolve().parent / "so101.urdf"   # put urdf in src/
# TOPIC      = "/joint_commands"
# PUBLISH_HZ = 30.0

# # SO-101 joint names published in JointState (must match your controller)
# SO101_JOINT_NAMES = [
#     "shoulder_pan",
#     "shoulder_lift",
#     "elbow_flex",
#     "wrist_flex",
#     "wrist_roll",
#     "gripper",
# ]

# # Joint limits from URDF (radians) — used for clamping IK output
# # SO101_JOINT_LIMITS = {
# #     "shoulder_pan":  (-1.91986,  1.91986),
# #     "shoulder_lift": (-1.74533,  1.74533),
# #     "elbow_flex":    (-1.69,     1.69),
# #     "wrist_flex":    (-1.65806,  1.65806),
# #     "wrist_roll":    (-2.74385,  2.84121),
# #     "gripper":       (-0.174533, 1.74533),
# # }
# SO101_JOINT_LIMITS = {
#     "shoulder_pan":  (-1.91986,  1.91986),
#     "shoulder_lift": (-1.74533,  1.74533),
#     "elbow_flex":    (-1.69,     1.69),
#     "wrist_flex":    (-1.65806,  1.65806),
#     "wrist_roll":    (-2.74385,  2.84121),
# }
# # Safe home position used when no hand is detected
# HOME_JOINTS = [0.0, -1.5, 1.6, 0.9, 0.0]   # shoulder_pan → wrist_roll

# # Hand workspace bounds — maps raw palm_center_3d to robot workspace.
# # NB: palm_center_3d packs [pixel_x, pixel_y, depth_m] (see build_hand_state),
# # i.e. X/Y are *image pixels*, only Z is metric. So X/Y are normalised against
# # the frame size below, and only Z uses a metric range.
# HAND_X_RANGE = (0.0, float(CAM_W))   # image x pixels (left → right)
# HAND_Y_RANGE = (0.0, float(CAM_H))   # image y pixels (top → bottom)
# HAND_Z_RANGE = (0.2,  0.8)           # near/far depth from camera (metres)

# # Robot workspace target centre and reach (metres)
# ROBOT_TARGET_CENTER = np.array([0.15, 0.0, 0.20])   # metres from base
# ROBOT_REACH         = 0.25                            # max reach radius

# # Solve position only. A 5-DOF arm can't hit arbitrary position+orientation, so
# # forcing the noisy palm normal makes IK contort/fail. Set True only if you've
# # stabilised the normal and accept reduced positional accuracy.
# USE_ORIENTATION = False

# # The forward axis comes from depth. On the webcam path that's *uncalibrated*
# # DA2 disparity (not metres) unless you press 'c' to calibrate, so it's noisy.
# # When True, the forward target is frozen at the workspace centre and the arm
# # tracks only lateral/height (image plane) — far less "weird" without depth.
# FREEZE_DEPTH_AXIS = True


# # ── IK solver wrapper ─────────────────────────────────────────────────────────

# class SO101IKSolver:
#     """Wraps IKPy chain for the SO-101 arm."""

#     # active_links_mask:
#     #   index 0 = Base link     (fixed  → False)
#     #   index 1 = shoulder_pan  (active → True)
#     #   index 2 = shoulder_lift (active → True)
#     #   index 3 = elbow_flex    (active → True)
#     #   index 4 = wrist_flex    (active → True)
#     #   index 5 = wrist_roll    (active → True)
#     #   index 6 = gripper_frame (fixed  → False)
#     ACTIVE_MASK = [False, True, True, True, True, True, False]

#     def __init__(self, urdf_path: str):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             self.chain = Chain.from_urdf_file(
#                 urdf_path,
#                 active_links_mask=self.ACTIVE_MASK,
#             )
#         print(f"[IK] Loaded chain from {urdf_path}")
#         print(f"[IK] Links: {[l.name for l in self.chain.links]}")

#         # seed: last successful joint solution (warm start speeds up IK)
#         self._last_joints = np.zeros(7)   # 7 = len(chain.links)

#     def solve(self, target_xyz: np.ndarray,
#               target_normal: np.ndarray | None = None) -> list[float] | None:
#         """
#         Solve IK for target_xyz position.
#         Returns 5 joint angles [shoulder_pan .. wrist_roll] clamped to limits,
#         or None if IK fails (caller should hold the previous pose rather than
#         snap to HOME, which causes the arm to flail on intermittent failures).

#         target_normal is ignored unless USE_ORIENTATION is set: a 5-DOF arm
#         cannot generally hit an arbitrary position *and* orientation, so forcing
#         the noisy palm normal makes the solver contort/fail. Position-only IK is
#         far calmer for teleoperation.
#         """
#         try:
#             if target_normal is not None and USE_ORIENTATION:
#                 # orientation_mode="Z" expects a plain (3,) unit vector —
#                 # the desired direction of the end-effector Z axis.
#                 # palm_normal is already a unit vector, pass it directly.
#                 z_axis = np.asarray(target_normal, dtype=np.float64).flatten()[:3]
#                 z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
#                 solution = self.chain.inverse_kinematics(
#                     target_position=target_xyz.tolist(),
#                     target_orientation=z_axis,
#                     orientation_mode="Z",
#                     initial_position=self._last_joints,
#                 )
#             else:
#                 solution = self.chain.inverse_kinematics(
#                     target_position=target_xyz.tolist(),
#                     initial_position=self._last_joints,
#                 )

#             self._last_joints = solution.copy()

#             # extract the 5 active joints (indices 1-5), skip base(0) and gripper_frame(6)
#             joints = solution[1:6].tolist()

#             # clamp each joint to its URDF limits
#             names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
#             #names  = ["Rotation", "Pitch", "Elbow",
#              #         "Wrist_Pitch", "Wrist_Roll"]
#             joints = [
#                 float(np.clip(j, SO101_JOINT_LIMITS[n][0], SO101_JOINT_LIMITS[n][1]))
#                 for j, n in zip(joints, names)
#             ]
#             return joints

#         except Exception as e:
#             print(f"[IK] solve failed: {e} — holding previous pose")
#             return None

#     @staticmethod
#     def _normal_to_matrix(normal: np.ndarray) -> np.ndarray:
#         """
#         Converts palm normal vector to a 3x3 rotation matrix.
#         Z-axis of end-effector is aligned to palm normal.
#         """
#         z = normal / (np.linalg.norm(normal) + 1e-9)
#         # pick an arbitrary up vector, avoid parallel
#         up = np.array([0.0, 0.0, 1.0])
#         if abs(np.dot(z, up)) > 0.9:
#             up = np.array([0.0, 1.0, 0.0])
#         x = np.cross(up, z)
#         x /= np.linalg.norm(x) + 1e-9
#         y = np.cross(z, x)
#         return np.column_stack([x, y, z])


# # ── hand workspace → robot workspace mapping ─────────────────────────────────

# def map_hand_to_robot(palm_xyz: np.ndarray) -> np.ndarray:
#     """
#     Normalises raw palm_center_3d to the robot workspace.

#     palm_xyz is [pixel_x, pixel_y, depth_m]: X/Y are image pixels, Z is metres.

#     Camera/image:  X=right, Y=down,  Z=depth (away from camera)
#     Robot frame:   X=front, Y=left,  Z=up

#     The signs on nx/ny below set the mirror direction. If the arm tracks the
#     mirror image of your hand, flip the sign on the corresponding line.
#     """
#     hx, hy, hz = palm_xyz

#     # normalise each axis to [-1, 1]; X/Y by image size, Z by metric depth range
#     nx = np.clip((hx - HAND_X_RANGE[0]) / (HAND_X_RANGE[1] - HAND_X_RANGE[0]) * 2 - 1, -1, 1)
#     ny = np.clip((hy - HAND_Y_RANGE[0]) / (HAND_Y_RANGE[1] - HAND_Y_RANGE[0]) * 2 - 1, -1, 1)
#     nz = np.clip((hz - HAND_Z_RANGE[0]) / (HAND_Z_RANGE[1] - HAND_Z_RANGE[0]) * 2 - 1, -1, 1)

#     # map to robot target around centre
#     forward = 0.0 if FREEZE_DEPTH_AXIS else nz * ROBOT_REACH
#     rx = ROBOT_TARGET_CENTER[0] + forward            # depth   → robot forward
#     ry = ROBOT_TARGET_CENTER[1] - nx * ROBOT_REACH   # image X → robot lateral
#     rz = ROBOT_TARGET_CENTER[2] - ny * ROBOT_REACH   # image Y → robot height

#     return np.array([rx, ry, rz], dtype=np.float32)


# # ── gesture → gripper ─────────────────────────────────────────────────────────

# def gesture_to_gripper(action: HandAction) -> float:
#     """
#     OPEN PALM  -> 1.0  (open)
#     PINCH      -> 0.0  (closed)
#     GRASP      -> 0.0  (closed)
#     """
#     gesture = action.meta.get("gesture", "GRASP")
#     return 1.0 if gesture == "OPEN PALM" else 0.0


# # ── ROS2 node ─────────────────────────────────────────────────────────────────

# class SO101ArmPublisher(Node):

#     def __init__(self, pipeline: RobotLearningHandPipeline, urdf_path: str):
#         super().__init__("so101_arm_publisher")

#         self.pipeline  = pipeline
#         self.ik_solver = SO101IKSolver(urdf_path)

#         # Smooth the IK target and the published joints so per-frame tracking
#         # noise doesn't reach the servos. Last published joints are held when IK
#         # fails or no hand is visible.
#         self._xyz_filter   = OneEuroFilter(freq=PUBLISH_HZ, min_cutoff=0.8, beta=0.02)
#         self._joint_filter = OneEuroFilter(freq=PUBLISH_HZ, min_cutoff=1.0, beta=0.03)
#         self._last_joints  = list(HOME_JOINTS)

#         self.publisher_ = self.create_publisher(JointState, TOPIC, 10)
#         self.timer      = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

#         self.get_logger().info(
#             f"SO101ArmPublisher ready — publishing to '{TOPIC}' at {PUBLISH_HZ} Hz"
#         )

#     def timer_callback(self):
#         frame_state = self.pipeline.last_frame_state

#         # ── pick the first *visible* hand; reject phantom detections ─────────
#         # The detector-less top-down pose model emits keypoints every frame, so
#         # gating on HandState.visible is what keeps spurious detections from
#         # driving the arm.
#         hand = action = None
#         if frame_state is not None and frame_state.actions:
#             for hs, act in zip(frame_state.hands, frame_state.actions):
#                 if hs.visible:
#                     hand, action = hs, act
#                     break

#         # ── no real hand → hold last pose, reset smoothing for clean re-acq ──
#         if hand is None:
#             self._xyz_filter.reset()
#             self._joint_filter.reset()
#             self._publish(
#                 joints  = self._last_joints,
#                 gripper = 1.0,
#                 gesture = "NONE",
#                 xyz     = None,
#             )
#             return

#         # ── 3D pose from pipeline (already computed, no new work needed) ─────
#         palm_xyz    = hand.palm_center_3d   # [pixel_x, pixel_y, depth_m]
#         palm_normal = hand.palm_normal      # [nx, ny, nz] unit vector

#         palm_xyz_safe = palm_xyz.copy()
#         if abs(palm_xyz_safe[2]) < 0.1:      # Z near zero = not calibrated
#             palm_xyz_safe[2] = 0.4            # assume 40cm from camera

#         # ── map image/depth → robot workspace, then smooth the target ─────────
#         robot_xyz = map_hand_to_robot(palm_xyz_safe)
#         robot_xyz = self._xyz_filter(robot_xyz, freq=PUBLISH_HZ)

#         # ── solve IK → 5 joint angles; hold last pose if it fails ─────────────
#         joint_angles = self.ik_solver.solve(robot_xyz, palm_normal)
#         if joint_angles is None:
#             joint_angles = self._last_joints
#         else:
#             joint_angles = self._joint_filter(
#                 np.asarray(joint_angles, dtype=np.float64), freq=PUBLISH_HZ).tolist()
#             self._last_joints = joint_angles

#         # ── gripper from gesture ──────────────────────────────────────────────
#         gripper_val = gesture_to_gripper(action)
#         gesture     = action.meta.get("gesture", "GRASP")

#         self._publish(
#             joints  = joint_angles,
#             gripper = gripper_val,
#             gesture = gesture,
#             xyz     = robot_xyz,
#         )

#     def _publish(self, joints: list[float], gripper: float,
#                  gesture: str = "", xyz=None):
#         msg                  = JointState()
#         msg.header.stamp     = self.get_clock().now().to_msg()
#         msg.name             = SO101_JOINT_NAMES
#         msg.position         = [*joints, gripper]   # 5 arm joints + gripper
#         msg.velocity         = []
#         msg.effort           = []

#         self.publisher_.publish(msg)

#         xyz_str = f"xyz=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]" if xyz is not None else "xyz=HOME"
#         j_str   = ", ".join(f"{j:.3f}" for j in joints)
#         self.get_logger().info(
#             f"gesture={gesture:<10}  {xyz_str}  "
#             f"joints=[{j_str}]  gripper={gripper:.1f}"
#         )


# # ── entry point ───────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="SO-101 Arm Publisher via Hand Tracking")
#     parser.add_argument("--device",       default="auto")
#     parser.add_argument("--camera-index", type=int,   default=2)
#     parser.add_argument("--score-thr",    type=float, default=0.02)
#     parser.add_argument("--depth-model",
#                         default=str(PROJECT_ROOT / "checkpoints" / "depth_anything_v2"))
#     parser.add_argument("--da2-encoder",  default="vits",
#                         choices=["vits", "vitb", "vitl"])
#     parser.add_argument("--infer-scale",  type=float, default=0.7)
#     parser.add_argument("--urdf",
#                         default=str(URDF_PATH),
#                         help="Path to so101.urdf")
#     args = parser.parse_args()

#     rclpy.init()

#     pipeline = RobotLearningHandPipeline(
#         depth_model  = args.depth_model,
#         device       = args.device,
#         da2_encoder  = args.da2_encoder,
#         camera_index = args.camera_index,
#         score_thr    = args.score_thr,
#         infer_scale  = args.infer_scale,
#     )

#     node       = SO101ArmPublisher(pipeline, urdf_path=args.urdf)
#     ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
#     ros_thread.start()

#     try:
#         pipeline.run()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()

# testing for whole body control with ik solver 
"""
joint_commands.py  —  SO-101 arm full control via hand tracking pipeline

Publishes all 6 joints to /joint_commands as a JointState message.
Joints 1-5 (arm) are solved via IKPy from hand 3D pose.
Joint 6 (gripper) is controlled by hand gesture.

Gripper logic:
    OPEN PALM  -> gripper = 1.0  (open)
    PINCH      -> gripper = 0.0  (closed)
    GRASP      -> gripper = 0.0  (closed)
    No hand    -> gripper = 1.0  (default open)

SO-101 joint names (from URDF):
    shoulder_pan   bounds=(-1.91986,  1.91986)   [index 1]
    shoulder_lift  bounds=(-1.74533,  1.74533)   [index 2]
    elbow_flex     bounds=(-1.69,     1.69)       [index 3]
    wrist_flex     bounds=(-1.65806,  1.65806)   [index 4]
    wrist_roll     bounds=(-2.74385,  2.84121)   [index 5]
    gripper        bounds=(-0.174533, 1.74533)   [separate]
"""

import argparse
import threading
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ikpy.chain import Chain

# ── resolve project root (ego-centric/) from src/joint_commands.py ───────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── src/ uses relative imports so we must treat it as a package ──────────────
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline import RobotLearningHandPipeline, CAM_W, CAM_H
from src.state import HandAction
from src.one_euro_filter import OneEuroFilter

# ── config ────────────────────────────────────────────────────────────────────
URDF_PATH  = Path(__file__).resolve().parent / "so101.urdf"   # put urdf in src/
TOPIC      = "/joint_commands"
PUBLISH_HZ = 30.0

# SO-101 joint names published in JointState (must match your controller)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Joint limits from URDF (radians) — used for clamping IK output
# SO101_JOINT_LIMITS = {
#     "shoulder_pan":  (-1.91986,  1.91986),
#     "shoulder_lift": (-1.74533,  1.74533),
#     "elbow_flex":    (-1.69,     1.69),
#     "wrist_flex":    (-1.65806,  1.65806),
#     "wrist_roll":    (-2.74385,  2.84121),
#     "gripper":       (-0.174533, 1.74533),
# }
SO101_JOINT_LIMITS = {
    "shoulder_pan":  (-1.91986,  1.91986),
    "shoulder_lift": (-1.74533,  1.74533),
    "elbow_flex":    (-1.69,     1.69),
    "wrist_flex":    (-1.65806,  1.65806),
    "wrist_roll":    (-2.74385,  2.84121),
}
# Safe home position used when no hand is detected
HOME_JOINTS = [0.0, -1.5, 1.6, 0.9, 0.0]   # shoulder_pan → wrist_roll

# Hand workspace bounds — maps raw palm_center_3d to robot workspace.
# NB: palm_center_3d packs [pixel_x, pixel_y, depth_m] (see build_hand_state),
# i.e. X/Y are *image pixels*, only Z is metric. So X/Y are normalised against
# the frame size below, and only Z uses a metric range.
HAND_X_RANGE = (0.0, float(CAM_W))   # image x pixels (left → right)
HAND_Y_RANGE = (0.0, float(CAM_H))   # image y pixels (top → bottom)
HAND_Z_RANGE = (0.2,  0.8)           # near/far depth from camera (metres)

# Robot workspace target centre and reach (metres)
ROBOT_TARGET_CENTER = np.array([0.15, 0.0, 0.20])   # metres from base
ROBOT_REACH         = 0.25                            # max reach radius

# Solve position only. A 5-DOF arm can't hit arbitrary position+orientation, so
# forcing the noisy palm normal makes IK contort/fail. Set True only if you've
# stabilised the normal and accept reduced positional accuracy.
USE_ORIENTATION = False

# The forward axis comes from depth. With a RealSense it's true metric depth and
# this should be False. On the webcam (DA2) path depth is monocular relative
# depth that drifts/jumps even after calibration, so main() flips this to True
# for that backend, freezing the forward target and tracking only the image
# plane. Tune HAND_Z_RANGE to your actual hand-to-camera working distance.
FREEZE_DEPTH_AXIS = False


# ── IK solver wrapper ─────────────────────────────────────────────────────────

class SO101IKSolver:
    """Wraps IKPy chain for the SO-101 arm."""

    # active_links_mask:
    #   index 0 = Base link     (fixed  → False)
    #   index 1 = shoulder_pan  (active → True)
    #   index 2 = shoulder_lift (active → True)
    #   index 3 = elbow_flex    (active → True)
    #   index 4 = wrist_flex    (active → True)
    #   index 5 = wrist_roll    (active → True)
    #   index 6 = gripper_frame (fixed  → False)
    ACTIVE_MASK = [False, True, True, True, True, True, False]

    def __init__(self, urdf_path: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.chain = Chain.from_urdf_file(
                urdf_path,
                active_links_mask=self.ACTIVE_MASK,
            )
        print(f"[IK] Loaded chain from {urdf_path}")
        print(f"[IK] Links: {[l.name for l in self.chain.links]}")

        # seed: last successful joint solution (warm start speeds up IK)
        self._last_joints = np.zeros(7)   # 7 = len(chain.links)

    def solve(self, target_xyz: np.ndarray,
              target_normal: np.ndarray | None = None) -> list[float] | None:
        """
        Solve IK for target_xyz position.
        Returns 5 joint angles [shoulder_pan .. wrist_roll] clamped to limits,
        or None if IK fails (caller should hold the previous pose rather than
        snap to HOME, which causes the arm to flail on intermittent failures).

        target_normal is ignored unless USE_ORIENTATION is set: a 5-DOF arm
        cannot generally hit an arbitrary position *and* orientation, so forcing
        the noisy palm normal makes the solver contort/fail. Position-only IK is
        far calmer for teleoperation.
        """
        try:
            if target_normal is not None and USE_ORIENTATION:
                # orientation_mode="Z" expects a plain (3,) unit vector —
                # the desired direction of the end-effector Z axis.
                # palm_normal is already a unit vector, pass it directly.
                z_axis = np.asarray(target_normal, dtype=np.float64).flatten()[:3]
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
                solution = self.chain.inverse_kinematics(
                    target_position=target_xyz.tolist(),
                    target_orientation=z_axis,
                    orientation_mode="Z",
                    initial_position=self._last_joints,
                )
            else:
                solution = self.chain.inverse_kinematics(
                    target_position=target_xyz.tolist(),
                    initial_position=self._last_joints,
                )

            self._last_joints = solution.copy()

            # extract the 5 active joints (indices 1-5), skip base(0) and gripper_frame(6)
            joints = solution[1:6].tolist()

            # clamp each joint to its URDF limits
            names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            #names  = ["Rotation", "Pitch", "Elbow",
             #         "Wrist_Pitch", "Wrist_Roll"]
            joints = [
                float(np.clip(j, SO101_JOINT_LIMITS[n][0], SO101_JOINT_LIMITS[n][1]))
                for j, n in zip(joints, names)
            ]
            return joints

        except Exception as e:
            print(f"[IK] solve failed: {e} — holding previous pose")
            return None

    @staticmethod
    def _normal_to_matrix(normal: np.ndarray) -> np.ndarray:
        """
        Converts palm normal vector to a 3x3 rotation matrix.
        Z-axis of end-effector is aligned to palm normal.
        """
        z = normal / (np.linalg.norm(normal) + 1e-9)
        # pick an arbitrary up vector, avoid parallel
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        return np.column_stack([x, y, z])


# ── hand workspace → robot workspace mapping ─────────────────────────────────

def map_hand_to_robot(palm_xyz: np.ndarray) -> np.ndarray:
    """
    Normalises raw palm_center_3d to the robot workspace.

    palm_xyz is [pixel_x, pixel_y, depth_m]: X/Y are image pixels, Z is metres.

    Camera/image:  X=right, Y=down,  Z=depth (away from camera)
    Robot frame:   X=front, Y=left,  Z=up

    The signs on nx/ny below set the mirror direction. If the arm tracks the
    mirror image of your hand, flip the sign on the corresponding line.
    """
    hx, hy, hz = palm_xyz

    # normalise each axis to [-1, 1]; X/Y by image size, Z by metric depth range
    nx = np.clip((hx - HAND_X_RANGE[0]) / (HAND_X_RANGE[1] - HAND_X_RANGE[0]) * 2 - 1, -1, 1)
    ny = np.clip((hy - HAND_Y_RANGE[0]) / (HAND_Y_RANGE[1] - HAND_Y_RANGE[0]) * 2 - 1, -1, 1)
    nz = np.clip((hz - HAND_Z_RANGE[0]) / (HAND_Z_RANGE[1] - HAND_Z_RANGE[0]) * 2 - 1, -1, 1)

    # map to robot target around centre
    forward = 0.0 if FREEZE_DEPTH_AXIS else nz * ROBOT_REACH
    rx = ROBOT_TARGET_CENTER[0] + forward            # depth   → robot forward
    ry = ROBOT_TARGET_CENTER[1] - nx * ROBOT_REACH   # image X → robot lateral
    rz = ROBOT_TARGET_CENTER[2] - ny * ROBOT_REACH   # image Y → robot height

    return np.array([rx, ry, rz], dtype=np.float32)


# ── gesture → gripper ─────────────────────────────────────────────────────────

def gesture_to_gripper(action: HandAction) -> float:
    """
    OPEN PALM  -> 1.0  (open)
    PINCH      -> 0.0  (closed)
    GRASP      -> 0.0  (closed)
    """
    gesture = action.meta.get("gesture", "GRASP")
    return 1.0 if gesture == "OPEN PALM" else 0.0


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class SO101ArmPublisher(Node):

    def __init__(self, pipeline: RobotLearningHandPipeline, urdf_path: str):
        super().__init__("so101_arm_publisher")

        self.pipeline  = pipeline
        self.ik_solver = SO101IKSolver(urdf_path)

        # Smooth the IK target and the published joints so per-frame tracking
        # noise doesn't reach the servos. Last published joints are held when IK
        # fails or no hand is visible.
        self._xyz_filter   = OneEuroFilter(freq=PUBLISH_HZ, min_cutoff=0.8, beta=0.02)
        self._joint_filter = OneEuroFilter(freq=PUBLISH_HZ, min_cutoff=1.0, beta=0.03)
        self._last_joints  = list(HOME_JOINTS)

        self.publisher_ = self.create_publisher(JointState, TOPIC, 10)
        self.timer      = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"SO101ArmPublisher ready — publishing to '{TOPIC}' at {PUBLISH_HZ} Hz"
        )

    def timer_callback(self):
        frame_state = self.pipeline.last_frame_state

        # ── pick the first *visible* hand; reject phantom detections ─────────
        # The detector-less top-down pose model emits keypoints every frame, so
        # gating on HandState.visible is what keeps spurious detections from
        # driving the arm.
        hand = action = None
        if frame_state is not None and frame_state.actions:
            for hs, act in zip(frame_state.hands, frame_state.actions):
                if hs.visible:
                    hand, action = hs, act
                    break

        # ── no real hand → hold last pose, reset smoothing for clean re-acq ──
        if hand is None:
            self._xyz_filter.reset()
            self._joint_filter.reset()
            self._publish(
                joints  = self._last_joints,
                gripper = 1.0,
                gesture = "NONE",
                xyz     = None,
            )
            return

        # ── 3D pose from pipeline (already computed, no new work needed) ─────
        palm_xyz    = hand.palm_center_3d   # [pixel_x, pixel_y, depth_m]
        palm_normal = hand.palm_normal      # [nx, ny, nz] unit vector

        palm_xyz_safe = palm_xyz.copy()
        if abs(palm_xyz_safe[2]) < 0.1:      # Z near zero = not calibrated
            palm_xyz_safe[2] = 0.4            # assume 40cm from camera

        # ── map image/depth → robot workspace, then smooth the target ─────────
        robot_xyz = map_hand_to_robot(palm_xyz_safe)
        robot_xyz = self._xyz_filter(robot_xyz, freq=PUBLISH_HZ)

        # ── solve IK → 5 joint angles; hold last pose if it fails ─────────────
        joint_angles = self.ik_solver.solve(robot_xyz, palm_normal)
        if joint_angles is None:
            joint_angles = self._last_joints
        else:
            joint_angles = self._joint_filter(
                np.asarray(joint_angles, dtype=np.float64), freq=PUBLISH_HZ).tolist()
            self._last_joints = joint_angles

        # ── gripper from gesture ──────────────────────────────────────────────
        gripper_val = gesture_to_gripper(action)
        gesture     = action.meta.get("gesture", "GRASP")

        self._publish(
            joints  = joint_angles,
            gripper = gripper_val,
            gesture = gesture,
            xyz     = robot_xyz,
        )

    def _publish(self, joints: list[float], gripper: float,
                 gesture: str = "", xyz=None):
        msg                  = JointState()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.name             = SO101_JOINT_NAMES
        msg.position         = [*joints, gripper]   # 5 arm joints + gripper
        msg.velocity         = []
        msg.effort           = []

        self.publisher_.publish(msg)

        xyz_str = f"xyz=[{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}]" if xyz is not None else "xyz=HOME"
        j_str   = ", ".join(f"{j:.3f}" for j in joints)
        self.get_logger().info(
            f"gesture={gesture:<10}  {xyz_str}  "
            f"joints=[{j_str}]  gripper={gripper:.1f}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SO-101 Arm Publisher via Hand Tracking")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--camera-backend", default="realsense",
                        choices=["realsense", "opencv"],
                        help="realsense = true metric depth (needed for the forward axis); "
                             "opencv = webcam + monocular DA2 depth")
    parser.add_argument("--camera-index", type=int,   default=2)
    parser.add_argument("--rs-width",     type=int,   default=1280)
    parser.add_argument("--rs-height",    type=int,   default=720)
    parser.add_argument("--rs-fps",       type=int,   default=30)
    parser.add_argument("--score-thr",    type=float, default=0.2)
    parser.add_argument("--depth-model",
                        default=str(PROJECT_ROOT / "checkpoints" / "depth_anything_v2"))
    parser.add_argument("--da2-encoder",  default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--infer-scale",  type=float, default=0.7)
    parser.add_argument("--urdf",
                        default=str(URDF_PATH),
                        help="Path to so101.urdf")
    args = parser.parse_args()

    # The webcam path can't give stable metric depth, so freeze the forward axis
    # there; the RealSense path has real depth, so leave it active.
    if args.camera_backend == "opencv":
        global FREEZE_DEPTH_AXIS
        FREEZE_DEPTH_AXIS = True

    rclpy.init()

    pipeline = RobotLearningHandPipeline(
        depth_model    = args.depth_model,
        device         = args.device,
        da2_encoder    = args.da2_encoder,
        camera_backend = args.camera_backend,
        camera_index   = args.camera_index,
        rs_width       = args.rs_width,
        rs_height      = args.rs_height,
        rs_fps         = args.rs_fps,
        score_thr      = args.score_thr,
        infer_scale    = args.infer_scale,
    )

    node       = SO101ArmPublisher(pipeline, urdf_path=args.urdf)
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()