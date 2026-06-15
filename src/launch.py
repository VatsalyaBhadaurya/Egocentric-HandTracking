
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
from src.pipeline import RobotLearningHandPipeline
from src.state import HandAction

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

# Hand workspace bounds (metres) — maps raw palm_center_3d to robot workspace
# Tune these based on where your camera is relative to the arm
HAND_X_RANGE = (-0.4, 0.4)    # left/right
HAND_Y_RANGE = (-0.4, 0.4)    # up/down
HAND_Z_RANGE = (0.2,  0.8)    # near/far (depth from camera)

# Robot workspace target centre and reach (metres)
ROBOT_TARGET_CENTER = np.array([0.15, 0.0, 0.20])   # metres from base
ROBOT_REACH         = 0.25                            # max reach radius


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
              target_normal: np.ndarray | None = None) -> list[float]:
        """
        Solve IK for target_xyz position.
        Returns 5 joint angles [shoulder_pan .. wrist_roll] clamped to limits.
        Falls back to HOME_JOINTS if IK fails.
        """
        try:
            if target_normal is not None:
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
            print(f"[IK] solve failed: {e} — using HOME")
            return HOME_JOINTS.copy()

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
    Normalises raw palm_center_3d (camera frame, metres) to robot workspace.

    Camera frame:  X=right, Y=down,  Z=depth (away from camera)
    Robot frame:   X=front, Y=left,  Z=up
    """
    hx, hy, hz = palm_xyz

    # normalise each axis to [-1, 1]
    nx = np.clip((hx - HAND_X_RANGE[0]) / (HAND_X_RANGE[1] - HAND_X_RANGE[0]) * 2 - 1, -1, 1)
    ny = np.clip((hy - HAND_Y_RANGE[0]) / (HAND_Y_RANGE[1] - HAND_Y_RANGE[0]) * 2 - 1, -1, 1)
    nz = np.clip((hz - HAND_Z_RANGE[0]) / (HAND_Z_RANGE[1] - HAND_Z_RANGE[0]) * 2 - 1, -1, 1)

    # map to robot target around centre
    rx = ROBOT_TARGET_CENTER[0] + nz * ROBOT_REACH   # depth  → robot forward
    ry = ROBOT_TARGET_CENTER[1] - nx * ROBOT_REACH   # hand X → robot lateral
    rz = ROBOT_TARGET_CENTER[2] - ny * ROBOT_REACH   # hand Y → robot height

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

        self.publisher_ = self.create_publisher(JointState, TOPIC, 10)
        self.timer      = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)

        self.get_logger().info(
            f"SO101ArmPublisher ready — publishing to '{TOPIC}' at {PUBLISH_HZ} Hz"
        )

    def timer_callback(self):
        frame_state = self.pipeline.last_frame_state

        # ── no hand detected → hold home position ────────────────────────────
        if frame_state is None or not frame_state.hands or not frame_state.actions:
            self._publish(
                joints  = HOME_JOINTS,
                gripper = 1.0,
                gesture = "NONE",
                xyz     = None,
            )
            return

        hand   = frame_state.hands[0]     # HandState  — 3D pose
        action = frame_state.actions[0]   # HandAction — gesture + gripper

        # ── 3D pose from pipeline (already computed, no new work needed) ─────
        palm_xyz    = hand.palm_center_3d   # [x, y, z] metres, camera frame
        palm_normal = hand.palm_normal      # [nx, ny, nz] unit vector
         
        palm_xyz_safe = palm_xyz.copy()
        if abs(palm_xyz_safe[2]) < 0.1:      # Z near zero = not calibrated
            palm_xyz_safe[2] = 0.4            # assume 40cm from camera
    # ─────────────────────────────────────────────────────────────────

        robot_xyz    = map_hand_to_robot(palm_xyz_safe)
        # ── map camera workspace → robot workspace ────────────────────────────
        #robot_xyz = map_hand_to_robot(palm_xyz)

        # ── solve IK → 5 joint angles ─────────────────────────────────────────
        joint_angles = self.ik_solver.solve(robot_xyz, palm_normal)

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
    parser.add_argument("--camera-index", type=int,   default=2)
    parser.add_argument("--score-thr",    type=float, default=0.08)
    parser.add_argument("--depth-model",
                        default=str(PROJECT_ROOT / "checkpoints" / "depth_anything_v2"))
    parser.add_argument("--da2-encoder",  default="vits",
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--infer-scale",  type=float, default=0.55)
    parser.add_argument("--urdf",
                        default=str(URDF_PATH),
                        help="Path to so101.urdf")
    args = parser.parse_args()

    rclpy.init()

    pipeline = RobotLearningHandPipeline(
        depth_model  = args.depth_model,
        device       = args.device,
        da2_encoder  = args.da2_encoder,
        camera_index = args.camera_index,
        score_thr    = args.score_thr,
        infer_scale  = args.infer_scale,
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