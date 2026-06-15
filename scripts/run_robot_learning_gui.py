# Author: adith
# CLI entry point for the Robot Learning Hand Pipeline with argument parsing and session report export

import argparse, sys
from pathlib import Path

THIS         = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RobotLearningHandPipeline


def main():
    # Parses command-line arguments and launches the RobotLearningHandPipeline.
    parser = argparse.ArgumentParser(
        description="Robot Learning Hand Pipeline")
    parser.add_argument("--device",      default="auto",
                        help="'auto' picks cuda if available, else cpu")
    parser.add_argument("--camera-backend", default="opencv",
                        choices=["opencv", "realsense"],
                        help="'opencv' uses a webcam + DA2 monocular depth; "
                             "'realsense' uses an Intel RealSense (e.g. D415) "
                             "for colour and real metric depth")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="OpenCV camera index for the video source "
                             "(e.g. 1 for an external/USB webcam); "
                             "ignored for --camera-backend realsense")
    parser.add_argument("--score-thr",   type=float, default=0.08)
    parser.add_argument("--depth-model",
                        default=str(PROJECT_ROOT/"checkpoints"/"depth_anything_v2"),
                        help="ignored for --camera-backend realsense")
    parser.add_argument("--da2-encoder", default="vitl",
                        choices=["vits","vitb","vitl"],
                        help="ignored for --camera-backend realsense")
    parser.add_argument("--infer-scale", type=float, default=0.55)
    parser.add_argument("--report-dir",
                        default=str(PROJECT_ROOT),
                        help="directory to save session_report_*.png")
    parser.add_argument("--rs-width",  type=int, default=1280,
                        help="RealSense colour/depth stream width")
    parser.add_argument("--rs-height", type=int, default=720,
                        help="RealSense colour/depth stream height")
    parser.add_argument("--rs-fps",    type=int, default=30,
                        help="RealSense colour/depth stream frame rate")
    parser.add_argument("--enable-ik", action="store_true",
                        help="retarget the tracked right hand to SO-101 right-arm "
                             "joint angles via the vbhs PyBullet IK solver; "
                             "requires --camera-backend realsense")
    parser.add_argument("--enable-robot", action="store_true",
                        help="drive a physical LeRobot SO-101 follower arm from "
                             "the IK solution; requires --enable-ik and --robot-port. "
                             "Starts in standby; press 'g' in the window to arm it "
                             "and 's' to return to standby")
    parser.add_argument("--robot-port", default="",
                        help="serial port of the SO-101 follower arm "
                             "(e.g. COM3 or /dev/ttyACM0)")
    parser.add_argument("--robot-id", default="right_follower",
                        help="LeRobot robot id / calibration file name for the "
                             "SO-101 follower arm")
    args = parser.parse_args()

    pipeline = RobotLearningHandPipeline(
        device         = args.device,
        score_thr      = args.score_thr,
        depth_model    = args.depth_model,
        da2_encoder    = args.da2_encoder,
        infer_scale    = args.infer_scale,
        report_dir     = args.report_dir,
        camera_index   = args.camera_index,
        camera_backend = args.camera_backend,
        rs_width       = args.rs_width,
        rs_height      = args.rs_height,
        rs_fps         = args.rs_fps,
        enable_ik      = args.enable_ik,
        enable_robot   = args.enable_robot,
        robot_port     = args.robot_port,
        robot_id       = args.robot_id,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
