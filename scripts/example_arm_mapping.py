import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.action_encoder import HandActionEncoder, build_hand_state
from src.robot_mapper import SimpleArmRetargeter


def main():
    encoder = HandActionEncoder()
    mapper = SimpleArmRetargeter(image_size=(960, 720))

    scores = np.ones(21, dtype=np.float32) * 0.95

    demo_frames = [
        {
            "label": "reach forward with open palm",
            "gesture": "OPEN PALM",
            "xy": np.array([430.0, 290.0], dtype=np.float32),
            "depth": 0.52,
            "normal": np.array([0.05, -0.12, 0.96], dtype=np.float32),
            "t": 0.00,
        },
        {
            "label": "move right and pinch to grasp",
            "gesture": "PINCH",
            "xy": np.array([590.0, 285.0], dtype=np.float32),
            "depth": 0.41,
            "normal": np.array([0.18, -0.08, 0.90], dtype=np.float32),
            "t": 0.12,
        },
        {
            "label": "hold object steady",
            "gesture": "GRASP",
            "xy": np.array([600.0, 290.0], dtype=np.float32),
            "depth": 0.39,
            "normal": np.array([0.20, -0.05, 0.88], dtype=np.float32),
            "t": 0.24,
        },
    ]

    for frame in demo_frames:
        kp = np.zeros((21, 3), dtype=np.float32)
        kp[:, 0] = frame["xy"][0]
        kp[:, 1] = frame["xy"][1]

        hand_state = build_hand_state(
            slot=0,
            side="right",
            keypoints=kp,
            scores=scores,
            gesture=frame["gesture"],
            meta={
                "confidence": 0.95,
                "pinch_norm": 0.22 if frame["gesture"] == "PINCH" else 0.78,
                "mean_ratio": 1.7 if frame["gesture"] == "OPEN PALM" else 1.0,
                "palm_normal": frame["normal"],
            },
            timestamp=frame["t"],
            palm_depth=frame["depth"],
        )

        actions = encoder.encode([hand_state], frame["t"])
        command = mapper.map(hand_state, actions[0])

        print(f"\nFrame: {frame['label']}")
        print(f"  action : {actions[0].label}")
        print(f"  gripper: {command.gripper:.2f}")
        print(f"  joints : {np.array2string(command.joint_targets, precision=3)}")


if __name__ == "__main__":
    main()
