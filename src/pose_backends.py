# Author: adith
# MMPose InterNet backend for 3D hand keypoint inference with NMS deduplication

from __future__ import annotations

from pathlib import Path
import os
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown


def _find_mmpose_root() -> Path:
    # Locates the mmpose repo root via MMPOSE_ROOT env var or by walking up from this file.
    # Raises RuntimeError if neither the env var nor a configs/hand_3d_keypoint directory is found.
    env = os.environ.get("MMPOSE_ROOT")
    if env:
        root = Path(env).resolve()
        if not (root / "configs").exists():
            raise RuntimeError(
                f"MMPOSE_ROOT={root} has no configs/ directory. "
                "Point MMPOSE_ROOT at your mmpose repo root.")
        return root

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "hand_3d_keypoint").exists():
            return parent

    raise RuntimeError(
        "Could not find mmpose root. Set MMPOSE_ROOT=/path/to/mmpose "
        "and rerun.")


class _HandPresenceGate:
    """Cheap MediaPipe Hands (lite) pre-filter used to reject MMPose detections
    that aren't actually a hand (faces, random objects), which InterNet can
    otherwise hallucinate with moderate confidence since it has no built-in
    hand/no-hand detector of its own."""

    def __init__(self, detection_confidence: float = 0.4, pad: float = 0.35):
        self._pad = pad
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=detection_confidence)
        except Exception as e:
            print(f"[pose] MediaPipe hand-presence gate disabled: {e}")
            self._hands = None

    @property
    def enabled(self) -> bool:
        return self._hands is not None

    def hand_boxes(self, frame_bgr: np.ndarray) -> list:
        # Returns padded (x0,y0,x1,y1) boxes for each hand MediaPipe finds in
        # the frame, in frame_bgr pixel coordinates. Empty list = no hands.
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        boxes = []
        if not results.multi_hand_landmarks:
            return boxes
        for lm in results.multi_hand_landmarks:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            bw, bh = x1 - x0, y1 - y0
            boxes.append((x0 - bw*self._pad, y0 - bh*self._pad,
                           x1 + bw*self._pad, y1 + bh*self._pad))
        return boxes

    def close(self):
        if self._hands is not None:
            self._hands.close()


def _center_in_box(center, box) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= center[0] <= x1 and y0 <= center[1] <= y1


def _filter_by_gate(hands: list, boxes: list) -> list:
    # Keeps only detections whose keypoint centroid falls inside one of the
    # gate's hand boxes. An empty `boxes` list (gate saw no hands) drops
    # everything for this frame.
    if not boxes:
        return []
    out = []
    for h in hands:
        center = h["keypoints"][:, :2].mean(0)
        if any(_center_in_box(center, b) for b in boxes):
            out.append(h)
    return out


class MMPoseHandBackend:
    DEPTH_W  = 0.4
    MAX_MISS = 12

    def __init__(self, device: str = "cpu", score_thr: float = 0.12,
                 infer_scale: float = 0.6):
        repo   = _find_mmpose_root()
        config = str(repo / "configs/hand_3d_keypoint/internet/interhand3d"
                           "/internet_res50_4xb16-20e_interhand3d-256x256.py")
        ckpt   = str(repo / "checkpoints/res50.pth")

        _orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo))
            self.model = init_model(config, ckpt, device=device)
        finally:
            os.chdir(_orig_cwd)

        self.score_thr   = score_thr
        self.infer_scale = infer_scale
        self._gate       = _HandPresenceGate()

        self._centers = [None, None]
        self._sizes   = [None, None]
        self._depths  = [None, None]
        self._missing = [0, 0]

    def infer(self, frame_bgr: np.ndarray, depth_info: dict | None = None):
        # Runs the full inference pipeline: model forward pass, NMS dedup, depth attachment, and slot matching.
        # Returns a list of hand dicts (up to two) with keypoints, scores, side, and palm depth.
        raw = self._run(frame_bgr)
        raw = _dedup(raw)

        if self._gate.enabled:
            raw = _filter_by_gate(raw, self._gate.hand_boxes(frame_bgr))

        if depth_info is not None:
            dm  = depth_info["depth_map"]
            est = depth_info["depth_est"]
            for h in raw:
                h["palm_depth"] = est.depth_at_hand(
                    dm, h["keypoints"][:, :2], h["scores"])
        else:
            for h in raw:
                h["palm_depth"] = None

        out = self._match(raw)

        for slot, hand in enumerate(out):
            if hand is not None:
                self._missing[slot] = 0
                self._ema(slot, hand)
            else:
                self._missing[slot] += 1
                if self._missing[slot] >= self.MAX_MISS:
                    self._centers[slot] = self._sizes[slot] = None
                    self._depths[slot]  = None

        return [h for h in out if h is not None]

    def close(self):
        self._gate.close()

    def _run(self, frame_bgr: np.ndarray):
        # Scales the frame by infer_scale, runs MMPose inference, and parses results into hand dicts.
        scaled  = cv2.resize(frame_bgr, None,
                             fx=self.infer_scale, fy=self.infer_scale)
        results = inference_topdown(self.model, scaled)
        return self._extract(results)

    def _extract(self, results):
        # Parses MMPose PredInstance results into a uniform list of hand dicts.
        # Handles both 42-keypoint (two-hand) and 21-keypoint (one-hand) model outputs.
        hands  = []
        sc_min = self.score_thr * 0.5
        for res in results:
            pred = getattr(res, "pred_instances", None)
            if pred is None:
                continue
            kpa = np.asarray(getattr(pred, "keypoints",       []))
            sca = np.asarray(getattr(pred, "keypoint_scores", []))
            if kpa.ndim < 3 or kpa.shape[0] == 0:
                continue
            kp = kpa[0].astype(np.float32)
            sc = sca[0].astype(np.float32)
            if sc.max() > 1.0:
                sc /= 255.0
            kp[:, 0] /= self.infer_scale
            kp[:, 1] /= self.infer_scale
            n = kp.shape[0]
            if n == 42:
                for start, side in [(0, "right"), (21, "left")]:
                    hkp = kp[start:start+21].copy()
                    hsc = sc[start:start+21].copy()
                    if np.mean(hsc) >= sc_min:
                        hands.append(_mkhand(hkp, hsc, side))
            elif n == 21 and np.mean(sc) >= sc_min:
                hands.append(_mkhand(kp, sc, "unknown"))
        return hands

    def _match(self, raw: list) -> list:
        # Assigns detections to slot 0 (right) and slot 1 (left) using hand_side labels then score-based fallback.
        out, used = [None, None], set()
        for j, h in enumerate(raw):
            slot = {"right": 0, "left": 1}.get(h["hand_side"])
            if slot is not None and out[slot] is None:
                out[slot] = h
                used.add(j)
        for slot in range(2):
            if out[slot] is not None:
                continue
            best_s, best_j = -1e9, None
            for j, h in enumerate(raw):
                if j in used:
                    continue
                s = self._slot_score(h, slot)
                if s > best_s:
                    best_s, best_j = s, j
            if (best_j is not None and
                    raw[best_j]["mean_score"] >= self.score_thr):
                out[slot] = raw[best_j]
                used.add(best_j)
        return out

    def _slot_score(self, hand: dict, slot: int) -> float:
        # Scores a detection for a slot using confidence penalised by spatial distance and depth discontinuity.
        conf = hand["mean_score"]
        if self._centers[slot] is None:
            return conf
        sz   = self._sizes[slot] or 150.0
        dist = np.linalg.norm(
            hand["keypoints"][:, :2].mean(0) - self._centers[slot])
        score = conf - 0.5 * (dist / sz)
        if (self._depths[slot] is not None and
                hand.get("palm_depth") is not None):
            diff  = abs(hand["palm_depth"] - self._depths[slot])
            score -= self.DEPTH_W * max(0.0, diff - 0.15)
        return score

    def _ema(self, slot: int, hand: dict, alpha: float = 0.35):
        # Updates the per-slot running EMA estimates of hand centre, bounding-box size, and palm depth.
        kp = hand["keypoints"][:, :2]
        nc = kp.mean(0)
        d  = kp.max(0) - kp.min(0)
        ns = float(max(d[0], d[1], 100.0))
        if self._centers[slot] is None:
            self._centers[slot] = nc
            self._sizes[slot]   = ns
        else:
            self._centers[slot] = alpha*nc + (1-alpha)*self._centers[slot]
            self._sizes[slot]   = alpha*ns + (1-alpha)*self._sizes[slot]
        pd = hand.get("palm_depth")
        if pd is not None:
            if self._depths[slot] is None:
                self._depths[slot] = pd
            else:
                self._depths[slot] = alpha*pd + (1-alpha)*self._depths[slot]


def _mkhand(kp: np.ndarray, sc: np.ndarray, side: str) -> dict:
    # Constructs a standardised hand dict with 21×3 keypoints, scores, mean score, side label, and null depth.
    if kp.shape[1] == 2:
        kp = np.concatenate([kp, np.zeros((21, 1), np.float32)], axis=1)
    return {"keypoints": kp[:, :3].copy(), "scores": sc.copy(),
            "mean_score": float(np.mean(sc)), "hand_side": side,
            "palm_depth": None}


def _iou(a, b) -> float:
    # Computes the intersection-over-union of two axis-aligned bounding boxes.
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / ((ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter + 1e-9)


def _dedup(hands: list, thr: float = 0.5) -> list:
    # Applies NMS over hand bounding boxes, keeping the higher-scoring detection when two overlap.
    if len(hands) <= 1:
        return hands
    boxes = [(*h["keypoints"][:, :2].min(0), *h["keypoints"][:, :2].max(0))
             for h in hands]
    keep, sup = [], set()
    for i in range(len(hands)):
        if i in sup:
            continue
        keep.append(i)
        for j in range(i+1, len(hands)):
            if j in sup:
                continue
            if _iou(boxes[i], boxes[j]) > thr:
                if hands[j]["mean_score"] > hands[keep[-1]]["mean_score"]:
                    keep[-1] = j
                sup.add(j)
    return [hands[k] for k in keep]
