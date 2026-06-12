# Author: adith
# Depth Anything V2 inference wrapper that converts raw output to absolute metres

from pathlib import Path
import sys
import cv2
import numpy as np
import torch

TARGET_M   = 0.40
DA2_H = DA2_W = 518


def _load_da2(model_path: Path, device: str, encoder: str):
    # Imports DepthAnythingV2 from a local repo clone and loads the specified checkpoints
    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    if encoder not in model_configs:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: vits, vitb, vitl")

    ckpt      = model_path / f"depth_anything_v2_{encoder}.pth"
    repo_path = model_path / "Depth-Anything-V2"

    if not repo_path.exists():
        raise RuntimeError(f"DA2 repo not found at {repo_path}")

    if not ckpt.exists():
        raise RuntimeError(f"DA2 checkpoint not found: {ckpt}")

    repo_str = str(repo_path)
    inserted = False
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        inserted = True
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg   = model_configs[encoder]
        model = DepthAnythingV2(**cfg)
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    except ImportError as e:
        raise RuntimeError(f"DA2 import failed from {repo_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"DA2 load failed: {e}") from e
    finally:
        if inserted and repo_str in sys.path:
            sys.path.remove(repo_str)


class DepthFusion:
    # Converts raw DA2 output to absolute metres using either relative or metric mode.
    # In relative mode raw values are treated as disparity, while metric mode outputs are in metres directly.

    def __init__(self, min_depth: float = 0.1, max_depth: float = 3.0,
                 is_relative: bool = False):
        self.min_depth   = min_depth
        self.max_depth   = max_depth
        self.is_relative = is_relative
        self.scale_trim  = None
        self._calib_samples: list[float] = []

    def fuse(self, raw: np.ndarray) -> np.ndarray:
        # Converts raw DA2 output to depth in metres and clamps to [min_depth, max_depth].
        if self.is_relative:
            depth = 1.0 / np.clip(raw, 1e-6, None)
        else:
            depth = raw.copy()
        if self.scale_trim is not None:
            depth = depth * self.scale_trim
        return np.clip(depth, self.min_depth, self.max_depth).astype(np.float32)

    def begin_calibration(self):
        # Clears any in-progress calibration samples.
        self._calib_samples = []

    def add_calibration_sample(self, raw_da2: np.ndarray, palm_kp2d: np.ndarray) -> bool:
        # Samples DA2 output at the palm centroid and queues it for calibration.
        # Returns False if the palm patch is empty or the raw value is near zero.
        H, W  = raw_da2.shape
        cx    = int(np.clip(palm_kp2d[:, 0].mean(), 0, W - 1))
        cy    = int(np.clip(palm_kp2d[:, 1].mean(), 0, H - 1))
        r     = 25
        patch = raw_da2[max(0, cy-r):min(H, cy+r),
                        max(0, cx-r):min(W, cx+r)]
        if patch.size == 0:
            return False

        raw_val = float(np.median(patch))
        if raw_val < 1e-6:
            return False

        self._calib_samples.append(raw_val)
        return True

    def finalize_calibration(self) -> bool:
        # Computes a scale from the median of the queued samples so the fused depth
        # reads TARGET_M at the palm. Using the median over the whole calibration
        # window is far more robust to single-frame noise/outliers than a one-shot sample.
        # Returns False if no samples were collected.
        if not self._calib_samples:
            return False

        raw_val = float(np.median(self._calib_samples))
        self._calib_samples = []
        if raw_val < 1e-6:
            return False

        if self.is_relative:
            depth_at_palm = 1.0 / raw_val
        else:
            depth_at_palm = raw_val

        self.scale_trim = TARGET_M / depth_at_palm
        return True

    def reset(self):
        # Clears the calibration scale so the estimator returns uncalibrated depths.
        self.scale_trim = None
        self._calib_samples = []


class BaseDepthEstimator:
    """Shared depth-map utilities: fusion/calibration, palm sampling, EMA smoothing, colorizing.

    Subclasses provide `estimate()`, which produces a raw depth map (in either
    relative-disparity or metric units, per `is_relative`) and fuses+smooths it
    into the shared (smoothed_depth_metres, raw_output) return shape.
    """

    def __init__(self, min_depth: float = 0.1, max_depth: float = 3.0,
                 is_relative: bool = False):
        self.fusion = DepthFusion(min_depth=min_depth, max_depth=max_depth,
                                  is_relative=is_relative)
        self._prev = None

    @property
    def calib_scale(self):
        return self.fusion.scale_trim

    @calib_scale.setter
    def calib_scale(self, v):
        self.fusion.scale_trim = v

    @property
    def enabled(self):
        return True

    def begin_calibration(self):
        # Start a new calibration window.
        self.fusion.begin_calibration()

    def add_calibration_sample(self, raw: np.ndarray, palm_kp2d: np.ndarray) -> bool:
        # Queue one frame's palm-anchored depth sample for calibration.
        return self.fusion.add_calibration_sample(raw, palm_kp2d)

    def finalize_calibration(self) -> bool:
        # Compute the palm-anchored scale trim from all queued samples.
        return self.fusion.finalize_calibration()

    def _smooth(self, metres: np.ndarray) -> np.ndarray:
        # Applies an 0.8/0.2 EMA over consecutive frames and returns the smoothed map.
        if self._prev is None:
            self._prev = metres.copy()
        else:
            self._prev = (0.8 * self._prev + 0.2 * metres).astype(np.float32)
        return self._prev.copy()

    def sample(self, depth_map: np.ndarray, xy, patch: int = 10) -> float:
        # Returns the median depth within a patch-sized window centred on xy.
        H, W = depth_map.shape
        x = int(np.clip(round(float(xy[0])), 0, W - 1))
        y = int(np.clip(round(float(xy[1])), 0, H - 1))
        return float(np.median(
            depth_map[max(0, y-patch):min(H, y+patch+1),
                      max(0, x-patch):min(W, x+patch+1)]))

    def depth_at_hand(self, depth_map: np.ndarray,
                      kp2d: np.ndarray, sc: np.ndarray,
                      joint_thr: float = 0.12):
        # Computes the median of per-joint depth samples at the wrist and four MCP
        # joints. Sampling each joint individually (rather than one patch at the
        # centroid) avoids depth-edge bleed when the palm centroid falls near a
        # foreground/background discontinuity (e.g. between splayed fingers).
        # Returns None if no palm joints exceed the score threshold.
        palm_ids = [20, 7, 11, 15, 19]
        valid    = [j for j in palm_ids if sc[j] > joint_thr]
        if not valid:
            return None
        samples = [self.sample(depth_map, kp2d[j], patch=6) for j in valid]
        return float(np.median(samples))

    @staticmethod
    def colorize(depth: np.ndarray) -> np.ndarray:
        # Normalises a depth map to [0, 255] and applies the TURBO colour map.
        d = depth - depth.min()
        d = (d / max(depth.max() - depth.min(), 1e-6) * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)


class DepthEstimator(BaseDepthEstimator):
    """Monocular depth via Depth Anything V2, scaled to metres through palm calibration."""

    def __init__(self, model_path: str, device: str = "cpu",
                 encoder: str = "vitl",
                 min_depth: float = 0.1, max_depth: float = 3.0):
        ckpt_name   = f"depth_anything_v2_{encoder}"
        is_relative = not any(k in ckpt_name for k in ("metric", "indoor", "outdoor"))
        super().__init__(min_depth=min_depth, max_depth=max_depth, is_relative=is_relative)

        self.device  = device
        self.path    = Path(model_path)
        self.encoder = encoder
        self.model   = _load_da2(self.path, self.device, encoder)

    def _da2_infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Normalises the frame to ImageNet stats, resizes to 518×518, runs the model, and resizes output back.
        # Returns the raw single-channel depth map at original resolution.
        H, W = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp  = cv2.resize(rgb, (DA2_W, DA2_H)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        inp  = (inp - mean) / std
        x    = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            raw = out["metric_depth"] if "metric_depth" in out else next(iter(out.values()))
        else:
            raw = out
        raw = raw.squeeze().cpu().numpy().astype(np.float32)
        return cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)

    def estimate(self, frame_bgr: np.ndarray):
        # Runs DA2 inference, fuses to metres via the (possibly calibrated) scale,
        # and applies EMA smoothing. Returns (smoothed_depth_metres, raw_da2_output).
        raw_output = self._da2_infer(frame_bgr)
        metres     = self.fusion.fuse(raw_output)
        return self._smooth(metres), raw_output


class RealSenseDepthEstimator(BaseDepthEstimator):
    """Wraps an Intel RealSense depth stream, which is already in metres.

    No monocular-to-metric calibration is needed by default — the sensor reports
    real-world distance directly. `begin_calibration`/`add_calibration_sample`/
    `finalize_calibration` remain available (inherited) for an optional fine-trim
    if the sensor reads consistently off.
    """

    def __init__(self, min_depth: float = 0.1, max_depth: float = 3.0):
        super().__init__(min_depth=min_depth, max_depth=max_depth, is_relative=False)

    def estimate(self, raw_depth_m: np.ndarray):
        # Fuses the sensor's metric depth (treating invalid 0-valued pixels as
        # out-of-range rather than "0 m away") and applies EMA smoothing.
        # Returns (smoothed_depth_metres, raw_depth_m).
        raw = raw_depth_m.copy()
        raw[raw <= 1e-6] = self.fusion.max_depth
        metres = self.fusion.fuse(raw)
        return self._smooth(metres), raw_depth_m
