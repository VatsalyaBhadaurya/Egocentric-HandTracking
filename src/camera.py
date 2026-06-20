import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics, in pixels."""
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

class ThreadedCamera:
    """Webcam wrapper that grabs frames on a background thread.

    Single-owner design: the grab thread is the ONLY thread that ever touches the
    VideoCapture — it both reads frames and releases the device when it stops.
    `read()` just hands back the latest frame under a lock, and `release()` only
    signals the thread to stop and joins it. This avoids calling cap.release()
    while cap.read() is still running on the thread, which can deadlock or crash
    the V4L2 backend (the bug that used to hang shutdown).
    """

    def __init__(self, src=0, width=848, height=480):
        self._src    = src
        self._width  = width
        self._height = height

        self._frame: np.ndarray | None = None
        self._lock    = threading.Lock()
        self._running = True
        self._opened  = threading.Event()   # set once the device is open (ok or not)
        self.opened   = False               # readable result of the open attempt

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        # Let the caller see the open result without racing the thread.
        self._opened.wait(timeout=5.0)

    def _grab_loop(self):
        # Owns the VideoCapture for its whole lifetime: open → grab latest frame
        # in a loop → release on exit. Nothing else touches `cap`.
        cap = cv2.VideoCapture(self._src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self.opened = cap.isOpened()
        self._opened.set()
        print(f"[camera debug] VideoCapture opened on src={self._src}: {self.opened}")
        try:
            while self._running:
                t_start = time.time()
                ok, frame = cap.read()
                t_read = time.time()
                if not ok:
                    # Dead/half-open device: don't busy-spin the CPU.
                    time.sleep(0.005)
                    continue
                with self._lock:
                    self._frame = frame
                if not hasattr(self, '_debug_count'): self._debug_count = 0
                self._debug_count += 1
                if self._debug_count % 30 == 0:
                    print(f"[camera debug] Frame fetch: {(t_read - t_start)*1000:.1f}ms | Shape: {frame.shape}")
        finally:
            cap.release()

    def read(self):
        # Returns the most recent frame grabbed by the background thread, or None
        # if no frame has arrived yet.
        with self._lock:
            return self._frame

    def release(self):
        # Signal the grab thread to stop and wait for it to release the device.
        # Only the grab thread touches `cap`, so there is no concurrent-access
        # race here. Bounded join so a wedged read() can't hang shutdown forever
        # (the thread is a daemon and dies with the process if it overruns).
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class RealSenseCamera:
    """Intel RealSense (D4xx) wrapper returning a colour frame aligned with metric depth.

    `read()` mirrors ThreadedCamera's interface (returns the BGR colour frame, or
    None on failure). The depth frame captured alongside it — converted to metres
    and aligned to the colour frame — is cached and retrieved via `read_depth()`.
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs
        self._rs = rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(config)

            # ── connection check ──────────────────────────────────────────
            device = profile.get_device()
            print(f"[camera] RealSenseCamera connected  "
                f"device={device.get_info(rs.camera_info.name)}  "
                f"serial={device.get_info(rs.camera_info.serial_number)}  "
                f"resolution={width}x{height}  fps={fps}")
            # ─────────────────────────────────────────────────────────────

            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align       = rs.align(rs.stream.color)

            color_intr = (profile.get_stream(rs.stream.color)
                                .as_video_stream_profile()
                                .get_intrinsics())
            self.intrinsics = Intrinsics(
                width=color_intr.width, height=color_intr.height,
                fx=color_intr.fx,       fy=color_intr.fy,
                ppx=color_intr.ppx,     ppy=color_intr.ppy)

            self._last_depth: np.ndarray | None = None

        except Exception as e:
            print(f"[camera] RealSenseCamera FAILED to connect: {e}")
            raise

    def read(self):
        # Reads one aligned colour+depth frame pair and returns the BGR colour
        # frame, caching the depth map (metres) for read_depth(). Returns None
        # on failure/timeout.
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return None
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        self._last_depth = (np.asanyarray(depth_frame.get_data())
                             .astype(np.float32) * self.depth_scale)
        return np.asanyarray(color_frame.get_data())

    def read_depth(self):
        # Returns the depth map (metres) aligned to the most recently read colour
        # frame, or None if no frame has been read yet.
        return self._last_depth

    def release(self):
        # Stops the RealSense pipeline, suppressing any errors.
        try:
            self.pipeline.stop()
        except Exception:
            pass
class OrbbecCamera:
    """Orbbec wrapper returning a colour frame aligned with metric depth.
    
    Mirrors the RealSenseCamera interface.
    """

    def __init__(self, **_):
        try:
            from pyorbbecsdk import Pipeline, Config, OBSensorType
        except ImportError:
            raise ImportError("pyorbbecsdk not found. Install it via: pip install pyorbbecsdk2")

        self.pipeline = Pipeline()
        config = Config()

        # Color stream — use SDK default (avoids resolution negotiation issues)
        color_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        config.enable_stream(color_profile)
        color_w, color_h = color_profile.get_width(), color_profile.get_height()

        # Depth stream — SDK default; non-default profiles corrupt the ToF raw-phase pipeline
        depth_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        try:
            self.pipeline.start(config)

            # Extract intrinsics from color camera
            camera_param = self.pipeline.get_camera_param()
            intr = camera_param.rgb_intrinsic
            self.intrinsics = Intrinsics(
                width=color_w, height=color_h,
                fx=intr.fx, fy=intr.fy,
                ppx=intr.cx, ppy=intr.cy)

            self._last_depth: np.ndarray | None = None
            self._error_count = 0
            print(f"[camera] OrbbecCamera connected  resolution={color_w}x{color_h}")

        except Exception as e:
            print(f"[camera] OrbbecCamera FAILED to connect: {e}")
            raise

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(200)
            if frames is None:
                return None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None:
                return None

            # Decode color — MJPEG (variable-size) or raw RGB
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            if color_bgr is None:
                cw = color_frame.get_width()
                ch = color_frame.get_height()
                color_bgr = cv2.cvtColor(color_data.reshape((ch, cw, 3)), cv2.COLOR_RGB2BGR)

            # Decode depth and resize to match color frame (no AlignFilter needed)
            if depth_frame is not None:
                dw = depth_frame.get_width()
                dh = depth_frame.get_height()
                depth_scale = depth_frame.get_depth_scale()
                raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))
                depth_m = raw.astype(np.float32) * depth_scale / 1000.0
                ch, cw = color_bgr.shape[:2]
                if dh != ch or dw != cw:
                    depth_m = cv2.resize(depth_m, (cw, ch), interpolation=cv2.INTER_NEAREST)
                self._last_depth = depth_m
            else:
                self._last_depth = None

            return color_bgr

        except Exception as e:
            count = getattr(self, '_err_count', 0) + 1
            self._err_count = count
            if count <= 5 or count % 100 == 0:
                print(f"[camera] OrbbecCamera read error #{count}: {e}")
            return None

    def read_depth(self):
        return self._last_depth

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass