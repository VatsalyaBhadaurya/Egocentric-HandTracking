# Author: adith
# OpenCV VideoCapture wrapper that always returns the latest frame

import cv2
import numpy as np


class ThreadedCamera:

    def __init__(self, src=0, width=960, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    def read(self):
        # Reads one frame from the capture device, returning None on failure.
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        # Releases the underlying VideoCapture handle, suppressing any errors.
        try:
            self.cap.release()
        except Exception:
            pass


class RealSenseCamera:
    """Intel RealSense (D4xx) wrapper returning a colour frame aligned with metric depth.

    `read()` mirrors ThreadedCamera's interface (returns the BGR colour frame, or
    None on failure). The depth frame captured alongside it — converted to metres
    and aligned to the colour frame — is cached and retrieved via `read_depth()`.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        import pyrealsense2 as rs
        self._rs = rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        self._last_depth: np.ndarray | None = None

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
