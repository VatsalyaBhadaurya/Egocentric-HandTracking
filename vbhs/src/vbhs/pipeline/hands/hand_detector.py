"""Abstract base class for hand detection."""
from typing import Optional
import abc

import cv2

from vbhs.pipeline import types


class HandDetector(abc.ABC):
    """Abstract base class for hand detector."""

    def __init__(self, camera_intrinsics: types.CameraIntrinsics):
        """Initialize the hand detector."""
        self._camera_intrinsics = camera_intrinsics

    @abc.abstractmethod
    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D],
                       Optional[types.HandPose3D], Optional[types.HandPose3D]]:
        """Detect hand landmarks from an RGB image.

        Returns:
            (left_2d, right_2d, left_world, right_world) where world landmarks
            are metric 3D in a hand-centric frame (MediaPipe hand_world_landmarks)
            or None when the detector does not produce them.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cleanup(self):
        """Cleanup the hand detector."""
        raise NotImplementedError()
