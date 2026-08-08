"""MediaPipe-based hand detector implementation."""
from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp  # type: ignore

from vbhs.config import config
from vbhs.pipeline import types
from vbhs.pipeline.hands import hand_detector


class MediaPipeHandDetector(hand_detector.HandDetector):
    """MediaPipe-based hand detector."""

    def __init__(self,
                camera_intrinsics: types.CameraIntrinsics,
                 detection_confidence: float=0.4,
                 tracking_confidence: float=0.4,
                 max_num_hands: int=2,
                 model_complexity: int=1,
                 handedness_confidence: float=0.0):
        """Initialize MediaPipe hand detection.

        Args:
            camera_intrinsics: Camera intrinsics for hand detection.
            detection_confidence: Minimum confidence for hand *detection* (0.0-1.0).
                Kept moderate (0.4) so the palm-detection model still acquires the
                hand when the back of the hand / arm faces the camera, which is a
                weaker signal than a palm-facing view in egocentric footage.
            tracking_confidence: Minimum confidence for hand tracking (0.0-1.0).
            max_num_hands: Maximum number of hands to detect (1 or 2)
            model_complexity: Model complexity (0=lite, 1=full). Full is more
                robust on dorsal (back-of-hand) views.
            handedness_confidence: Minimum left-vs-right *classification* score
                required to keep a detection. Defaults to 0.0 on purpose: a hand
                seen from the back is still a real hand even when MediaPipe can't
                tell left from right (the score collapses toward 0.5), so we must
                not discard it here. Slot assignment is disambiguated temporally
                in ``detect`` instead. Raise this only if you want to *reject*
                ambiguous hands.
        """
        super().__init__(camera_intrinsics)
        self._detection_confidence = detection_confidence
        self._tracking_confidence = tracking_confidence
        self._max_num_hands = max_num_hands
        self._model_complexity = model_complexity
        self._handedness_confidence = handedness_confidence
        # Score above which we trust MediaPipe's left/right label outright; below
        # it the label is treated as unreliable (typical for back-of-hand views)
        # and we fall back to temporal continuity.
        self._reliable_handedness = 0.6
        # Last-known image centroid per FPV slot ('left'/'right'), used to keep an
        # ambiguous hand on the same arm across frames instead of flipping or
        # dropping it when the back of the hand faces the camera.
        self._prev_centroids: dict[str, tuple[float, float]] = {}

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self._max_num_hands,
            model_complexity=self._model_complexity,
            min_detection_confidence=self._detection_confidence,
            min_tracking_confidence=self._tracking_confidence
        )

    def detect(
            self, rgb_image: cv2.typing.MatLike
            ) -> tuple[Optional[types.HandPose2D], Optional[types.HandPose2D],
                       Optional[types.HandPose3D], Optional[types.HandPose3D]]:
        """Detect hand landmarks from an RGB image.

        Returns ``(left_2d, right_2d, left_world, right_world)`` in FPV convention.
        A hand is returned whenever MediaPipe locates it, regardless of whether the
        palm or the back of the hand faces the camera. When the dorsal view makes
        the left/right label unreliable, the detection is assigned to the arm it
        occupied on the previous frame so tracking stays seamless instead of
        dropping out.

        World landmarks are from MediaPipe's ``hand_world_landmarks``: metric 3D
        in a hand-centric frame (origin at the geometric centre of the hand).
        These are more stable for inter-joint distances and relative orientation
        than the depth-deprojected camera-space positions, especially at fingertips.
        """
        image_height, image_width = rgb_image.shape[:2]

        results = self._hands.process(rgb_image)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self._prev_centroids = {}
            return None, None, None, None

        world_lms_list = results.multi_hand_world_landmarks or []

        # ── 1. Collect every detected hand without dropping ambiguous ones ──────
        detections = []
        for i, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            confidence = handedness.classification[0].score

            if confidence < self._handedness_confidence:
                continue

            if not len(hand_landmarks.landmark) == len(config.MEDIAPIPE_HAND_LANDMARKS):
                raise ValueError(
                    f'Expected {len(config.MEDIAPIPE_HAND_LANDMARKS)} '
                    f'landmarks, got {len(hand_landmarks.landmark)}')

            landmarks_uv: types.HandPose2D = {}
            sum_u = sum_v = 0.0
            for key, index in config.MEDIAPIPE_HAND_LANDMARKS.items():
                lm = hand_landmarks.landmark[index]
                u = lm.x * image_width
                v = lm.y * image_height
                landmarks_uv[key] = (u, v)
                sum_u += u
                sum_v += v
            n = len(landmarks_uv)
            centroid = (sum_u / n, sum_v / n)

            # Extract metric world landmarks when available (same index as 2D list).
            world_landmarks_xyz: Optional[types.HandPose3D] = None
            if i < len(world_lms_list):
                wlms = world_lms_list[i]
                world_landmarks_xyz = {
                    key: (wlms.landmark[idx].x,
                          wlms.landmark[idx].y,
                          wlms.landmark[idx].z)
                    for key, idx in config.MEDIAPIPE_HAND_LANDMARKS.items()
                }

            detections.append({
                'label': hand_label,
                'score': confidence,
                'landmarks': landmarks_uv,
                'world_landmarks': world_landmarks_xyz,
                'centroid': centroid,
            })

        if not detections:
            self._prev_centroids = {}
            return None, None, None, None

        # ── 2. Assign detections to FPV left/right slots ────────────────────────
        slots: dict[str, types.HandPose2D] = {}
        world_slots: dict[str, Optional[types.HandPose3D]] = {}

        def fpv_side(label: str) -> str:
            # MediaPipe labels are from the imaged person's perspective, mirrored
            # relative to the egocentric (FPV) view, so we swap them here.
            return 'right' if label == 'Left' else 'left'

        ambiguous = (len(detections) == 1
                     and detections[0]['score'] < self._reliable_handedness)
        if ambiguous and self._prev_centroids:
            d = detections[0]
            side = min(
                self._prev_centroids,
                key=lambda s: _sq_dist(d['centroid'], self._prev_centroids[s]))
            slots[side] = d['landmarks']
            world_slots[side] = d['world_landmarks']
        else:
            for d in detections:
                side = fpv_side(d['label'])
                if side in slots:
                    side = 'left' if 'left' not in slots else 'right'
                    if side in slots:
                        continue
                slots[side] = d['landmarks']
                world_slots[side] = d['world_landmarks']

        # ── 3. Refresh slot-history centroids for next-frame continuity ─────────
        self._prev_centroids = {
            side: _centroid_of(lm) for side, lm in slots.items()
        }

        return (
            slots.get('left'),
            slots.get('right'),
            world_slots.get('left'),
            world_slots.get('right'),
        )

    def cleanup(self):
        """Cleanup MediaPipe resources."""
        self._hands.close()


def _sq_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared Euclidean distance between two image points."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _centroid_of(landmarks: types.HandPose2D) -> tuple[float, float]:
    """Mean (u, v) of a hand's landmarks."""
    n = len(landmarks)
    sum_u = sum(uv[0] for uv in landmarks.values())
    sum_v = sum(uv[1] for uv in landmarks.values())
    return (sum_u / n, sum_v / n)
