"""
Loads the YOLO model once and reports whether a person is visible in a
frame. This module only looks at a single frame at a time - it has no idea
about time or state, that's presence.py's job.
"""

from dataclasses import dataclass

from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO dataset class id for "person"


@dataclass
class Detection:
    present: bool          # True if at least one person was detected at all
    confidence: float      # confidence of the chosen box (0.0 if none found)
    box_area_ratio: float  # chosen box's area / frame area (0.0 if none found)


class Detector:
    def __init__(self, model_path: str, imgsz: int, confidence: float):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.confidence = confidence

    def detect(self, frame) -> Detection:
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w
        frame_cx, frame_cy = frame_w / 2, frame_h / 2

        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        best_area = None
        best_confidence = None
        best_score = None

        # YOLO can see several people (or false positives) in one frame.
        # We only care about whoever is closest to the camera, so we pick
        # the largest box, breaking near-ties by whichever is more central.
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            distance_from_center = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5

            score = (area, -distance_from_center)
            if best_score is None or score > best_score:
                best_score = score
                best_area = area
                best_confidence = float(box.conf[0])

        if best_area is None:
            return Detection(present=False, confidence=0.0, box_area_ratio=0.0)

        return Detection(
            present=True,
            confidence=best_confidence,
            box_area_ratio=best_area / frame_area,
        )
