import numpy as np
from .model import DetectionModel


CLASS_MAP = {
    0: "person",
    32: "ball",
}

GOALKEEPER_LABEL = "goalkeeper"
REFEREE_LABEL = "referee"
PLAYER_LABEL = "player"


class Detection:
    def __init__(self, bbox, confidence, class_id, class_name, tracker_id=None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.tracker_id = tracker_id

    @property
    def x1(self):
        return self.bbox[0]

    @property
    def y1(self):
        return self.bbox[1]

    @property
    def x2(self):
        return self.bbox[2]

    @property
    def y2(self):
        return self.bbox[3]

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "tracker_id": self.tracker_id,
        }

    def __repr__(self):
        return f"Detection({self.class_name}, id={self.tracker_id}, conf={self.confidence:.2f})"


class Detector:
    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.3):
        self.model = DetectionModel(model_path, confidence)
        self.model.load()
        self.confidence = confidence

    def detect(self, frame: np.ndarray):
        results = self.model.predict(frame)
        detections = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            if cls in CLASS_MAP:
                det = Detection(
                    bbox=box.tolist(),
                    confidence=float(score),
                    class_id=int(cls),
                    class_name=CLASS_MAP[cls],
                )
                detections.append(det)

        return detections
