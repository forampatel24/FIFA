from pathlib import Path
import ultralytics
ultralytics.checks()


class DetectionModel:
    MODELS = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
        "l": "yolov8l.pt",
        "x": "yolov8x.pt",
    }

    def __init__(self, model_path: str = "yolov8x.pt", confidence: float = 0.3):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

    def load(self):
        path = Path(self.model_path)
        if not path.exists() and self.model_path in self.MODELS.values():
            pass
        self.model = ultralytics.YOLO(self.model_path)

    def predict(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        return results[0]
