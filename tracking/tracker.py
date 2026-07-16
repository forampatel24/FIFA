import warnings
import numpy as np

warnings.filterwarnings("ignore", message="The `ByteTrack` was deprecated")

from supervision import ByteTrack as SupervisionByteTrack
from supervision.detection.core import Detections
from detection.detector import Detection


class Tracker:
    def __init__(self, track_activation_threshold: float = 0.5, lost_track_buffer: int = 90):
        self.tracker = SupervisionByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=0.9,
        )
        self._next_uid = 1
        self._tracker_id_map = {}

    def _map_external_ids(self, sup_tracker_ids: np.ndarray) -> np.ndarray:
        mapped = []
        for sid in sup_tracker_ids.tolist():
            sid = int(sid)
            if sid not in self._tracker_id_map:
                self._tracker_id_map[sid] = self._next_uid
                self._next_uid += 1
            mapped.append(self._tracker_id_map[sid])
        return np.array(mapped, dtype=int)

    def update(self, detections: list):
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sup_detections = Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_ids,
        )

        tracked = self.tracker.update_with_detections(sup_detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return []

        tracked_ids = self._map_external_ids(tracked.tracker_id)
        tracked_boxes = tracked.xyxy

        for i, det in enumerate(detections):
            det_box = np.array(det.bbox, dtype=np.float32)
            match = np.where(np.all(np.isclose(tracked_boxes, det_box, atol=1e-4), axis=1))[0]
            if len(match) > 0:
                idx_in_tracked = match[0]
                if idx_in_tracked < len(tracked_ids):
                    det.tracker_id = int(tracked_ids[idx_in_tracked])

        return [d for d in detections if d.tracker_id is not None]
