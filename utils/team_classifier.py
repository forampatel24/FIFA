import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict


GRASS_COLOR_LOWER = np.array([35, 40, 40], dtype=np.uint8)
GRASS_COLOR_UPPER = np.array([85, 255, 255], dtype=np.uint8)

REFEREE_COLORS = [
    (0, 0, 0),
    (255, 255, 255),
    (50, 50, 50),
    (200, 200, 200),
]


class TeamClassifier:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters
        self.team_colors = {}
        self.team_models = {}
        self.fitted = False

    def _extract_jersey_colors(self, frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return np.array([])

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([])

        jersey_roi = crop[int(crop.shape[0] * 0.2):int(crop.shape[0] * 0.7),
                          int(crop.shape[1] * 0.1):int(crop.shape[1] * 0.9)]

        if jersey_roi.size == 0:
            return np.array([])

        hsv = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, GRASS_COLOR_LOWER, GRASS_COLOR_UPPER)
        non_grass = jersey_roi[grass_mask == 0]

        if non_grass.size == 0:
            non_grass = jersey_roi

        pixels = non_grass.reshape(-1, 3)
        if len(pixels) > 1000:
            idx = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[idx]

        return pixels

    def _is_referee(self, dominant_color: np.ndarray) -> bool:
        for ref_color in REFEREE_COLORS:
            diff = np.abs(dominant_color.astype(int) - np.array(ref_color, dtype=int))
            if np.all(diff < 30):
                return True
        return False

    def fit(self, frame: np.ndarray, player_detections: list, goalkeeper_detections: list = None):
        all_player_pixels = []

        player_bboxes = []
        for det in player_detections:
            pixels = self._extract_jersey_colors(frame, det.bbox)
            if len(pixels) > 10:
                all_player_pixels.append(pixels)
                player_bboxes.append(det.bbox)

        if goalkeeper_detections:
            for det in goalkeeper_detections:
                pixels = self._extract_jersey_colors(frame, det.bbox)
                if len(pixels) > 10:
                    all_player_pixels.append(pixels)

        if not all_player_pixels:
            self.fitted = False
            return

        all_pixels = np.vstack(all_player_pixels)
        all_pixels = all_pixels.astype(np.float32)

        if len(all_pixels) < self.n_clusters * 10:
            self.fitted = False
            return

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_pixels)

        colors = kmeans.cluster_centers_.astype(int)

        non_referee_colors = [c for c in colors if not self._is_referee(c)]
        if len(non_referee_colors) >= 2:
            colors = np.array(non_referee_colors[:2])
        elif len(non_referee_colors) == 1:
            other = [c for c in colors if self._is_referee(c)]
            if other:
                colors = np.array([non_referee_colors[0], other[0]])
            else:
                colors = np.array([non_referee_colors[0], non_referee_colors[0]])

        brightness = np.sum(colors, axis=1)
        if brightness[0] > brightness[1]:
            self.team_colors = {0: colors[0].tolist(), 1: colors[1].tolist()}
        else:
            self.team_colors = {0: colors[1].tolist(), 1: colors[0].tolist()}

        self.kmeans = kmeans
        self.fitted = True

    def predict(self, frame: np.ndarray, bbox):
        if not self.fitted:
            return -1

        pixels = self._extract_jersey_colors(frame, bbox)
        if len(pixels) == 0:
            return -1

        pixels = pixels.astype(np.float32)
        labels = self.kmeans.predict(pixels)
        label_counts = np.bincount(labels, minlength=self.n_clusters)
        team_id = int(np.argmax(label_counts))

        dominant = self.kmeans.cluster_centers_[team_id]
        if self._is_referee(dominant):
            return -1

        return team_id

    def predict_batch(self, frame: np.ndarray, detections: list) -> dict:
        results = {}
        for det in detections:
            if det.class_name not in ("person", "player", "goalkeeper"):
                continue
            if det.tracker_id is None:
                continue
            team = self.predict(frame, det.bbox)
            results[det.tracker_id] = team
        return results

    def get_team_color(self, team_id: int, bgr: bool = True):
        if team_id not in self.team_colors:
            return (128, 128, 128)

        color = self.team_colors[team_id]
        if bgr:
            return (int(color[2]), int(color[1]), int(color[0]))
        return tuple(color)

    def get_team_hex_color(self, team_id: int) -> str:
        if team_id not in self.team_colors:
            return "#808080"

        color = self.team_colors[team_id]
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
