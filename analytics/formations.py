import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


FORMATION_MAP = {
    (4, 4, 2): "4-4-2",
    (4, 3, 3): "4-3-3",
    (3, 5, 2): "3-5-2",
    (3, 4, 3): "3-4-3",
    (5, 4, 1): "5-4-1",
    (5, 3, 2): "5-3-2",
    (4, 2, 3, 1): "4-2-3-1",
    (4, 1, 4, 1): "4-1-4-1",
    (4, 5, 1): "4-5-1",
    (3, 6, 1): "3-6-1",
    (4, 3, 2, 1): "4-3-2-1",
    (4, 4, 1, 1): "4-4-1-1",
    (3, 4, 2, 1): "3-4-2-1",
    (3, 5, 1, 1): "3-5-1-1",
}


class FormationAnalyzer:
    def __init__(self, n_rows: int = 3):
        self.n_rows = n_rows
        self.formation_history = []

    def _sort_by_pitch_position(self, positions: np.ndarray):
        sorted_idx = np.argsort(positions[:, 1])
        return positions[sorted_idx]

    def _cluster_rows(self, positions: np.ndarray, team_id: int = 0):
        if len(positions) < self.n_rows:
            return []

        y_coords = positions[:, 1].reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_rows, random_state=42, n_init=10)
        labels = kmeans.fit_predict(y_coords)

        row_centers = {}
        for i, label in enumerate(labels):
            if label not in row_centers:
                row_centers[label] = []
            row_centers[label].append(i)

        sorted_rows = sorted(row_centers.items(), key=lambda x: kmeans.cluster_centers_[x[0]][0])

        row_counts = []
        for _, indices in sorted_rows:
            row_counts.append(len(indices))

        return tuple(row_counts)

    def _classify_formation(self, row_counts: tuple) -> str:
        if row_counts in FORMATION_MAP:
            return FORMATION_MAP[row_counts]

        sorted_counts = tuple(sorted(row_counts, reverse=True))
        if sorted_counts in FORMATION_MAP:
            return FORMATION_MAP[sorted_counts]

        return "-".join(str(c) for c in row_counts)

    def analyze(self, positions: dict, team_id: int = 0):
        team_positions = []
        for pid, pos in positions.items():
            if isinstance(pos, dict):
                team_positions.append([pos.get("x", 0), pos.get("y", 0)])
            else:
                team_positions.append([pos[0], pos[1]])

        team_positions = np.array(team_positions)

        if len(team_positions) < 3:
            return "Unknown", team_positions

        sorted_positions = self._sort_by_pitch_position(team_positions)

        row_counts = self._cluster_rows(sorted_positions, team_id)

        if not row_counts:
            return "Unknown", sorted_positions

        formation = self._classify_formation(row_counts)

        return formation, sorted_positions

    def analyze_frame(self, team_a_positions: dict, team_b_positions: dict, frame_idx: int):
        formation_a, pos_a = self.analyze(team_a_positions, team_id=0)
        formation_b, pos_b = self.analyze(team_b_positions, team_id=1)

        entry = {
            "frame": frame_idx,
            "team_a_formation": formation_a,
            "team_b_formation": formation_b,
            "team_a_positions": pos_a.tolist() if hasattr(pos_a, 'tolist') else pos_a,
            "team_b_positions": pos_b.tolist() if hasattr(pos_b, 'tolist') else pos_b,
        }

        self.formation_history.append(entry)
        return entry

    def get_formation_changes(self):
        if not self.formation_history:
            return []

        changes = []
        last_a = None
        last_b = None

        for entry in self.formation_history:
            if entry["team_a_formation"] != last_a or entry["team_b_formation"] != last_b:
                changes.append(entry)
                last_a = entry["team_a_formation"]
                last_b = entry["team_b_formation"]

        return changes

    def get_dominant_formation(self, team_id: int = 0):
        if not self.formation_history:
            return "Unknown"

        formations = []
        for entry in self.formation_history:
            f = entry["team_a_formation"] if team_id == 0 else entry["team_b_formation"]
            formations.append(f)

        counter = Counter(formations)
        return counter.most_common(1)[0][0] if counter else "Unknown"

    def reset(self):
        self.formation_history = []
