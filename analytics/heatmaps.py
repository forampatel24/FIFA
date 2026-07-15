import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


PITCH_WIDTH = 105.0
PITCH_LENGTH = 68.0


class HeatmapGenerator:
    def __init__(self, bins: tuple = (50, 50), sigma: float = 1.8):
        self.bins = bins
        self.sigma = sigma

    def generate(self, positions: np.ndarray):
        if len(positions) == 0:
            return np.zeros(self.bins)

        positions = np.array(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 2)

        heatmap, x_edges, y_edges = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=self.bins,
            range=[[0, PITCH_WIDTH], [0, PITCH_LENGTH]],
        )

        heatmap = gaussian_filter(heatmap, sigma=self.sigma)
        heatmap = heatmap.T

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_team_heatmap(self, all_positions: dict):
        all_pts = []
        for pid, hist in all_positions.items():
            if isinstance(hist, np.ndarray) and hist.ndim == 2:
                all_pts.extend(hist.tolist())
            elif isinstance(hist, list):
                all_pts.extend(hist)

        if not all_pts:
            return np.zeros(self.bins)

        return self.generate(np.array(all_pts))

    def generate_player_heatmap(self, player_positions: list):
        if not player_positions:
            return np.zeros(self.bins)

        return self.generate(np.array(player_positions))
