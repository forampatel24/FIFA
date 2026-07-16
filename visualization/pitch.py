import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.colors import LinearSegmentedColormap


PITCH_WIDTH = 105
PITCH_LENGTH = 68

TEAM_COLORS = {
    0: "#e41a1c",
    1: "#377eb8",
    -1: "#808080",
}

BALL_COLOR = "#000000"


class PitchVisualizer:
    def __init__(self, pitch_type: str = "uefa", layout: tuple = None):
        self.pitch_type = pitch_type
        self.layout = layout or (1, 1)
        self.pitch = Pitch(
            pitch_type=pitch_type,
            pitch_color="grass",
            line_color="white",
            linewidth=1.5,
            goal_type="box",
        )

    def _get_fig_ax(self, figsize=(10, 7)):
        fig, ax = self.pitch.draw(figsize=figsize)
        return fig, ax

    def _to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _plot_players(self, ax, players: dict, color: str, label_prefix: str = ""):
        for pid, pos in players.items():
            if isinstance(pos, dict):
                x, y = pos["x"], pos["y"]
            else:
                x, y = pos[0], pos[1]
            ax.scatter(x, y, c=color, s=120, edgecolors="white",
                       linewidths=1.5, zorder=5)
            ax.annotate(f"{label_prefix}{pid}", (x, y), fontsize=7,
                        ha="center", va="center", color="white",
                        fontweight="bold", zorder=6)

    def plot_positions(self, team_a_players: dict, team_b_players: dict,
                       unassigned_players: dict = None, ball_pos=None, figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)
        self._plot_players(ax, team_a_players, TEAM_COLORS[0])
        self._plot_players(ax, team_b_players, TEAM_COLORS[1])
        if unassigned_players:
            self._plot_players(ax, unassigned_players, TEAM_COLORS[-1], "?")

        if ball_pos is not None:
            if isinstance(ball_pos, dict):
                bx, by = ball_pos["x"], ball_pos["y"]
            else:
                bx, by = ball_pos[0], ball_pos[1]
            ax.scatter(bx, by, c=BALL_COLOR, s=80, edgecolors="white",
                       linewidths=2, zorder=7, marker="o")

        return self._to_image(fig)

    def plot_heatmap(self, positions: np.ndarray, figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)
        if len(positions) < 3:
            ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                    color="white", fontsize=12, transform=ax.transAxes)
            return self._to_image(fig)

        stats = self.pitch.bin_statistic(positions[:, 0], positions[:, 1])
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.3), (1, 0, 0, 0.7), (1, 0, 0, 1)]
        cmap = LinearSegmentedColormap.from_list("heatmap", colors, N=256)
        self.pitch.heatmap(stats, ax=ax, cmap=cmap, alpha=0.8)
        return self._to_image(fig)

    def plot_dual_heatmap(self, team_a_positions: np.ndarray, team_b_positions: np.ndarray,
                          figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)

        if len(team_a_positions) >= 3:
            stats_a = self.pitch.bin_statistic(team_a_positions[:, 0], team_a_positions[:, 1])
            cmap_a = LinearSegmentedColormap.from_list(
                "team_a", [(0, 0, 0, 0), (1, 0, 0, 0.8)], N=256
            )
            self.pitch.heatmap(stats_a, ax=ax, cmap=cmap_a, alpha=0.6)

        if len(team_b_positions) >= 3:
            stats_b = self.pitch.bin_statistic(team_b_positions[:, 0], team_b_positions[:, 1])
            cmap_b = LinearSegmentedColormap.from_list(
                "team_b", [(0, 0, 0, 0), (0, 0, 1, 0.8)], N=256
            )
            self.pitch.heatmap(stats_b, ax=ax, cmap=cmap_b, alpha=0.6)

        return self._to_image(fig)

    def plot_movement_trails(self, team_a_trails: dict, team_b_trails: dict,
                             ball_trail: list = None, figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)

        for pid, trail in team_a_trails.items():
            if len(trail) > 1:
                xs = [p.get("x", p[0]) if isinstance(p, dict) else p[0] for p in trail]
                ys = [p.get("y", p[1]) if isinstance(p, dict) else p[1] for p in trail]
                ax.plot(xs, ys, color=TEAM_COLORS[0], alpha=0.4, linewidth=0.8)

        for pid, trail in team_b_trails.items():
            if len(trail) > 1:
                xs = [p.get("x", p[0]) if isinstance(p, dict) else p[0] for p in trail]
                ys = [p.get("y", p[1]) if isinstance(p, dict) else p[1] for p in trail]
                ax.plot(xs, ys, color=TEAM_COLORS[1], alpha=0.4, linewidth=0.8)

        if ball_trail and len(ball_trail) > 1:
            xs = [p.get("x", p[0]) if isinstance(p, dict) else p[0] for p in ball_trail]
            ys = [p.get("y", p[1]) if isinstance(p, dict) else p[1] for p in ball_trail]
            ax.plot(xs, ys, color=BALL_COLOR, alpha=0.5, linewidth=1.5)

        return self._to_image(fig)

    def plot_average_positions(self, team_a_avg: dict, team_b_avg: dict,
                               figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)

        for pid, pos in team_a_avg.items():
            x, y = pos["x"], pos["y"]
            ax.scatter(x, y, c=TEAM_COLORS[0], s=200, edgecolors="white",
                       linewidths=2, zorder=5, alpha=0.7)

        for pid, pos in team_b_avg.items():
            x, y = pos["x"], pos["y"]
            ax.scatter(x, y, c=TEAM_COLORS[1], s=200, edgecolors="white",
                       linewidths=2, zorder=5, alpha=0.7)

        return self._to_image(fig)

    def plot_formation(self, positions: np.ndarray, formation_label: str,
                       team_color: str = "#e41a1c", figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)

        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], c=team_color, s=150,
                       edgecolors="white", linewidths=2, zorder=5)

        ax.set_title(f"Formation: {formation_label}", fontsize=14, color="white",
                     fontweight="bold", pad=15)

        return self._to_image(fig)

    def plot_player_heatmap(self, player_positions: list, figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)

        if len(player_positions) >= 3:
            positions = np.array(player_positions)
            self.pitch.kdeplot(
                positions[:, 0], positions[:, 1], ax=ax,
                fill=True, levels=20, alpha=0.6,
                cmap="hot",
            )

        return self._to_image(fig)

    def plot_empty_pitch(self, figsize=(10, 7)):
        fig, ax = self._get_fig_ax(figsize)
        return self._to_image(fig)
