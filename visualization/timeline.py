import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class TimelineVisualizer:
    def __init__(self):
        self.events = []

    def add_event(self, frame_idx: int, event_type: str, description: str):
        self.events.append({
            "frame": frame_idx,
            "type": event_type,
            "description": description,
        })

    def render(self, formation_history: list = None, total_frames: int = 1,
               fps: float = 30.0, figsize=(12, 3)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        if not formation_history:
            ax.text(0.5, 0.5, "No formation data available",
                    ha="center", va="center", color="white", fontsize=12,
                    transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return self._to_image(fig)

        duration_min = total_frames / fps / 60

        formation_changes = self._find_formation_changes(formation_history)

        if not formation_changes:
            ax.text(0.5, 0.5, "No formation changes detected",
                    ha="center", va="center", color="white", fontsize=12,
                    transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return self._to_image(fig)

        times_min = [f["frame"] / fps / 60 for f in formation_changes]
        formations_a = [f["team_a_formation"] for f in formation_changes]
        formations_b = [f["team_b_formation"] for f in formation_changes]

        colors = plt.cm.Set3(np.linspace(0, 1, len(formations_a)))

        for i, (t, fa, fb) in enumerate(zip(times_min, formations_a, formations_b)):
            ax.barh(1, 0.8, left=t, height=0.6, color=colors[i], alpha=0.8,
                    edgecolor="white", linewidth=0.5)
            label = f"{fa}"
            ax.text(t + 0.4, 1, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black",
                    rotation=45 if len(label) > 5 else 0)

        ax.set_xlim(0, duration_min)
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel("Match Time (minutes)", color="white", fontsize=10)
        ax.set_title("Formation Timeline", color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        for spine in ax.spines.values():
            spine.set_color("#333333")

        plt.tight_layout()
        return self._to_image(fig)

    def _find_formation_changes(self, formation_history: list) -> list:
        if not formation_history:
            return []

        changes = [formation_history[0]]
        last_entry = formation_history[0]

        for entry in formation_history[1:]:
            if (entry["team_a_formation"] != last_entry["team_a_formation"] or
                    entry["team_b_formation"] != last_entry["team_b_formation"]):
                changes.append(entry)
                last_entry = entry

        return changes

    def _to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)
        return buf

    def create_event_timeline(self, events: list, total_frames: int, fps: float,
                              figsize=(12, 2)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        if not events:
            ax.text(0.5, 0.5, "No events detected", ha="center", va="center",
                    color="white", fontsize=12, transform=ax.transAxes)
            ax.axis("off")
            return self._to_image(fig)

        duration_min = total_frames / fps / 60

        event_colors = {
            "formation": "#ff6b6b",
            "ball_transition": "#4ecdc4",
            "press": "#ffe66d",
            "attack": "#95e1d3",
            "default": "#aaaaaa",
        }

        y_positions = {
            "formation": 2,
            "ball_transition": 1,
            "press": 0.5,
            "attack": 1.5,
        }

        for event in events:
            t = event["frame"] / fps / 60
            etype = event.get("type", "default")
            color = event_colors.get(etype, event_colors["default"])
            y = y_positions.get(etype, 1)

            ax.scatter(t, y, c=color, s=80, zorder=5, edgecolors="white", linewidths=1)
            ax.annotate(event["description"], (t, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=7, color="white", rotation=30)

        ax.set_xlim(0, duration_min)
        ax.set_ylim(0, 3)
        ax.set_xlabel("Match Time (minutes)", color="white", fontsize=10)
        ax.set_title("Match Events Timeline", color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        for spine in ax.spines.values():
            spine.set_color("#333333")

        plt.tight_layout()
        return self._to_image(fig)
