import numpy as np
from collections import defaultdict


PITCH_WIDTH = 105.0
PITCH_LENGTH = 68.0


class MetricsCalculator:
    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 1.0):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.team_a_histories = defaultdict(dict)
        self.team_b_histories = defaultdict(dict)
        self.ball_history = []
        self.frame_count = 0

    def update(self, frame_idx: int, team_a_players: dict, team_b_players: dict, ball_pos=None):
        self.frame_count = max(self.frame_count, frame_idx + 1)

        for pid, pos in team_a_players.items():
            self.team_a_histories[pid][frame_idx] = pos

        for pid, pos in team_b_players.items():
            self.team_b_histories[pid][frame_idx] = pos

        if ball_pos is not None:
            self.ball_history.append((frame_idx, ball_pos))

    def calculate_player_distance(self, player_history: dict) -> float:
        sorted_frames = sorted(player_history.keys())
        if len(sorted_frames) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(sorted_frames)):
            prev = player_history[sorted_frames[i - 1]]
            curr = player_history[sorted_frames[i]]

            if isinstance(prev, dict):
                prev = (prev["x"], prev["y"])
            if isinstance(curr, dict):
                curr = (curr["x"], curr["y"])

            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)

        return total_distance

    def calculate_speed(self, player_history: dict) -> dict:
        sorted_frames = sorted(player_history.keys())
        if len(sorted_frames) < 3:
            return {"avg_speed": 0.0, "max_speed": 0.0}

        speeds = []
        for i in range(1, len(sorted_frames)):
            prev = player_history[sorted_frames[i - 1]]
            curr = player_history[sorted_frames[i]]
            dt = (sorted_frames[i] - sorted_frames[i - 1]) / self.fps

            if dt <= 0:
                continue

            if isinstance(prev, dict):
                prev = (prev["x"], prev["y"])
            if isinstance(curr, dict):
                curr = (curr["x"], curr["y"])

            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            speed = dist / dt
            speeds.append(speed)

        if not speeds:
            return {"avg_speed": 0.0, "max_speed": 0.0}

        return {
            "avg_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
        }

    def calculate_average_position(self, player_history: dict):
        if not player_history:
            return {"x": PITCH_WIDTH / 2, "y": PITCH_LENGTH / 2}

        xs, ys = [], []
        for pos in player_history.values():
            if isinstance(pos, dict):
                xs.append(pos["x"])
                ys.append(pos["y"])
            else:
                xs.append(pos[0])
                ys.append(pos[1])

        return {"x": float(np.mean(xs)), "y": float(np.mean(ys))}

    def calculate_player_metrics(self, player_history: dict):
        distance = self.calculate_player_distance(player_history)
        speed = self.calculate_speed(player_history)
        avg_pos = self.calculate_average_position(player_history)

        return {
            "distance_covered": distance,
            "avg_speed": speed["avg_speed"],
            "max_speed": speed["max_speed"],
            "avg_position": avg_pos,
            "frames_tracked": len(player_history),
        }

    def calculate_team_width(self, positions: dict) -> float:
        xs = []
        for pos in positions.values():
            if isinstance(pos, dict):
                xs.append(pos["x"])
            else:
                xs.append(pos[0])

        if len(xs) < 2:
            return 0.0

        return float(np.max(xs) - np.min(xs))

    def calculate_team_depth(self, positions: dict) -> float:
        ys = []
        for pos in positions.values():
            if isinstance(pos, dict):
                ys.append(pos["y"])
            else:
                ys.append(pos[1])

        if len(ys) < 2:
            return 0.0

        return float(np.max(ys) - np.min(ys))

    def calculate_center_of_mass(self, positions: dict):
        xs, ys = [], []
        for pos in positions.values():
            if isinstance(pos, dict):
                xs.append(pos["x"])
                ys.append(pos["y"])
            else:
                xs.append(pos[0])
                ys.append(pos[1])

        if not xs:
            return {"x": PITCH_WIDTH / 2, "y": PITCH_LENGTH / 2}

        return {"x": float(np.mean(xs)), "y": float(np.mean(ys))}

    def calculate_defensive_line(self, positions: dict, attacking_direction: int = 1) -> float:
        ys = []
        for pos in positions.values():
            if isinstance(pos, dict):
                ys.append(pos["y"])
            else:
                ys.append(pos[1])

        if not ys:
            return PITCH_LENGTH / 2

        if attacking_direction == 1:
            return float(np.min(ys))
        else:
            return float(np.max(ys))

    def calculate_attacking_line(self, positions: dict, attacking_direction: int = 1) -> float:
        ys = []
        for pos in positions.values():
            if isinstance(pos, dict):
                ys.append(pos["y"])
            else:
                ys.append(pos[1])

        if not ys:
            return PITCH_LENGTH / 2

        if attacking_direction == 1:
            return float(np.max(ys))
        else:
            return float(np.min(ys))

    def calculate_team_metrics(self, positions: dict, attacking_direction: int = 1):
        if not positions:
            return {
                "width": 0,
                "depth": 0,
                "center_of_mass": {"x": PITCH_WIDTH / 2, "y": PITCH_LENGTH / 2},
                "defensive_line": PITCH_LENGTH / 2,
                "attacking_line": PITCH_LENGTH / 2,
                "compactness": 0,
            }

        width = self.calculate_team_width(positions)
        depth = self.calculate_team_depth(positions)
        com = self.calculate_center_of_mass(positions)
        def_line = self.calculate_defensive_line(positions, attacking_direction)
        att_line = self.calculate_attacking_line(positions, attacking_direction)

        return {
            "width": width,
            "depth": depth,
            "compactness": np.sqrt(width ** 2 + depth ** 2) if width and depth else 0,
            "center_of_mass": com,
            "defensive_line": def_line,
            "attacking_line": att_line,
            "num_players": len(positions),
        }

    def estimate_possession(self, team_a_touches: int, team_b_touches: int) -> dict:
        total = team_a_touches + team_b_touches
        if total == 0:
            return {"team_a": 50.0, "team_b": 50.0}

        return {
            "team_a": round(team_a_touches / total * 100, 1),
            "team_b": round(team_b_touches / total * 100, 1),
        }

    def calculate_ball_speed(self) -> dict:
        if len(self.ball_history) < 2:
            return {"avg_speed": 0.0, "max_speed": 0.0}

        speeds = []
        for i in range(1, len(self.ball_history)):
            f_prev, pos_prev = self.ball_history[i - 1]
            f_curr, pos_curr = self.ball_history[i]
            dt = (f_curr - f_prev) / self.fps

            if dt <= 0:
                continue

            if isinstance(pos_prev, dict):
                pos_prev = (pos_prev["x"], pos_prev["y"])
            if isinstance(pos_curr, dict):
                pos_curr = (pos_curr["x"], pos_curr["y"])

            dx = pos_curr[0] - pos_prev[0]
            dy = pos_curr[1] - pos_prev[1]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            speed = dist / dt
            speeds.append(speed)

        if not speeds:
            return {"avg_speed": 0.0, "max_speed": 0.0}

        return {
            "avg_speed": float(np.mean(speeds)),
            "max_speed": float(np.max(speeds)),
        }
