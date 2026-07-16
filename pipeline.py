import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import logging

from detection.detector import Detector
from tracking.tracker import Tracker
from homography.homography import HomographyTransformer
from analytics.formations import FormationAnalyzer
from analytics.heatmaps import HeatmapGenerator
from analytics.metrics import MetricsCalculator
from utils.video_utils import VideoLoader
from utils.team_classifier import TeamClassifier

logger = logging.getLogger(__name__)

PITCH_WIDTH = 105.0
PITCH_LENGTH = 68.0


class ProcessingPipeline:
    def __init__(self, model_path: str = "yolov8x.pt",
                 detection_confidence: float = 0.3,
                 process_every_n: int = 5,
                 fps_target: int = 30):

        self.detector = Detector(model_path, confidence=detection_confidence)
        self.tracker = Tracker()
        self.homography = HomographyTransformer()
        self.formation_analyzer = FormationAnalyzer(n_rows=3)
        self.heatmap_generator = HeatmapGenerator()
        self.metrics_calculator = None
        self.team_classifier = TeamClassifier()

        self.process_every_n = process_every_n
        self.fps_target = fps_target

        self.frame_count = 0
        self.processed_frames = 0
        self.is_running = False
        self.progress = 0.0
        self.error = None

        self.player_positions = {
            0: {},
            1: {},
        }
        self.player_trails = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        self.ball_positions = []
        self.all_detections = []
        self.team_assignments = {}

        self.goalkeeper_ids = {0: set(), 1: set()}
        self.referee_ids = set()

        self.annotated_frames_dir = None
        self.annotated_video_path = None
        self.total_frames = 0
        self.video_path = None
        self.video_metadata = None

    def process_video(self, video_path: str, output_dir: str = "outputs",
                      progress_callback=None):
        self.video_path = video_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.annotated_frames_dir = output_dir / "frames"
        self.annotated_frames_dir.mkdir(exist_ok=True)

        loader = VideoLoader(video_path)
        self.video_metadata = loader.metadata
        self.total_frames = loader.metadata.total_frames
        self.frame_count = 0
        self.processed_frames = 0

        self.metrics_calculator = MetricsCalculator(
            fps=loader.metadata.fps,
            pixels_per_meter=1.0,
        )

        self.is_running = True
        self.progress = 0.0
        self.error = None

        self.team_classifier_fitted = False
        self.team_classifier_frame = None
        self.frame_buffer = []

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_video_path = str(output_dir / "annotated_match.mp4")
        writer = None

        frame_idx = 0
        try:
            while True:
                frame = loader.read_frame()
                if frame is None:
                    break

                if frame_idx % self.process_every_n != 0:
                    if writer is not None:
                        writer.write(frame)
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    detections = self.detector.detect(frame_rgb)
                    detections = self.tracker.update(detections)
                except Exception as e:
                    logger.warning(f"Detection/tracking failed on frame {frame_idx}: {e}")
                    if writer is not None:
                        writer.write(frame)
                    frame_idx += 1
                    continue

                if not self.team_classifier_fitted and detections:
                    player_dets = [d for d in detections if d.class_name == "person"]
                    if len(player_dets) >= 5:
                        self.team_classifier.fit(frame, player_dets)
                        self.team_classifier_fitted = True

                if self.team_classifier_fitted:
                    person_dets_for_team = [d for d in detections if d.class_name == "person"]
                    if person_dets_for_team:
                        team_map = self.team_classifier.predict_batch(frame, person_dets_for_team)
                        self.team_assignments.update(team_map)

                if not self.homography.calibrated:
                    try:
                        self.homography.estimate_from_pitch_color(frame)
                    except Exception as e:
                        logger.warning(f"Homography calibration failed on frame {frame_idx}: {e}")

                for det in detections:
                    self.all_detections.append({
                        "frame": frame_idx,
                        "detection": det.to_dict(),
                    })

                person_dets = [d for d in detections if d.class_name == "person" and d.height >= 40]
                ball_dets = [d for d in detections if d.class_name == "ball"]

                pitch_positions = {}

                for det in person_dets:
                    if det.tracker_id is None:
                        continue
                    tid = det.tracker_id
                    team = self.team_assignments.get(tid, -1)

                    cx_pixel = det.cx
                    cy_pixel = det.cy

                    if self.homography.calibrated:
                        px, py = self.homography.transform(cx_pixel, cy_pixel)
                        px = np.clip(px, 0, PITCH_WIDTH)
                        py = np.clip(py, 0, PITCH_LENGTH)
                    else:
                        px = cx_pixel / frame.shape[1] * PITCH_WIDTH
                        py = cy_pixel / frame.shape[0] * PITCH_LENGTH

                    if team not in self.player_positions:
                        self.player_positions[team] = {}
                    if team not in self.player_trails:
                        self.player_trails[team] = defaultdict(list)
                    if tid not in self.player_positions[team]:
                        self.player_positions[team][tid] = []
                    self.player_positions[team][tid].append((px, py))
                    self.player_trails[team][tid].append((px, py))
                    if team not in pitch_positions:
                        pitch_positions[team] = {}
                    pitch_positions[team][tid] = {"x": float(px), "y": float(py)}

                ball_pitch_pos = None
                for det in ball_dets:
                    if det.tracker_id is None:
                        continue
                    cx_pixel = det.cx
                    cy_pixel = det.cy

                    if self.homography.calibrated:
                        bx, by = self.homography.transform(cx_pixel, cy_pixel)
                        bx = np.clip(bx, 0, PITCH_WIDTH)
                        by = np.clip(by, 0, PITCH_LENGTH)
                    else:
                        bx = cx_pixel / frame.shape[1] * PITCH_WIDTH
                        by = cy_pixel / frame.shape[0] * PITCH_LENGTH

                    ball_pitch_pos = {"x": float(bx), "y": float(by)}
                    self.ball_positions.append((frame_idx, ball_pitch_pos))

                self.metrics_calculator.update(
                    frame_idx,
                    pitch_positions.get(0, {}),
                    pitch_positions.get(1, {}),
                    ball_pitch_pos,
                )

                if frame_idx % (self.process_every_n * 30) == 0:
                    self.formation_analyzer.analyze_frame(
                        pitch_positions.get(0, {}),
                        pitch_positions.get(1, {}),
                        frame_idx,
                    )

                try:
                    annotated = self._draw_annotations(frame, detections)
                except Exception as e:
                    logger.warning(f"Annotation failed on frame {frame_idx}: {e}")
                    annotated = frame

                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        output_video_path, fourcc, loader.metadata.fps, (w, h)
                    )

                writer.write(annotated)

                self.processed_frames += 1
                self.frame_count = frame_idx

                if frame_idx % 10 == 0:
                    self.progress = (frame_idx / max(self.total_frames, 1)) * 100
                    if progress_callback:
                        progress_callback(self.progress, frame_idx, self.total_frames)

                frame_idx += 1

        except Exception as e:
            logger.error(f"Pipeline crashed at frame {frame_idx}: {e}", exc_info=True)
            self.error = str(e)
        finally:
            if writer is not None:
                writer.release()
            loader.release()
            self.is_running = False
            self.progress = 100.0 if self.error is None else self.progress

        self.annotated_video_path = output_video_path

        if self.error is None:
            results = self._compile_results()
            with open(output_dir / "analysis_results.json", "w") as f:
                json.dump(results, f, indent=2)
            return results
        return None

    def _draw_annotations(self, frame, detections):
        annotated = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            tid = det.tracker_id

            if det.class_name == "ball":
                color = (0, 0, 0)
                label = f"Ball {tid}" if tid else "Ball"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (int(det.cx), int(det.cy)), 5, (255, 255, 255), -1)
            else:
                team = self.team_assignments.get(tid, -1) if tid is not None else -1
                if team == 0:
                    color = (0, 0, 255)
                elif team == 1:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 255)

                label = f"#{tid}" if tid else "?"
                if team != -1:
                    label += f" T{team}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0] + 4, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 3),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return annotated

    def _compile_results(self):
        player_stats = {}

        for team_id in list(self.player_positions.keys()):
            for pid, history in self.player_positions[team_id].items():
                if not history:
                    continue
                metrics = self.metrics_calculator.calculate_player_metrics(
                    {i: p for i, p in enumerate(history)}
                )
                player_stats[pid] = {
                    "team": team_id,
                    "positions": history,
                    "metrics": metrics,
                    "avg_position": metrics["avg_position"],
                }

        ball_trajectory = [(f, p) for f, p in self.ball_positions]

        formation_changes = self.formation_analyzer.get_formation_changes()

        ball_metrics = self.metrics_calculator.calculate_ball_speed()

        team_a_current_positions = {}
        team_b_current_positions = {}
        for pid, stats in player_stats.items():
            if stats["team"] == 0 and stats["positions"]:
                team_a_current_positions[pid] = stats["positions"][-1]
            elif stats["team"] == 1 and stats["positions"]:
                team_b_current_positions[pid] = stats["positions"][-1]

        team_a_metrics = self.metrics_calculator.calculate_team_metrics(
            {pid: stats["avg_position"] for pid, stats in player_stats.items()
             if stats["team"] == 0}
        )
        team_b_metrics = self.metrics_calculator.calculate_team_metrics(
            {pid: stats["avg_position"] for pid, stats in player_stats.items()
             if stats["team"] == 1}
        )

        return {
            "video_metadata": {
                "path": self.video_path,
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "fps": self.video_metadata.fps if self.video_metadata else 30,
                "width": self.video_metadata.width if self.video_metadata else 0,
                "height": self.video_metadata.height if self.video_metadata else 0,
                "duration_sec": self.video_metadata.duration_sec if self.video_metadata else 0,
            },
            "player_stats": player_stats,
            "ball_trajectory": ball_trajectory,
            "ball_metrics": ball_metrics,
            "formation_changes": formation_changes,
            "dominant_formation_team_a": self.formation_analyzer.get_dominant_formation(0),
            "dominant_formation_team_b": self.formation_analyzer.get_dominant_formation(1),
            "team_a_metrics": team_a_metrics,
            "team_b_metrics": team_b_metrics,
            "formation_history": self.formation_analyzer.formation_history,
            "homography_calibrated": self.homography.calibrated,
        }

    def get_frame_positions(self, frame_idx):
        p0 = {}
        p1 = {}
        ball = None

        for team_id in list(self.player_positions.keys()):
            for pid, history in self.player_positions[team_id].items():
                if isinstance(history, list) and history:
                    idx = min(frame_idx, len(history) - 1)
                    pos = history[idx]
                    if team_id == 0:
                        p0[pid] = {"x": float(pos[0]), "y": float(pos[1])}
                    elif team_id == 1:
                        p1[pid] = {"x": float(pos[0]), "y": float(pos[1])}

        for f_idx, pos in self.ball_positions:
            if f_idx == frame_idx:
                ball = pos
                break

        return p0, p1, ball
