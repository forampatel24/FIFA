import numpy as np
import cv2


PITCH_WIDTH_M = 105.0
PITCH_LENGTH_M = 68.0

PITCH_TEMPLATE_POINTS = np.array([
    [0.0, 0.0],
    [PITCH_WIDTH_M, 0.0],
    [PITCH_WIDTH_M, PITCH_LENGTH_M],
    [0.0, PITCH_LENGTH_M],
], dtype=np.float32)


class HomographyTransformer:
    def __init__(self):
        self.H = None
        self.H_inv = None
        self.calibrated = False

    def calibrate(self, image_points: np.ndarray):
        if len(image_points) != 4:
            raise ValueError("Exactly 4 point correspondences required")

        image_points = np.array(image_points, dtype=np.float32)

        self.H, _ = cv2.findHomography(image_points, PITCH_TEMPLATE_POINTS)
        self.H_inv = cv2.invert(self.H)[1] if self.H is not None else None
        self.calibrated = self.H is not None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            return points

        pts = np.array(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)

        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([pts, ones])
        transformed = (self.H @ homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        return transformed

    def transform(self, x: float, y: float):
        result = self.transform_points(np.array([[x, y]]))
        return result[0, 0], result[0, 1]

    def estimate_from_pitch_color(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.calibrated = False
            return False

        pitch_contour = max(contours, key=cv2.contourArea)
        pitch_area = cv2.contourArea(pitch_contour)
        frame_area = frame.shape[0] * frame.shape[1]

        if pitch_area < 0.15 * frame_area:
            self.calibrated = False
            return False

        epsilon = 0.02 * cv2.arcLength(pitch_contour, True)
        approx = cv2.approxPolyDP(pitch_contour, epsilon, True)

        if len(approx) < 4:
            self.calibrated = False
            return False

        rect = cv2.minAreaRect(pitch_contour)
        box_w, box_h = rect[1]
        if box_w > 0 and box_h > 0:
            aspect = max(box_w, box_h) / min(box_w, box_h)
            if aspect < 1.2 or aspect > 2.5:
                self.calibrated = False
                return False

        corners = cv2.boxPoints(rect)
        corners = np.array(sorted(corners, key=lambda p: (p[1], p[0])))

        corners_sorted = np.zeros((4, 2), dtype=np.float32)
        corners_sorted[0] = corners[0]
        corners_sorted[1] = corners[1]
        corners_sorted[2] = corners[3]
        corners_sorted[3] = corners[2]

        self.calibrate(corners_sorted)
        return True

    def draw_pitch_overlay(self, frame: np.ndarray):
        if not self.calibrated:
            return frame

        pitch_corners = PITCH_TEMPLATE_POINTS
        img_corners = self.transform_points(pitch_corners)
        img_corners = self.inverse_transform_points(pitch_corners)

        return frame

    def inverse_transform_points(self, pitch_points: np.ndarray) -> np.ndarray:
        if not self.calibrated or self.H_inv is None:
            return pitch_points

        pts = np.array(pitch_points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)

        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        homogeneous = np.hstack([pts, ones])
        transformed = (self.H_inv @ homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]

        return transformed
