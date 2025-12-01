import cv2
import math
import mediapipe as mp

from helper_functions import get_distance

class BodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # "Sticky" tracker settings: Increased min_tracking_confidence to 0.7
        # This prevents the skeleton from "glitching" when you turn around quickly.
        self.min_detection_confidence = 0.6
        self.min_tracking_confidence = 0.7

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=2, # 1 is default, 2 is more accurate but slower
        )

        # Calibration State
        self.focal_ratio_width = None
        self.focal_ratio_height = None

    @staticmethod
    def get_bounding_box(pose_landmarks, frame_width, frame_height):
        """
        Returns the bounding box coordinates (x_min, y_min, x_max, y_max)
        """
        if not pose_landmarks:
            return None

        x_list = [landmark.x for landmark in pose_landmarks.landmark]
        y_list = [landmark.y for landmark in pose_landmarks.landmark]

        return (int(min(x_list) * frame_width), int(min(y_list) * frame_height),
                int(max(x_list) * frame_width), int(max(y_list) * frame_height))

    @staticmethod
    def get_body_dimensions(pose_landmarks, frame_width, frame_height):
        """
        Returns (width_px, height_px)
        Width: Left Shoulder (11) to Right Shoulder (12)
        Height: Mid-Shoulder to Mid-Hip (Torso Length) - stable during rotation
        """
        if not pose_landmarks:
            return 0, 0

        lm = pose_landmarks.landmark

        # Helpers to get pixel coords
        def get_pt(idx):
            return int(lm[idx].x * frame_width), int(lm[idx].y * frame_height)

        l_shoulder = get_pt(11)
        r_shoulder = get_pt(12)
        l_hip = get_pt(23)
        r_hip = get_pt(24)

        # 1. Width: Shoulder to Shoulder
        width_px = math.dist(l_shoulder, r_shoulder)

        # 2. Height: Mid-Shoulder to Mid-Hip
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

        height_px = math.dist(mid_shoulder, mid_hip)

        return width_px, height_px

    # Legacy wrapper
    @staticmethod
    def get_shoulder_width_pixels(pose_landmarks, frame_width, frame_height):
        """
        Calculates the Euclidean distance between left (11) and right (12) shoulders in pixels.
        """
        if not pose_landmarks:
            return 0

        # Extract landmarks 11 and 12
        left_shoulder = pose_landmarks.landmark[11]
        right_shoulder = pose_landmarks.landmark[12]

        # Convert normalized coordinates (0.0-1.0) to pixels
        ls_point = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
        rs_point = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))

        # Calculate Euclidean distance
        pixel_width = math.dist(ls_point, rs_point)
        return pixel_width

    def detect_body(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        landmarks = getattr(results, 'pose_landmarks', None)

        if landmarks and draw:
            # 1. Draw the Standard Skeleton
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 2. Prepare for Custom Drawing
            height, width, c = frame.shape
            lm = landmarks.landmark  # <--- Define 'lm' here so we can use it below

            # --- DRAW HORIZONTAL AXIS (Yellow) ---
            # Point 11 = Left Shoulder, Point 12 = Right Shoulder
            l_shoulder = (int(lm[11].x * width), int(lm[11].y * height))
            r_shoulder = (int(lm[12].x * width), int(lm[12].y * height))

            # Draw Yellow Line (Shoulder Width)
            cv2.line(frame, l_shoulder, r_shoulder, (0, 255, 255), 3)

            # --- DRAW VERTICAL AXIS (Magenta) ---
            # We need this to visualize the "Height" tracking that prevents the rotation bug

            # Calculate Mid-Shoulder Point
            mid_shoulder_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
            mid_shoulder_y = int((l_shoulder[1] + r_shoulder[1]) / 2)

            # Calculate Mid-Hip Point (23=Left Hip, 24=Right Hip)
            l_hip = (int(lm[23].x * width), int(lm[23].y * height))
            r_hip = (int(lm[24].x * width), int(lm[24].y * height))
            mid_hip_x = int((l_hip[0] + r_hip[0]) / 2)
            mid_hip_y = int((l_hip[1] + r_hip[1]) / 2)

            # Draw Magenta Line (Torso Height)
            cv2.line(frame, (mid_shoulder_x, mid_shoulder_y), (mid_hip_x, mid_hip_y), (255, 0, 255), 3)

        return landmarks


    def train(self, known_distance_cm, width_px, height_px):
        """
        Learns the shoulder width ratio.
        """
        if width_px > 0:
            self.focal_ratio_width = known_distance_cm * width_px
        if height_px > 0:
            self.focal_ratio_height = known_distance_cm * height_px


    def get_distance(self, width_px, height_px):
        """

        :param width_px:
        :param height_px:
        :return:
        """
        return get_distance(self, width_px, height_px)
