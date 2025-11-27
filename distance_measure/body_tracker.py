import cv2
import math
import mediapipe as mp

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
            model_complexity=1, # 1 is default, 2 is more accurate but slower
        )

        # --- Calibration State ---
        self.focal_ratio = None

    @staticmethod
    def get_bounding_box(pose_landmarks, frame_width, frame_height):
        """
        Returns the bounding box coordinates (x_min, y_min, x_max, y_max)
        """
        if not pose_landmarks:
            return None

        x_list = [landmark.x for landmark in pose_landmarks.landmark]
        y_list = [landmark.y for landmark in pose_landmarks.landmark]

        bbox_min_x = int(min(x_list) * frame_width)
        bbox_max_x = int(max(x_list) * frame_width)
        bbox_min_y = int(min(y_list) * frame_height)
        bbox_max_y = int(max(y_list) * frame_height)

        return bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y

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

            # 2. Draw the "Measurement Ruler" (Shoulder to Shoulder)
            h, w, c = frame.shape
            lm = landmarks.landmark  # Access the list of 33 points

            # Point 11 = Left Shoulder, Point 12 = Right Shoulder
            l_shoulder = (int(lm[11].x * w), int(lm[11].y * h))
            r_shoulder = (int(lm[12].x * w), int(lm[12].y * h))

            # Draw Yellow Line (Thickness 3)
            cv2.line(frame, l_shoulder, r_shoulder, (0, 255, 255), 3)

        return landmarks


    def train(self, known_distance_cm, shoulder_pixel_width):
        """
        Learns the shoulder width ratio.
        """
        if shoulder_pixel_width > 0:
            self.focal_ratio = known_distance_cm * shoulder_pixel_width


    def get_distance(self, shoulder_pixel_width):
        if self.focal_ratio is None or shoulder_pixel_width == 0:
            return 0
        return self.focal_ratio / shoulder_pixel_width
