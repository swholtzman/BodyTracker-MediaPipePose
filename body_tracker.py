import cv2
import math
import mediapipe as mp

class BodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # "Sticky" tracker settings
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.5

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def detect_body(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results.pose_landmarks

    def get_shoulder_width_pixels(self, pose_landmarks, frame_width, frame_height):
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

    def get_bounding_box(self, pose_landmarks, frame_width, frame_height):
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