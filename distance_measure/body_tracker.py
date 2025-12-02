import cv2
import math
import mediapipe as mp
import numpy as np

from helper_functions import get_distance

class BodyTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # "Sticky" tracker settings
        self.min_detection_confidence = 0.6
        self.min_tracking_confidence = 0.7

        # --- Stateful Rotation Tracking ---
        self.accumulated_yaw = 0.0

        # We need to learn the user's max shoulder width to calculate geometry
        self.max_shoulder_width_px = 1.0  # Start non-zero to prevent div/0

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=2,
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


    @staticmethod
    def get_body_rotation(pose_landmarks):
        """
        Calculates body yaw using shoulder alignment.
        Returns yaw in degrees.
        """
        if not pose_landmarks:
            return 0

        lm = pose_landmarks.landmark

        # 11: Left Shoulder, 12: Right Shoulder
        l = lm[11]
        r = lm[12]

        # Calculate vector components
        # dx: Horizontal difference
        dx = r.x - l.x

        # dz: Depth difference.
        # NOTE: MediaPipe Z is not 1:1 with X. It's often smaller.
        #   Scale it up to make the rotation more responsive.
        dz = (r.z - l.z) * 2.5

        # Calculate Angle
        # Negate the result to match the HPE direction (Left +, Right -) if needed
        #   Test: If turning left, Left Shoulder moves back (Z increases), Right moves fwd.
        yaw = math.degrees(math.atan2(dz, dx))

        return yaw


    def detect_body(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        landmarks = getattr(results, 'pose_landmarks', None)

        if landmarks and draw:
            height, width, c = frame.shape
            lm = landmarks.landmark

            # 1. Draw the Standard Skeleton
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # --- 2. DRAW HORIZONTAL AXIS (Yellow) ---
            # Point 11 = Left Shoulder, Point 12 = Right Shoulder
            l_shoulder = (int(lm[11].x * width), int(lm[11].y * height))
            r_shoulder = (int(lm[12].x * width), int(lm[12].y * height))

            # --- 3. DRAW CONNECTION LINE (Shoulder Width) ---
            cv2.line(frame, l_shoulder, r_shoulder, (0, 255, 255), 2)

            # 4. Draw Labeled Points (A = Left/Orange, B = Right/Magenta)
            # LEFT SHOULDER (A) -> Orange (BGR: 0, 165, 255)
            cv2.circle(frame, l_shoulder, 8, (0, 165, 255), -1)
            cv2.putText(frame, "A", (l_shoulder[0] - 5, l_shoulder[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # RIGHT SHOULDER (B) -> Magenta (BGR: 255, 0, 255)
            cv2.circle(frame, r_shoulder, 8, (255, 0, 255), -1)
            cv2.putText(frame, "B", (r_shoulder[0] - 5, r_shoulder[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Debug: Draw "Normal Vector" (Which way is chest facing?)
            # We infer this from the cross product of shoulders and spine, but that's complex.
            # Simple visual check: Is A left of B?
            is_facing_forward = l_shoulder[0] > r_shoulder[0]  # Mirror view logic

            status_text = "BACK" if not is_facing_forward else "FRONT"
            cv2.putText(frame, f"SKELETON: {status_text}", (50, height - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 3. Draw "Synthetic Z" Debug info
            # This shows the user if the "Rigid Bar" logic is working
            mid_x = (l_shoulder[0] + r_shoulder[0]) // 2
            mid_y = (l_shoulder[1] + r_shoulder[1]) // 2

            # Visualize the calculated "facing" direction
            # This is purely visual, derived from the same logic as get_body_rotation
            dx = r_shoulder[0] - l_shoulder[0]
            if self.max_shoulder_width_px > 10:
                # Simple visual projection
                scale_factor = 100
                cv2.arrowedLine(frame, (mid_x, mid_y), (mid_x + int(dx / self.max_shoulder_width_px * 50), mid_y),
                                (0, 0, 255), 2)

        return landmarks

    def get_body_rotation_continuous(self, pose_landmarks):
        """
        Calculates absolute rotation using Geometric Constraint (Synthetic Z).
        Returns angle in degrees (-180 to 180).
        """
        if not pose_landmarks:
            return self.accumulated_yaw

        lm = pose_landmarks.landmark

        # 1. Get Shoulder X Coordinates (Normalized 0.0 - 1.0)
        lx = lm[11].x
        rx = lm[12].x

        # 2. Get Raw Z to determine "Quadrant" (Front vs Back)
        # We don't trust the magnitude, but we trust the sign.
        # If Left Shoulder Z < Right Shoulder Z, Left is closer to camera.
        lz_raw = lm[11].z
        rz_raw = lm[12].z
        left_is_closer = lz_raw < rz_raw

        # 3. Calculate Horizontal Projection (dx)
        # Note: If facing camera (0 deg), Left is on Right of screen (mirror) or Left?
        # Standard: x increases left-to-right on screen.
        # If facing camera: Left Shoulder is at higher X than Right Shoulder.
        dx = lx - rx

        # 4. Auto-Calibrate Max Width (The "Rigid Bar" Length)
        # We track the maximum width seen to define the radius of the turn.
        current_width = abs(dx)
        # Slowly decay max width to adapt to user moving further away (Z-depth change)
        self.max_shoulder_width_px = max(self.max_shoulder_width_px * 0.999, current_width)

        # Safety clamp to prevent math domain errors
        if self.max_shoulder_width_px < 0.001:
            return self.accumulated_yaw

        # 5. Calculate "Synthetic Z" (The missing depth)
        # Pythagorean theorem: dx^2 + dz^2 = max_width^2
        # So: dz = sqrt(max_width^2 - dx^2)

        ratio = current_width / self.max_shoulder_width_px
        ratio = min(ratio, 1.0)  # Clamp to 1.0

        # This gives us the magnitude of the Z component
        synthetic_dz_mag = math.sqrt(1.0 - ratio ** 2)

        # 6. Determine Direction (Sign of Z) using raw data
        # If left is closer, we are rotated one way. If right is closer, the other.
        # The logic below aligns the Synthetic Z with the Raw Z sign.
        sign = 1 if left_is_closer else -1
        synthetic_dz = synthetic_dz_mag * sign

        # 7. Calculate Angle
        # We use atan2(dz, dx) to get the full 360 rotation
        # dx is Cosine component, dz is Sine component
        angle_rad = math.atan2(synthetic_dz, ratio if dx > 0 else -ratio)
        angle_deg = math.degrees(angle_rad)

        # 8. Align Coordinates
        # 0 degrees should be "Facing Camera".
        # When facing camera: dx is Max, dz is 0.
        # atan2(0, 1) = 0. Correct.

        # 9. Handle Quadrants for 360 tracking
        # The geometric math above naturally handles 0-180 and 0 to -180.
        # However, we need to correct for the "Backwards" case.
        # If we are facing away, dx flips.

        # Simplified robust check:
        # Just use the raw dx and the signed synthetic_dz.
        # We normalize dx by max_width to treat it as a unit circle.

        norm_dx = dx / self.max_shoulder_width_px

        # Final Angle Calculation
        final_angle = math.degrees(math.atan2(synthetic_dz, norm_dx))

        # Refine Orientation offset (-90 shift usually needed for compass)
        # Based on standard usage: 0 = Up/Camera.
        # Our math gives 0 when dx is max positive.
        self.accumulated_yaw = final_angle

        return self.accumulated_yaw


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
