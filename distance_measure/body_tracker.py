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
        self.prev_shoulders = None  # Stores (x, z) of left/right from previous frame
        self.last_valid_delta = 0.0  # Momentum for coasting

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

        return landmarks


    def get_body_rotation_continuous(self, pose_landmarks):
        """
        Calculates continuous rotation using Vector Delta tracking.
        This method is robust to the "Singularity" (90-degree turn) by detecting
        vector direction flips.
        """
        if not pose_landmarks:
            return self.accumulated_yaw

        lm = pose_landmarks.landmark

        # High Z-Sensitivity for rotation responsiveness
        l_curr = np.array([lm[11].x, lm[11].z * 3.5])
        r_curr = np.array([lm[12].x, lm[12].z * 3.5])

        # Calculate Shoulder Width (2D plane)
        # When this is small, we are at 90 or 270 degrees (Profile view)
        shoulder_width_2d = abs(l_curr[0] - r_curr[0])

        # INCREASED THRESHOLD: 15% of screen width
        # This catches the flip EARLIER, switching to coasting before the math explodes.
        CROSSOVER_THRESHOLD = 0.15

        if self.prev_shoulders is not None:
            l_prev, r_prev = self.prev_shoulders

            # --- CALCULATE ROTATION DELTA ---
            # We use the 2D projected change of the shoulder bar length and orientation.
            # Ideally, we'd use atan2(dz, dx), but since we lack reliable Z,
            # we infer rotation from the change in the 2D projection.

            # Simplified Logic:
            # The shoulder bar is a rigid rod.
            # If 'dx' gets smaller, we are turning away from 0 or 180.
            # The SIGN of the change tells us the direction.

            # 1. Calculate the 'Heading' of the shoulder vector in 2D space
            #    This isn't user yaw, this is just the tilt of shoulders on screen.
            #    However, we can treat the change in this 2D projection as a proxy for yaw
            #    when combined with state.

            # BETTER APPROACH: "Virtual Z"
            # We assume shoulder width is constant (let's say 1.0 unit).
            # The observed width 'w' is cos(yaw).
            # So yaw = acos(w).
            # This gives us absolute rotation (0-90), but not direction or quadrant.
            # We use the accumulated_yaw to know which quadrant we are in.

            # ... But that is complex. Let's stick to the trusted method:
            # Just track the 2D Vector Delta, but filter out the "Flip".

            # --- DANGER ZONE LOGIC (Coasting) ---
            if shoulder_width_2d < CROSSOVER_THRESHOLD:
                # Unstable profile view. Vector math is garbage here.
                # Use MOMENTUM instead of math.
                if abs(self.last_valid_delta) > 0.1:
                    # Apply decaying momentum to carry through the turn
                    coast_delta = self.last_valid_delta * 0.98
                    self.accumulated_yaw -= coast_delta
                    self.last_valid_delta = coast_delta

            # --- SAFE ZONE LOGIC (Normal Tracking) ---
            else:
                prev_vector = r_prev - l_prev
                current_vector = r_curr - l_curr

                # Cross Product for signed rotation
                cross_prod = prev_vector[0] * current_vector[1] - prev_vector[1] * current_vector[0]
                dot_prod = np.dot(prev_vector, current_vector)

                delta_deg = math.degrees(math.atan2(cross_prod, dot_prod))

                # Sanity Filter
                if abs(delta_deg) < 30.0:
                    self.accumulated_yaw -= delta_deg
                    self.last_valid_delta = delta_deg  # Save momentum

        self.prev_shoulders = (l_curr, r_curr)
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
