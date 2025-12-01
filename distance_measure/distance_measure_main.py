from fiducial_markers import FiducialMarker
from facial_tracker import FacialTracker
from body_tracker import BodyTracker

# Integrate HPE by @MickMcB with edits by @swholtzman
from head_pose_estimation.pose_estimator import HeadPoseEstimator

import math
import cv2  # cv2 must be imported AFTER fiducial_markers/stag
import time
import cvzone
import numpy as np
import sys

# --- CONFIGURATION ---
# Measure the black edge of the square marker in CM.
# If this is wrong, the distance (Y) will be wrong.
MARKER_SIZE_CM = 10.0

# Smoothing factor (0.1 = slow/smooth, 0.9 = fast/jittery)
ALPHA = 0.6


class Main(object):
    def __init__(self):
        # Initialize FaceMesh with maxFaces=1 for cleaner tracking
        self.face_tracker = FacialTracker()
        self.body_tracker = BodyTracker()
        self.markers = FiducialMarker(marker_size_cm=MARKER_SIZE_CM)

        # Integrate HPE by @MickMcB with edits by @swholtzman
        self.head_pose = HeadPoseEstimator()

        self.camera = cv2.VideoCapture(0)

        # Optimize Camera Buffer (reduces latency)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Smoothing Filter State
        self.smoothed_y = 0  # Depth (Y)
        self.smoothed_x = 0  # Lateral (X)

        # Output Throttling State
        self.last_sent_x = -999.0
        self.last_sent_y = -999.0

    def calibrate_lens(self):
        """
        Automated "Hold Still" calibration process.

        :return:
        """
        calibration_frames = []
        REQUIRED_FRAMES = 15
        HOLD_TIME = 3.0  # Seconds to hold still
        MOVEMENT_THRESHOLD = 3.0  # Max pixel movement allowed to be considered "still"

        print("Entering Lens Calibration Mode...", file=sys.stderr)

        # State variables for the automation
        last_corners = None
        stability_start_time = None

        while not self.markers.is_calibrated:
            success, frame = self.camera.read()
            if not success:
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Look for Checkerboard
            found, corners = cv2.findChessboardCorners(
                gray,
                (self.markers.CHECKERBOARD_COLS, self.markers.CHECKERBOARD_ROWS),
                None
            )

            instruction_text = "Searching for Checkerboard..."
            box_color = (0, 0, 255)  # Red

            if found:
                # Draw the corners so user knows it's detected
                cv2.drawChessboardCorners(
                    display_frame,
                    (self.markers.CHECKERBOARD_COLS, self.markers.CHECKERBOARD_ROWS),
                    corners,
                    found
                )

                # 2. Check Stability
                is_stable = False
                if last_corners is not None:
                    # Calculate how much the board moved since last frame (Euclidean distance of corner 0)
                    movement = np.linalg.norm(corners[0] - last_corners[0])

                    if movement < MOVEMENT_THRESHOLD:
                        is_stable = True
                    else:
                        is_stable = False
                        stability_start_time = None

                last_corners = corners

                # 3. Handle Timer
                if is_stable:
                    if stability_start_time is None:
                        stability_start_time = time.time()

                    elapsed = time.time() - stability_start_time
                    countdown = math.ceil(HOLD_TIME - elapsed)

                    if elapsed >= HOLD_TIME:
                        # --- CAPTURE! ---
                        calibration_frames.append(frame)
                        print(f"Captured frame {len(calibration_frames)}/{REQUIRED_FRAMES}", file=sys.stderr)

                        # Visual Flash effect
                        cv2.rectangle(
                            display_frame,
                            (0, 0),
                            (frame.shape[1],
                             frame.shape[0]),
                            (255, 255, 255),
                            cv2.FILLED
                        )

                        # Force a pause so they have to move it to a new angle
                        stability_start_time = None
                        last_corners = None
                        time.sleep(0.5)
                    else:
                        # --- Countdown UI ---
                        instruction_text = f"HOLD STILL: {countdown}"
                        box_color = (0, 255, 255)

                        # 1. Get the frame dimensions
                        height, width, _ = display_frame.shape

                        # Define positions relative to screen size
                        center_x = width // 2
                        center_y = height - 100

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        font_thickness = 2
                        text_color = (0, 255, 255)

                        # Draw Timer Circle
                        cv2.circle(display_frame, (center_x, center_y), 30, text_color, 2)
                        cv2.circle(display_frame, (center_x, center_y), int(30 * (elapsed / HOLD_TIME)), text_color, -1)

                        # Draw Text Centered
                        (text_w, text_h), _ = cv2.getTextSize(instruction_text, font, font_scale, font_thickness)
                        text_x = center_x - (text_w // 2)
                        text_y = center_y + 50

                        cv2.putText(display_frame, instruction_text, (text_x, text_y),
                                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                else:
                    instruction_text = "Stabilizing..."

            # 4. Check Completion
            if len(calibration_frames) >= REQUIRED_FRAMES:
                cvzone.putTextRect(display_frame, "CALCULATING...", (50, 200), scale=3, thickness=3)
                cv2.imshow('Facial Tracker', display_frame)
                cv2.waitKey(100)
                self.markers.run_checkerboard_calibration(calibration_frames)
                break

            # UI Overlay
            cvzone.putTextRect(display_frame, f"Calibration: {len(calibration_frames)}/{REQUIRED_FRAMES}", (50, 50),
                               scale=2, thickness=2)
            cvzone.putTextRect(display_frame, instruction_text, (50, 100), scale=2, thickness=2, colorR=box_color)

            # Helper text
            if len(calibration_frames) > 0 and not found:
                cvzone.putTextRect(display_frame, "Move board to new angle", (50, 150), scale=1, thickness=1)

            cv2.imshow('Facial Tracker', display_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("Skipping Lens Calibration", file=sys.stderr)
                break

        print("Lens Calibration Complete", file=sys.stderr)

    def run(self):
        if not self.markers.is_calibrated:
            self.calibrate_lens()

        print("System Ready. Streaming data format: x,y,yaw (meters)", flush=True, file=sys.stderr)

        while True:
            success, frame = self.camera.read()
            if not success:
                break

            # 1. --- RUN HEAD POSE ESTIMATION FIRST ---
            # Create a SEPARATE copy for drawing.
            # We MUST keep 'frame' clean for the STag marker detector.
            # If we draw axes on 'frame' before STag sees it, STag will crash (SIGSEGV).
            display_frame = frame.copy()

            height, width, _ = frame.shape
            frame_center_x = width // 2

            # --- 2. RUN HEAD POSE ESTIMATION ---
            # Pass 'display_frame' so the axes are drawn on the visual copy, NOT the raw data.
            display_frame, yaw_deg, pitch_deg, roll_deg = self.head_pose.process_frame(display_frame)

            # Convert Yaw to rotation units (0.0 - 1.0)
            yaw_normalized = yaw_deg / 360.0

            # --- 3. GATHER RAW DATA ---
            # Can use display_frame for body/face as they are robust to drawings,
            #   but using 'frame' is safer if wanting pure data.
            # Visuals will be drawn on display_frame.

            # Body Data: Get Width AND Height
            pose_landmarks = self.body_tracker.detect_body(display_frame, draw=True)
            body_w_px, body_h_px = self.body_tracker.get_body_dimensions(pose_landmarks, width, height)

            # Face Data: Get WIDTH and HEIGHT
            _, faces = self.face_tracker.detect_face(display_frame)  # Don't redraw mesh if HPE already did (optional)
            face_w_px, face_h_px = 0, 0
            face_center_x = frame_center_x
            if faces:
                face_w_px, face_h_px = self.face_tracker.get_face_dimensions(faces[0])
                face_center_x = (faces[0][145][0] + faces[0][374][0]) // 2

            # --- 4. MARKER DETECTION (ON CLEAN FRAME) ---
            # Pass the clean 'frame' here. It has NO lines/axes drawn on it.
            marker_results = self.markers.detect_marker_pose(frame)

            # --- 5. DETERMINE LOCATION ---
            # Default to Last Known Location (prevents snapping to 0)
            target_z_cm = self.smoothed_y
            target_x_cm = self.smoothed_x

            source_label = "LOST (Holding)"
            color = (100, 100, 100)  # Gray

            # PRIORITY 1: Marker (The "Teacher")
            if marker_results:
                first_id = list(marker_results.keys())[0]
                _, tvec, corners = marker_results[first_id]

                # Accurate Marker Z/X
                target_x_cm = tvec[0][0]
                target_z_cm = tvec[2][0]

                source_label = "MARKER"
                color = (0, 255, 0)

                # TRAIN STUDENTS using the valid Marker Z
                self.face_tracker.train(target_z_cm, face_w_px, face_h_px)
                self.body_tracker.train(target_z_cm, body_w_px, body_h_px)

                # Visuals
                pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(display_frame, [pts], True, (255, 0, 255), 2)

            # PRIORITY 2: Face (Student 1)
            elif faces and self.face_tracker.focal_ratio_width:
                # Use Multi-Axis distance (filters rotation)
                target_z_cm = self.face_tracker.get_distance(face_w_px, face_h_px)

                # Approximate X using horizontal offset
                # Need approximate scale (cm per pixel) at this depth
                # Scale ~ Distance / FocalRatio. Using Width Ratio for X calculations
                if face_w_px > 0:
                    scale = target_z_cm / self.face_tracker.focal_ratio_width
                    target_x_cm = (face_center_x - frame_center_x) * scale * self.face_tracker.real_width_cm  # Rough approx

                source_label = "FACE"
                color = (0, 255, 255)  # Yellow

            # PRIORITY 3: Body (Student 2)
            elif pose_landmarks and self.body_tracker.focal_ratio_width:
                target_z_cm = self.body_tracker.get_distance(body_w_px, body_h_px)

                # Approximate X
                bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)
                if bbox:
                    body_center = (bbox[0] + bbox[2]) // 2
                    if body_w_px > 0:
                        scale = target_z_cm / self.body_tracker.focal_ratio_width
                        target_x_cm = (body_center - frame_center_x) * scale * 40.0  # fallback scaling

                source_label = "BODY"
                color = (255, 0, 0)

            # --- 6. SMOOTHING ---
            # Apply alpha filter to remove jitter
            self.smoothed_y = (ALPHA * target_z_cm) + ((1 - ALPHA) * self.smoothed_y)
            self.smoothed_x = (ALPHA * target_x_cm) + ((1 - ALPHA) * self.smoothed_x)

            # --- 7. OUTPUT ---
            out_x_m = self.smoothed_x / 100.0
            out_y_m = self.smoothed_y / 100.0

            delta = abs(out_x_m - self.last_sent_x) + abs(out_y_m - self.last_sent_y)

            if delta > 0.01:
                print(f"{out_x_m:.4f},{out_y_m:.4f},{yaw_normalized:.4f}", flush=True)
                self.last_sent_x = out_x_m
                self.last_sent_y = out_y_m

            # --- 8. UI DRAWING ---
            # We always draw the "Last Known" or "Current" location
            info_text = f"Y:{int(self.smoothed_y)}cm X:{int(self.smoothed_x)}cm Yaw:{int(yaw_deg)}"

            # Attach label to whatever is visible, or top left if Lost
            label_pos = (50, 50)

            if pose_landmarks:
                bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)
                if bbox:
                    # Drawing body box on top of axes
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    label_pos = (bbox[0], bbox[1] - 40)

            cvzone.putTextRect(
                display_frame,
                info_text, label_pos,
                scale=2,
                thickness=2,
                colorR=color,

                colorT=(0, 0, 0)
            )

            cvzone.putTextRect(
                display_frame,
                source_label,
                (label_pos[0], label_pos[1] + 30),
                scale=1,
                thickness=1,
                colorR=color, colorT=(0, 0, 0)
            )

            cv2.imshow('Facial Tracker', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main()
    main.run()