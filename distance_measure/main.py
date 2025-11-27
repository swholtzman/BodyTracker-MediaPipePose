
from fiducial_markers import FiducialMarker
from facial_tracker import FacialTracker
from body_tracker import BodyTracker

import math
import cv2  # cv2 must be imported AFTER fiducial_markers/stag
import time
import cvzone
import numpy as np

from cvzone.FaceMeshModule import FaceMeshDetector
from body_tracker import BodyTracker

class Main(object):
    def __init__(self):
        # Initialize FaceMesh with maxFaces=1 for cleaner tracking
        self.face_tracker = FacialTracker()
        self.body_tracker = BodyTracker()
        self.markers = FiducialMarker(marker_size_cm=10.0)

        self.camera = cv2.VideoCapture(0)

        # Smoothing Filter for Distance
        self.smoothed_distance = 0
        self.ALPHA = 0.6  # Smoothing factor (0.1 = slow/smooth, 0.9 = fast/jittery)

    def calibrate_lens(self):
        """
        Automated "Hold Still" calibration process.

        :return:
        """
        calibration_frames = []
        REQUIRED_FRAMES = 15
        HOLD_TIME = 3.0  # Seconds to hold still
        MOVEMENT_THRESHOLD = 3.0  # Max pixel movement allowed to be considered "still"

        print("Entering Lens Calibration Mode...")

        # State variables for the automation
        last_corners = None
        stability_start_time = None
        last_capture_time = 0

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
            box_color = (0, 0, 255)  # Red by default

            if found:
                # Draw the corners so user knows it's detected
                cv2.drawChessboardCorners(display_frame,
                                          (self.markers.CHECKERBOARD_COLS, self.markers.CHECKERBOARD_ROWS),
                                          corners, found)

                # 2. Check Stability
                is_stable = False
                if last_corners is not None:
                    # Calculate how much the board moved since last frame (Euclidean distance of corner 0)
                    movement = np.linalg.norm(corners[0] - last_corners[0])

                    if movement < MOVEMENT_THRESHOLD:
                        is_stable = True
                    else:
                        is_stable = False
                        stability_start_time = None  # Reset timer if moved

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
                        print(f"Captured frame {len(calibration_frames)}/{REQUIRED_FRAMES}")

                        # Visual Flash effect
                        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255),
                                      cv2.FILLED)

                        # Force a pause so they have to move it to a new angle
                        stability_start_time = None
                        last_corners = None  # Reset stability check
                        time.sleep(0.5)
                    else:
                        # Countdown UI
                        instruction_text = f"HOLD STILL: {countdown}"
                        box_color = (0, 255, 255)  # Yellow

                        # Draw a progress bar or shrinking circle
                        cv2.circle(display_frame, (50, 50), 30, (0, 255, 255), 2)
                        cv2.circle(display_frame, (50, 50), int(30 * (elapsed / HOLD_TIME)), (0, 255, 255), -1)

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
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27:  # ESC
                print("Skipping Lens Calibration")
                break

        print("Lens Calibration Complete")

    def run(self):
        if not self.markers.is_calibrated:
            self.calibrate_lens()

        print("Tracking started.")

        while True:
            success, frame = self.camera.read()
            if not success:
                break

            # Use a writable copy for drawing
            display_frame = frame.copy()
            height, width, _ = frame.shape

            # --- DATA GATHERING ---
            # 1. Body Data (Always run this, we need the box and shoulders)
            pose_landmarks = self.body_tracker.detect_body(display_frame, draw=True)
            shoulder_width = self.body_tracker.get_shoulder_width_pixels(pose_landmarks, width, height)

            # 2. Face Data
            display_frame, faces = self.face_tracker.detect_face(display_frame)
            face_width = 0
            if faces:
                face_width = self.face_tracker.get_eye_width(faces[0])

            # 3. Marker Data
            marker_results = self.markers.detect_marker_pose(frame)

            # --- DECISION LOGIC ---
            raw_distance = 0
            source_label = "SEARCHING"
            color = (0, 0, 255)  # Red

            # CASE A: Marker Detected (The Teacher)
            if marker_results:
                first_id = list(marker_results.keys())[0]
                _, tvec, corners = marker_results[first_id]

                raw_distance = tvec[2][0]
                source_label = "MARKER (Accurate)"
                color = (0, 255, 0)  # Green

                # TRAIN STUDENTS
                self.face_tracker.train(raw_distance, face_width)
                self.body_tracker.train(raw_distance, shoulder_width)

                # Draw Marker
                pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(display_frame, [pts], True, (255, 0, 255), 2)

            # CASE B: Face Detected (Student 1)
            elif faces and self.face_tracker.focal_ratio:
                raw_distance = self.face_tracker.get_distance(face_width)
                source_label = "FACE (Robust)"
                color = (0, 255, 255)  # Yellow

                # Chain Training: Face can keep Body updated if Marker is lost
                self.body_tracker.train(raw_distance, shoulder_width)

            # CASE C: Body Only (Student 2 - Turning Around)
            elif pose_landmarks and self.body_tracker.focal_ratio:
                raw_distance = self.body_tracker.get_distance(shoulder_width)
                source_label = "BODY (Fallback)"
                color = (255, 0, 0)  # Blue

            # --- SMOOTHING ---
            # Prevents jitter when switching between modes
            if raw_distance > 0:
                if self.smoothed_distance == 0:
                    self.smoothed_distance = raw_distance
                else:
                    self.smoothed_distance = (self.ALPHA * raw_distance) + ((1 - self.ALPHA) * self.smoothed_distance)

            # --- UI DRAWING ---
            if pose_landmarks:
                bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), color, 2)

                    # Info Tag
                    cvzone.putTextRect(display_frame, f"{int(self.smoothed_distance)}cm",
                                       (x_min, y_min - 40), scale=2, thickness=2, colorR=color, colorT=(0, 0, 0))
                    cvzone.putTextRect(display_frame, source_label,
                                       (x_min, y_min - 10), scale=1, thickness=1, colorR=color, colorT=(0, 0, 0))

            cv2.imshow('Facial Tracker', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main()
    main.run()
