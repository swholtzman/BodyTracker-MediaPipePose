import math

import cv2
import time
import cvzone
import numpy as np

from cvzone.FaceMeshModule import FaceMeshDetector
from body_tracker import BodyTracker
from fiducial_markers import FiducialMarker

class Main(object):
    def __init__(self):
        self.detector = FaceMeshDetector()
        self.body_tracker = BodyTracker()
        self.markers = FiducialMarker(marker_size_cm=10.0)
        self.camera = cv2.VideoCapture(0)

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


    def calibrate_distance(self):
        print("Entering Distance Calibration Mode...")
        alignment_start_time = None

        while True:
            success, frame = self.camera.read()
            if not success:
                print("Failed to read from camera")
                break

            # Calculate center dynamically (in case camera resizes or first frame was bad)
            height, width, _ = frame.shape
            center_pixel = np.array([width // 2, height // 2])

            # Draw the Green Square (The Target)
            cv2.rectangle(frame,
                          (int(center_pixel[0] - 100), int(center_pixel[1] - 100)),
                          (int(center_pixel[0] + 100), int(center_pixel[1] + 100)),
                          (0, 255, 0), 3)

            cvzone.putTextRect(frame, "ALIGN STag MARKER", (50, 50), scale=2, thickness=2)

            marker_results = self.markers.detect_marker_pose(frame)
            if marker_results:
                for rvec, tvec, corners in marker_results.values():

                    # 1. Draw the detected marker for visual confirmation
                    #   corners comes as shape (1, 4, 2), need (4, 2)
                    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

                    # 2. Check Alignment
                    # corners comes in shape (1, 4, 2)
                    # We take corners[0] to get the 4 points: (4, 2)
                    # We average those 4 points to get the center (x,y)
                    middle_point = np.mean(corners[0], axis=0)

                    # Calculate pixel distance from center of screen to center of marker
                    difference = middle_point - np.array(center_pixel)
                    euclidean_distance = np.linalg.norm(difference)

                    # 3. Calculate Focal Length Live
                    # Distance from Camera (tvec[2][0]) is in cm because marker_size was 10.0
                    distance_cm = tvec[2][0]

                    # Calculate pixel width of the marker (distance between top-left and top-right)
                    # corners[0][0] is TopLeft, corners[0][1] is TopRight
                    point_tl = corners[0][0]
                    point_tr = corners[0][1]
                    pixel_width = np.linalg.norm(point_tl - point_tr)

                    # F = (P * D) / W
                    current_focal_length = (pixel_width * distance_cm) / 10.0

                    if euclidean_distance < 50:
                        if alignment_start_time is None:
                            alignment_start_time = time.time()

                        seconds_elapsed = time.time() - alignment_start_time

                        if seconds_elapsed > 3:
                            cvzone.putTextRect(frame,
                                               "LOCKED!",
                                               (center_pixel[0], center_pixel[1]),
                                               colorR=(0, 255, 0)
                                               )

                            cv2.imshow('Facial Tracker', frame)
                            cv2.waitKey(500)  # Pause briefly to show success

                            print(f"Alignment Successful. Focal Length: {current_focal_length}")
                            return current_focal_length # Return the focal length

                        else:
                            count_down = 3 - int(seconds_elapsed)
                            cvzone.putTextRect(frame, f"Hold: {count_down}",
                                               (center_pixel[0] - 50, center_pixel[1] - 20), scale=2,
                                               colorR=(0, 0, 255))
                    else:
                        alignment_start_time = None

            cv2.imshow('Facial Tracker', frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q') or key_pressed == 27: return 800

        print("Distance Calibration Complete!")

    # def run(self):
    #     # 1. Lens Calibration (Get the focal length)
    #     if not self.markers.is_calibrated:
    #         self.calibrate_lens()
    #
    #     # 2. Distance Calibration (Get the Focal Length)
    #     focal_length = self.calibrate_distance()
    #     print(f"Using Focal Length: {focal_length}")
    #
    #     # 3. Main Body Tracking Loop
    #     while True:
    #         success, frame = self.camera.read()
    #         if not success:
    #             print("Failed to read from camera")
    #             break
    #
    #         height, width, _ = frame.shape
    #
    #         # --- BODY TRACKING ---
    #         # 1. Get Landmarks
    #         pose_landmarks = self.body_tracker.detect_body(frame)
    #
    #         if pose_landmarks:
    #             # 2. Ask Tracker for Measurements
    #             shoulder_width_px = self.body_tracker.get_shoulder_width_pixels(pose_landmarks, width, height)
    #             bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)
    #
    #             # 3. Calculate Distance
    #             if shoulder_width_px > 0:
    #                 distance_cm = (self.AVG_SHOULDER_WIDTH_CM * focal_length) / shoulder_width_px
    #             else:
    #                 distance_cm = 0
    #
    #             # 4. Draw UI
    #             if bbox:
    #                 x_min, y_min, x_max, y_max = bbox
    #                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    #
    #                 cvzone.putTextRect(
    #                     img=frame,
    #                     text=f"Body: {int(distance_cm)}cm",
    #                     pos=(x_min, y_min - 20),
    #                     scale=2,
    #                     thickness=2,
    #                     colorR=(0, 255, 0),
    #                     colorT=(0, 0, 0)
    #                 )
    #
    #         cv2.imshow('Facial Tracker', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     self.camera.release()
    #     cv2.destroyAllWindows()

    def run(self):
        # 1. Lens Calibration (Get the camera matrix/dist coefficients)
        if not self.markers.is_calibrated:
            self.calibrate_lens()

        print("Tracking started. Align STag marker on chest for distance.")

        # 3. Main Body Tracking Loop
        while True:
            success, frame = self.camera.read()
            if not success:
                print("Failed to read from camera")
                break

            height, width, _ = frame.shape
            distance_cm = 0
            marker_detected = False

            # --- MARKER TRACKING (Priority for Distance) ---
            marker_results = self.markers.detect_marker_pose(frame)
            if marker_results:
                # We assume the first detected marker is the one on the chest
                # In a multi-marker setup, you would filter by specific ID here
                first_marker_id = list(marker_results.keys())[0]
                rvec, tvec, corners = marker_results[first_marker_id]

                # tvec[2] is the Z-axis translation (Distance)
                distance_cm = tvec[2][0]
                marker_detected = True

                # Optional: Visual Debug for Marker
                pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

            # --- BODY TRACKING ---
            pose_landmarks = self.body_tracker.detect_body(frame)

            if pose_landmarks:
                bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)

                # 4. Draw UI
                if bbox:
                    x_min, y_min, x_max, y_max = bbox

                    # Color feedback: Green if marker locked, Yellow if Body only
                    box_color = (0, 255, 0) if marker_detected else (0, 255, 255)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)

                    display_text = f"Dist: {int(distance_cm)}cm" if marker_detected else "No Marker"

                    cvzone.putTextRect(
                        img=frame,
                        text=display_text,
                        pos=(x_min, y_min - 20),
                        scale=2,
                        thickness=2,
                        colorR=box_color,
                        colorT=(0, 0, 0)
                    )

            cv2.imshow('Facial Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main = Main()
    main.run()
