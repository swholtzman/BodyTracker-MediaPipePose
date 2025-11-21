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

        # Standard average human shoulder width in cm
        # self.AVG_SHOULDER_WIDTH_CM = 45.0

        # Sam's shoulder width (for testing)
        self.AVG_SHOULDER_WIDTH_CM = 50

    def calibrate_lens(self):
        """

        :return:
        """
        calibration_frames = []
        print("Entering Lens Calibration Mode...")

        while not self.markers.is_calibrated:
            success, frame = self.camera.read()

            if not success:
                print("Failed to read from camera")
                break

            # UI: Tell the user what to do
            cvzone.putTextRect(frame, f"LENS CALIBRATION MODE", (50, 50), scale=2, thickness=2)
            cvzone.putTextRect(frame, f"Frames Captured: {len(calibration_frames)}", (50, 100), scale=2,
                               thickness=2)
            cvzone.putTextRect(frame, "Press 'c' to capture, 'q' to compute, 'esc' to skip", (50, 150), scale=1,
                               thickness=1)

            cv2.imshow('Facial Tracker', frame)
            key_pressed = cv2.waitKey(1) & 0xFF

            if key_pressed == ord('c'):
                print(f"Captured frame for camera calibration")
                calibration_frames.append(frame)
            elif key_pressed == ord('q'):
                print(f"Initiating calibration with {len(calibration_frames)} frames")
                if calibration_frames:
                    self.markers.run_checkerboard_calibration(calibration_frames)
                else:
                    print("No calibration frames detected. Calibration aborted.")
                break
            elif key_pressed == 27:
                print("Skipping Lens Calibration (Distance tracking will be inaccurate)")
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

            cvzone.putTextRect(frame, "ALIGN ArUco MARKER", (50, 50), scale=2, thickness=2)

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

    def run(self):
        # 1. Lens Calibration (Get the focal length)
        if not self.markers.is_calibrated:
            self.calibrate_lens()

        # 2. Distance Calibration (Get the Focal Length)
        focal_length = self.calibrate_distance()
        print(f"Using Focal Length: {focal_length}")

        # 3. Main Body Tracking Loop
        while True:
            success, frame = self.camera.read()
            if not success:
                print("Failed to read from camera")
                break

            height, width, _ = frame.shape

            # --- BODY TRACKING ---
            # 1. Get Landmarks
            pose_landmarks = self.body_tracker.detect_body(frame)

            if pose_landmarks:
                # 2. Ask Tracker for Measurements
                shoulder_width_px = self.body_tracker.get_shoulder_width_pixels(pose_landmarks, width, height)
                bbox = self.body_tracker.get_bounding_box(pose_landmarks, width, height)

                # 3. Calculate Distance
                if shoulder_width_px > 0:
                    distance_cm = (self.AVG_SHOULDER_WIDTH_CM * focal_length) / shoulder_width_px
                else:
                    distance_cm = 0

                # 4. Draw UI
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

                    cvzone.putTextRect(
                        img=frame,
                        text=f"Body: {int(distance_cm)}cm",
                        pos=(x_min, y_min - 20),
                        scale=2,
                        thickness=2,
                        colorR=(0, 255, 0),
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
