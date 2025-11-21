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

        # cv2.destroyWindow('Facial Tracker')
        print("Calibration complete")


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

            center_pixel = (int(width / 2), int(height / 2))
            top_left = (int(center_pixel[0] - 100), int(center_pixel[1] - 100))
            bottom_right = (int(center_pixel[0] + 100), int(center_pixel[1] + 100))

            # Draw the Green Square (The Target)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

            # UI Header
            cvzone.putTextRect(frame, "DISTANCE ZEROING MODE", (50, 50), scale=2, thickness=2)
            cvzone.putTextRect(frame, "Align STag in Green Square", (50, 100), scale=1, thickness=1)

            # Marker Detection
            marker_results = self.markers.detect_marker_pose(frame)
            if marker_results:
                for rvec, tvec, corners in marker_results.values():
                    # Draw the detected marker for visual confirmation
                    #   corners comes as shape (1, 4, 2), need (4, 2)
                    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

                    middle_point = np.mean(corners, axis=0).flatten() # ensure a flat array

                    # Calculate pixel distance from center of screen to center of marker
                    difference = middle_point - np.array(center_pixel)
                    euclidean_distance = np.linalg.norm(difference)

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

                            print("Alignment Successful")
                            return  # Exit function

                        else:
                            count_down = 3 - int(seconds_elapsed)
                            cvzone.putTextRect(frame, f"Hold: {count_down}",
                                               (center_pixel[0] - 50, center_pixel[1] - 20), scale=2,
                                               colorR=(0, 0, 255))

                    else:
                        alignment_start_time = None

            cv2.imshow('Facial Tracker', frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('s'):
                print(f"Distance Set Manually")
                break
            elif key_pressed == ord('q'):
                print(f"Calibration aborted by user")
                break

        print("Distance Calibration Complete!")

    def run(self):
        if not self.markers.is_calibrated:
            self.calibrate_lens()

        self.calibrate_distance()

        while True:
            success, frame = self.camera.read()
            if not success:
                print("Failed to read from camera")
                break

            results = self.markers.detect_marker_pose(frame)

            cvzone.putTextRect(frame, "MAIN TRACKING", (50, 50), scale=2, thickness=2)

            if results:
                for rvec, tvec, corners in results.values():
                    # tvec is usually [[x], [y], [z]]
                    dist_cm = tvec[2][0]

                    # Draw marker box
                    points = corners.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)

                    cvzone.putTextRect(
                        img=frame,
                        text=f"{int(dist_cm)}cm",
                        pos=tuple(corners[0][0].astype(int)),
                        scale=2,
                        thickness=2,
                        colorR=(0, 0, 0),
                        colorT=(0, 255, 50)
                    )

            cv2.imshow('Facial Tracker', frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                print(f"Process aborted by user")
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main = Main()
    main.run()
