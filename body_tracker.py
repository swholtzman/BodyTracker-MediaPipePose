import cv2
import mediapipe as mp

# move body tracking logic into a class
class BodyTracker:
    def __init__(self):
        self.mp_pose =mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    # To make a "sticky" tracker, we will want a low threshold for the tracking (e.g., 0.5)
    #   The AI is "sticky." "This looks messy, but I'm pretty sure it's still the person, so I'll keep holding on."
        self.min_detection_confidence = 0.7 # higher to avoid false starts
        self.min_tracking_confidence = 0.5 # lower for a "sticky" tracker

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def detect_body(self, frame):
        """

        :param frame:
        :return:
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        pose_landmarks = results.pose_landmarks

        return pose_landmarks

# def main():
#     # 1. Initialize MediaPipe Pose classes
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles
#
#     # 2. Setup the Pose function
#     # min_detection_confidence: Confidence threshold for the first detection
#     # min_tracking_confidence: Confidence threshold for tracking continuously
#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
#     # 3. Open the Webcam (Try index 0 first; if that fails, try 1)
#     cap = cv2.VideoCapture(0)
#
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
#     print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
#
#     print("Press 'q' to exit the video window.")
#
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#             continue
#
#         # 4. Pre-process the image
#         # MediaPipe requires RGB input, but OpenCV gives us BGR.
#         image.flags.writeable = False
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # 5. Make the prediction
#         results = pose.process(image_rgb)
#
#         # 6. Draw the landmarks
#         image.flags.writeable = True
#         # Convert back to BGR for OpenCV rendering
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#
#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#             )
#
#         # 7. Display the resulting frame
#         cv2.imshow('MediaPipe Pose Test', image_bgr)
#
#         # Break loop on 'q' key press
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()