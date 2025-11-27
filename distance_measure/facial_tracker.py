import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

class FacialTracker:
    def __init__(self, max_faces=1):
        # 'refine_landmarks=True' improves eye/iris tracking stability
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        self.focal_ratio = None # The "learned" constant

    def detect_face(self, frame, draw=True):
        """
        Returns the frame with visuals and the face data
        """
        return self.detector.findFaceMesh(frame, draw=draw)

    def get_eye_width(self, face):
        """
        Returns pixel distance between left eye (145) and right eye (374)
        """
        pointLeft = face[145]
        pointRight = face[374]
        # findDistance returns (length, info_tuple)
        width, _ = self.detector.findDistance(pointLeft, pointRight)
        return width

    def train(self, known_distance_cm, current_pixel_width):
        """
        Learns the focal ratio based on ground truth (Marker)
        Ratio = Distance * PixelWidth
        """
        if current_pixel_width > 0:
            self.focal_ratio = known_distance_cm * current_pixel_width

    def get_distance(self, current_pixel_width):
        """
        Calculates distance using the learned ratio.
        Distance = Ratio / PixelWidth
        """
        if self.focal_ratio is None or current_pixel_width == 0:
            return 0
        return self.focal_ratio / current_pixel_width