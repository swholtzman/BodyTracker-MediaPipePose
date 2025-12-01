import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

from helper_functions import get_distance

class FacialTracker:
    def __init__(self, max_faces=1):
        # 'refine_landmarks=True' improves eye/iris tracking stability
        self.detector = FaceMeshDetector(maxFaces=max_faces)

        # Store focal ratios for both axes to handle rotation
        self.focal_ratio_width = None
        self.focal_ratio_height = None

        # Fallbacks (overwritten by training)
        self.real_width_cm = 6.3
        self.real_height_cm = 18.0  # Approx avg face height

    def detect_face(self, frame, draw=True):
        """
        Returns the frame with visuals and the face data
        """
        return self.detector.findFaceMesh(frame, draw=draw)

    def get_face_dimensions(self, face):
        """
        Returns (width_px, height_px)
        Width: Distance between Left Eye (145) and Right Eye (374)
        Height: Distance between Forehead (10) and Chin (152) - stable during Yaw rotation
        """
        # Horizontal (Eyes)
        width, _ = self.detector.findDistance(face[145], face[374])

        # Vertical (Forehead to Chin)
        height, _ = self.detector.findDistance(face[10], face[152])

        return width, height

    # Legacy wrapper for compatibility if needed, but main loop will use get_face_dimensions
    def get_eye_width(self, face):
        """
        Returns pixel distance between left eye (145) and right eye (374)
        """
        pointLeft = face[145]
        pointRight = face[374]
        # findDistance returns (length, info_tuple)
        width, _ = self.detector.findDistance(pointLeft, pointRight)
        return width

    def train(self, known_distance_cm, current_width_px, current_height_px):
        """
        1. Learns Focal Ratio for Depth (Z)
        2. Learns Real Face Width for Lateral (X)
        """
        if current_width_px > 0:
            self.focal_ratio_width = known_distance_cm * current_width_px

        if current_height_px > 0:
            self.focal_ratio_height = known_distance_cm * current_height_px

    def get_distance(self, width_px, height_px):
        """

        :param width_px:
        :param height_px:
        :return:
        """
        return get_distance(self, width_px, height_px)