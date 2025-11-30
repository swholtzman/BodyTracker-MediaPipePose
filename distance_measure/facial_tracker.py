import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

class FacialTracker:
    def __init__(self, max_faces=1):
        # 'refine_landmarks=True' improves eye/iris tracking stability
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        self.focal_ratio = None # The "learned" constant

        # Default fallback, but will be overwritten by training
        self.real_width_cm = 6.3

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

    def train(self, known_distance_cm, current_pixel_width, marker_px_width, marker_real_width_cm):
        """
        1. Learns Focal Ratio for Depth (Z)
        2. Learns Real Face Width for Lateral (X)
        """
        if current_pixel_width > 0 and marker_px_width > 0:
            # Calibrate Depth (Z)
            self.focal_ratio = known_distance_cm * current_pixel_width

            # Calibrate Size (X) - The "Same Distance" Logic
            # Ratio: Face_Px / Marker_Px = Face_Real / Marker_Real
            ratio = current_pixel_width / marker_px_width
            self.real_width_cm = ratio * marker_real_width_cm

    def get_distance(self, current_pixel_width):
        """
        Calculates distance using the learned ratio.
        Distance = Ratio / PixelWidth
        """
        if self.focal_ratio is None or current_pixel_width == 0:
            return 0
        return self.focal_ratio / current_pixel_width