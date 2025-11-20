import os
from typing import Any

import cv2
import stag
import numpy as np
from cv2 import Mat
from numpy import ndarray, dtype


# move STag tracker logic into a class
class FiducialMarker(object):
    def __init__(self, marker_size_cm, calibration_file="calibration_data.npz"):
        self.libraryHD = 21 # STag library HD21
        self.marker_size = marker_size_cm # Physical size of the printed marker
        self.calibration_file = calibration_file

        # Load calibration if it exists, otherwise, we are uncalibrated
        if os.path.exists(self.calibration_file):
            self.load_calibration()
            self.is_calibrated = True
        else:
            self.camera_matrix = None
            self.distortion_coefficients = None
            self.is_calibrated = False

    def load_calibration(self):
        # Extra redundancy for safety
        if not os.path.exists(self.calibration_file):
            print("calibration file does not exist")
            return

        with np.load(self.calibration_file) as data:
            self.camera_matrix = data['camera_matrix']
            self.distortion_coefficients = data['distortion_coefficients']

    def run_checkerboard_calibration(self, frames) -> None:
        """
        Input: A list of image frames containing a checkerboard
        Action: Calculates camera matrix (camera_matrix) and distortion coefficients (distortion_coefficients)
        Output: Saves data to self.calibration_file
        :param frames:
        :return:
        """
        # Logic for cv2.findChessboardCorners goes here
        object_points = []
        image_points = []
        image_size = 0
        frame_dimensions = None

        # Hardcoded dimensions for the inner corners of the checkerboard
        rows = 6
        columns = 9

        # Prepare the object points: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        #   Initialize a matrix of zeros with shape (Total Points, 3 coordinates)
        object_point_structure = np.zeros((rows * columns, 3), np.float32)

        # Fill the X and Y columns using mgrid, keeping Z as zero
        object_point_structure[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

        for frame in frames:
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_dimensions = grey_frame.shape[1], grey_frame.shape[0]

            found, corners = cv2.findChessboardCorners(grey_frame, (columns, rows), None)

            if found:
                object_points.append(object_point_structure)
                image_points.append(corners)

        ret, self.camera_matrix, self.distortion_coefficients, rvectors, tvectors = cv2.calibrateCamera(
            object_points, # list of 3D points ("Answer Key")
            image_points, # list of 2D points (What the camera saw)
            frame_dimensions,
            cameraMatrix = None,
            distCoeffs = None
        )

        np.savez(self.calibration_file, camera_matrix = self.camera_matrix, distortion_coefficients = self.distortion_coefficients)
        self.is_calibrated = True

    def detect_marker_pose(self, frame) -> tuple[bool, ndarray | Any, ndarray | Any, Any] | tuple[
        bool, None, None, None]:
        """
        Input: Current video frame
        Action: Detects STag, uses self.matrix to solve PnP
        Output: Distance (Tz) and Rotation (Rx, Ry, Rz)
        :param frame:
        :return:
        """
        # In OpenCV, y-axis is DOWN (POSITIVE y-values are DOWN, NEGATIVE y-values are UP)
        #   Expected point layout is clockwise: [top_left, top_right, bottom_right, bottom_left]
        top_left_point = (0, 0, 0)
        top_right_point = (self.marker_size, 0, 0)
        bottom_right_point = (self.marker_size, self.marker_size, 0)
        bottom_left_point = (0, self.marker_size, 0)

        marker_object_points = np.array(
            [top_left_point, top_right_point, bottom_right_point, bottom_left_point],
            dtype=np.float32)

        frame_corners, frame_ids, _ = stag.detectMarkers(image=frame, libraryHD=self.libraryHD)

        if len(frame_ids) > 0:
            success, rvec, tvec = cv2.solvePnP(marker_object_points, frame_corners[0], self.camera_matrix, self.distortion_coefficients)
            return success, rvec, tvec, frame_ids[0]

        return False, None, None, None