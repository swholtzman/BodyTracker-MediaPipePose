import os
from typing import Any

import cv2
import numpy as np
import stag


if not hasattr(stag, "detectMarkers"):
    raise ImportError(
        "Incorrect 'stag' library detected. "
        "Please run 'pip uninstall stag' and 'pip install stag-python'."
    )


class FiducialMarker(object):
    def __init__(self, marker_size_cm, calibration_file="calibration_data.npz"):
        """

        :param marker_size_cm:
        :param calibration_file:
        """
        self.libraryHD = 23  # STag library HD23

        # --- Expose Checkerboard Dimensions for Main.py ---
        self.CHECKERBOARD_ROWS = 6
        self.CHECKERBOARD_COLS = 9

        self.marker_size = marker_size_cm
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
        """

        :return:
        """
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
        # Safety check: If no frames, we cannot calibrate.
        # This fixes the "frame_dimensions might be None" warning.
        if not frames:
            print("Calibration failed: No frames provided.")
            return

        # Logic for cv2.findChessboardCorners goes here
        object_points = []
        image_points = []

        # Initialize with a default to silence PyCharm, though the loop below overwrites it
        frame_dimensions = (frames[0].shape[1], frames[0].shape[0])

        # Prepare the object points: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
        #   Initialize a matrix of zeros with shape (Total Points, 3 coordinates)
        object_point_structure = np.zeros((self.CHECKERBOARD_ROWS * self.CHECKERBOARD_COLS, 3), np.float32)

        # Fill the X and Y columns using mgrid, keeping Z as zero
        object_point_structure[:, :2] = np.mgrid[0:self.CHECKERBOARD_COLS, 0:self.CHECKERBOARD_ROWS].T.reshape(-1, 2)

        for frame in frames:
            # Check if frame is valid
            if frame is None:
                continue

            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_dimensions = grey_frame.shape[1], grey_frame.shape[0]

            found, corners = cv2.findChessboardCorners(
                grey_frame,
                (self.CHECKERBOARD_COLS, self.CHECKERBOARD_ROWS),
                None
            )

            if found:
                object_points.append(object_point_structure)
                image_points.append(corners)

        # [FIX 2] Ensure we actually found corners before running the math
        if not object_points:
            print("Calibration failed: Checkerboard not detected in any frame.")
            return

        ret, self.camera_matrix, self.distortion_coefficients, rvectors, tvectors = cv2.calibrateCamera(
            object_points, # list of 3D points ("Answer Key")
            image_points, # list of 2D points (What the camera saw)
            frame_dimensions,
            cameraMatrix = None,
            distCoeffs = None
        )

        np.savez(self.calibration_file,
                 camera_matrix = self.camera_matrix,
                 distortion_coefficients = self.distortion_coefficients
                 )

        self.is_calibrated = True
        print("Calibration saved to", self.calibration_file)

    def detect_marker_pose(self, frame) -> dict[Any, Any] | None:
        """
        Input: Current video frame
        Action: Detects STag, uses self.matrix to solve PnP
        Output: Distance (Tz) and Rotation (Rx, Ry, Rz)
        :param frame:
        :return:
        """
        frame_dict = {}

        # 1. Convert to Grayscale (STag requires 1 channel)
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        # 2. Force Contiguous Memory (C++ requires strict memory layout)
        # This prevents the SIGSEGV (Exit Code 139)
        gray_frame = np.ascontiguousarray(gray_frame)

        # 3. Define the 3D world coordinates of the marker corners (Clockwise from Top-Left)
        # ArUco default corner order: TL, TR, BR, BL
        top_left_point = (0, 0, 0)
        top_right_point = (self.marker_size, 0, 0)
        bottom_right_point = (self.marker_size, self.marker_size, 0)
        bottom_left_point = (0, self.marker_size, 0)

        marker_object_points = np.array(
            [top_left_point, top_right_point, bottom_right_point, bottom_left_point],
            dtype=np.float32
        )

        # 4. Detect Markers using STag
        try:
            corners, ids, rejected = stag.detectMarkers(image=gray_frame, libraryHD=self.libraryHD)
        except Exception as e:
            print(f"STag Error: {e}")
            return frame_dict

        # 5. If markers found, solve PnP for each
        if ids is not None and len(ids) > 0:
            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])  # Extract ID from array

                # current_corners comes as shape (1, 4, 2) or (4, 2) depending on version
                current_corners = np.array(corners[i], dtype=np.float32)

                # Ensure shape is correct for solvePnP (needs to be list of points)
                if current_corners.shape == (1, 4, 2):
                    current_corners = current_corners.reshape(4, 2)

                # Solve Perspective-n-Point to find pose
                success, rvec, tvec = cv2.solvePnP(
                    marker_object_points,
                    current_corners,
                    self.camera_matrix,
                    self.distortion_coefficients
                )

                if success:
                    # Store results.
                    # Reshape corners to (1, 4, 2) to match the format Main.py expects for polylines
                    reshaped_corners = current_corners.reshape(1, 4, 2)
                    frame_dict[marker_id] = rvec, tvec, reshaped_corners

        return frame_dict
