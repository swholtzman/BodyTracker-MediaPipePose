from __future__ import annotations

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

        # 1. Force a DEEP COPY to ensure we own this memory
        #    This prevents issues if 'frame' is a slice or view from another library
        safe_frame = frame.copy()

        # 2. Convert to Grayscale (STag requires 1 channel)
        if len(safe_frame.shape) == 3:
            gray_frame = cv2.cvtColor(safe_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = safe_frame

        # --- AVOID SIGSEGV CRASH ---
        # High-res images cause STag's internal buffers to overflow (JoinAnchorPoints crash).
        #   Resize the image to a safe resolution (e.g., max width 640 or 800)
        #       and then scale the resulting corner coordinates back up.
        height, width = gray_frame.shape
        MAX_WIDTH = 800  # Safe resolution for STag

        scale_factor = 1.0
        processing_frame = gray_frame

        if width > MAX_WIDTH:
            scale_factor = MAX_WIDTH / width
            new_height = int(height * scale_factor)
            processing_frame = cv2.resize(gray_frame, (MAX_WIDTH, new_height))

        # 3. Add Padding (The secondary crash fix)
        # --- SAFETY PADDING & MEMORY PACKING ---
        # The STag C++ library has a known bug where edge detection reads out of bounds.
        # We solve this by:
        # A) Adding a 16px black border so "out of bounds" is just valid black pixels.
        # B) Forcing 'ascontiguousarray' to ensure no memory gaps exist.
        padding = 16
        padded_frame = cv2.copyMakeBorder(
            processing_frame,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # 4. Force contiguous memory layout (The Anti-Crash Shield)
        #    We do this on the PADDED frame immediately before passing to C++
        contiguous_padded_frame = np.ascontiguousarray(padded_frame)

        # 5. Define the 3D world coordinates of the marker corners (Clockwise from Top-Left)
        #   STag default corner order: TL, TR, BR, BL
        top_left_point = (0, 0, 0)
        top_right_point = (self.marker_size, 0, 0)
        bottom_right_point = (self.marker_size, self.marker_size, 0)
        bottom_left_point = (0, self.marker_size, 0)

        marker_object_points = np.array(
            [top_left_point, top_right_point, bottom_right_point, bottom_left_point],
            dtype=np.float32
        )

        # 6. Detect Markers using STag
        try:
            # PASS THE CONTIGUOUS PADDED FRAME
            corners, ids, rejected = stag.detectMarkers(
                image=contiguous_padded_frame,
                libraryHD=self.libraryHD
            )
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

                # --- REMOVE PADDING OFFSET ---
                # 1. Remove Padding
                #   Since we detected on a padded image, coordinates are shifted by +10.
                #   We subtract 10 to map back to the original image space.
                current_corners[:, 0] -= padding  # X
                current_corners[:, 1] -= padding  # Y

                # 2. Reverse Scaling (Map back to original high-res coordinates)
                current_corners[:, 0] /= scale_factor
                current_corners[:, 1] /= scale_factor

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
