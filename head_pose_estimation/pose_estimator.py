# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:09:02 2025

@author: Mahdi Ghafourian
"""

import math
import cv2
import numpy as np
import torch
import mediapipe as mp
from math import cos, sin
import os

# Relative import because we added this folder to sys.path in runner.py
from .helpers import FeatureExtractor as FE

class HeadPoseEstimator:
    def __init__(self, model_filename="combined_model_scripted.pth"):
        # 1. Device Selection (Added MPS for Mac M-series)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"HPE running on: {self.device}")

        # 2. Robust Path Handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", model_filename)

        # 3. Load Model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 4. Settings
        self.alpha = 0.75
        self.prev_tdx, self.prev_tdy = None, None
        self.MAX_CENTER_JUMP = 200

        # Initialize FaceMesh (Refine landmarks for better iris/eye data)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # Set False for video stream optimization
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6, # Tracking "confidence" threshold
            min_tracking_confidence=0.6
        )

        self.yaw_smoothed = 0.0
        self.pitch_smoothed = 0.0
        self.roll_smoothed = 0.0

    def process_frame(self, frame):
        """
        Processes frame, draws axes, and returns (annotated_frame, yaw, pitch, roll).
        Yaw/Pitch/Roll are in DEGREES.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # If no face detected, return False flag
        if not results.multi_face_landmarks:
            # Return original frame and last known smoothed values
            return frame, self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed, False

        for landmarks in results.multi_face_landmarks:
            # Extract features for the PyTorch model
            input_landmarks = FE.get_feature_vector_from_image(
                self.face_mesh, frame, normalize=True, isPil=False)

            if (input_landmarks == 0).all():
                return frame, self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed, False

            input_landmarks = input_landmarks.unsqueeze(dim=0).to(self.device)

            # Predict
            with torch.no_grad():
                predictions = self.model(input_landmarks)
                # Predictions are in radians, convert to degrees
                yaw, pitch, roll = map(lambda x: np.degrees(x.item()), predictions)

            # Exponential smoothing
            self.yaw_smoothed = self.alpha * yaw + (1 - self.alpha) * self.yaw_smoothed
            self.pitch_smoothed = self.alpha * pitch + (1 - self.alpha) * self.pitch_smoothed
            self.roll_smoothed = self.alpha * roll + (1 - self.alpha) * self.roll_smoothed

            # Visuals
            frame, self.prev_tdx, self.prev_tdy = visualize_axes_on_face(
                self.prev_tdx, self.prev_tdy, self.MAX_CENTER_JUMP, frame, landmarks.landmark,
                self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed
            )

        # Return True for success
        return frame, self.yaw_smoothed, self.pitch_smoothed, self.roll_smoothed, True


def visualize_axes_on_face(prev_tdx, prev_tdy, MAX_CENTER_JUMP, frame, landmarks, yaw, pitch, roll, size=80):
    # Convert back to radians for drawing geometry
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # Calculate the center of the face by averaging the coordinates of key landmarks
    # We can take landmarks around the eyes and nose as they roughly form the center
    nose_idx = 1  # Typically, the nose is at index 1 in MediaPipe
    left_eye_idx = 33  # Left eye can be a good reference
    right_eye_idx = 263  # Right eye can be a good reference

    # Get the 2D coordinates of the landmarks
    nose = landmarks[nose_idx]
    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    # # Calculate the center of the face
    # new_tdx = (nose.x + left_eye.x + right_eye.x) * frame.shape[1] / 3
    # new_tdy = (nose.y + left_eye.y + right_eye.y) * frame.shape[0] / 3

    # Map normalized coordinates to pixel coordinates
    # We use ONLY the nose for the anchor to prevent "wobble" from eye movement
    new_tdx = nose.x * frame.shape[1]
    new_tdy = nose.y * frame.shape[0]

    # If first frame, just use the new values
    if prev_tdx is None or prev_tdy is None:
        tdx, tdy = new_tdx, new_tdy
    else:
        # Compute Euclidean distance between previous and new center
        dist = math.sqrt((new_tdx - prev_tdx)**2 + (new_tdy - prev_tdy)**2)

        # if dist > MAX_CENTER_JUMP:
        #     # Use previous stable values
        #     tdx, tdy = prev_tdx, prev_tdy
        # else:
        #     # Accept new values
        #     tdx, tdy = new_tdx, new_tdy

        # If movement is small, smooth it. If large, snap to it.
        if dist < 5.0:  # Small jitter filter
            tdx = 0.5 * new_tdx + 0.5 * prev_tdx
            tdy = 0.5 * new_tdy + 0.5 * prev_tdy
        else:
            # Follow the face immediately
            tdx, tdy = new_tdx, new_tdy

    # Calculate Axis Endpoints
    # X-Axis pointing to right, drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis, drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen), drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    # Draw the axes
    cv2.line(frame, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)  # X-Axis (Red)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)  # Y-Axis (Green)
    cv2.line(frame, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Z-Axis (Blue)

    return frame, tdx, tdy