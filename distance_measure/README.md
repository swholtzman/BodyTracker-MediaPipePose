
# Distance Measure System

This module implements a real-time distance and position tracking system using a hybrid approach of fiducial markers (STag), facial feature tracking (MediaPipe FaceMesh), and skeletal body tracking (MediaPipe Pose).

It is designed to feed spatial coordinates (`x`, `y`, `yaw`) into an external audio control system via `stdout`.
___
## Coordinate System

The system outputs coordinates relative to the camera's optical center:

* **X (Lateral):** Distance from the center of the frame (Left/Right).
    * Negative = Left of center.
    * Positive = Right of center.
* **Y (Depth):** Distance from the camera lens (Forward/Backward).
    * Values are always positive.
* **Yaw:** Rotation around the vertical axis (Placeholder: currently output as `0.0`).

**Units:** Meters.
___
## Core Logic: Multi-Axis Distance Estimation

A primary challenge in single-camera depth estimation is differentiating between a subject moving away (scale decrease) and a subject rotating (width decrease). To solve this, the system implements a **Multi-Axis Heuristic**.

### The Problem
Standard focal-length estimation relies on `Distance = (Real_Width * Focal_Length) / Pixel_Width`.
When a user rotates 90 degrees (yaw):
1.  The visible `Pixel_Width` (e.g., shoulder-to-shoulder) shrinks significantly.
2.  The algorithm interprets this shrinking width as the object moving further away.
3.  The calculated distance spikes falsely (e.g., jumping from 1m to 10m).

### The Solution
The trackers (Facial and Body) now measure two orthogonal axes:
1.  **Horizontal Axis (Susceptible to Yaw):** Eye-to-Eye width or Shoulder-to-Shoulder width.
2.  **Vertical Axis (Resistant to Yaw):** Forehead-to-Chin height or Torso height (Mid-Shoulder to Mid-Hip).

The distance is calculated independently for both axes. The system assumes the **minimum** valid distance is the true distance, as rotation can only artificially inflate the distance reading (by shrinking the projected size), not deflate it.

```python
True_Distance = min(
    Focal_Ratio_Width / Current_Width_Px,
    Focal_Ratio_Height / Current_Height_Px
)
```
___
## Module Overview

### `distance_measure_main.py`

The orchestration layer that fuses data from all trackers.

* **Frame Management:** Creates distinct copies of the video frame to prevent cross-talk between libraries.
    * *Calculation Frame:* Kept pristine for STag marker detection (prevents C++ crashes).
    * *Display Frame:* Used for drawing debug lines, skeletons, and head axes.
* **Priority Logic:** Prioritizes tracking sources in the order: Marker > Face > Body.
* **Hybrid Training:**
    * When the **Marker** is visible, it provides "Ground Truth" depth.
    * It effectively "trains" the Face and Body trackers by teaching them the user's specific physical dimensions (e.g., "This specific user's shoulders are 42cm wide").
* **Smoothing:** Applies an exponential moving average (alpha filter) to raw coordinates to reduce jitter.
* **Output:** Prints aggregated `x, y, yaw` coordinates to `stdout` (throttled to updates >1cm).

### `fiducial_markers.py`

Handles STag marker detection and pose estimation.

* **Library:** Uses `stag-python` (HD23 family).
* **Crash Prevention:** Implements a safety border (padding) around image frames before processing to prevent C++ Segmentation Faults (SIGSEGV) caused by edge detection algorithms reading out-of-bounds memory.
* **Calibration:** Includes an automated lens calibration routine using a chessboard pattern to solve for the camera matrix and distortion coefficients.

### `facial_tracker.py`

Wraps MediaPipe FaceMesh.

* **Tracking:** Uses landmarks 145/374 (eyes) for width and 10/152 (forehead/chin) for height.
* **Relative Scaling:** Learns the user's specific facial dimensions relative to the physical marker size during the "Training" phase.

### `body_tracker.py`

Wraps MediaPipe Pose.

* **Tracking:** Uses landmarks 11/12 (shoulders) for width and midpoints of 11/12 to 23/24 (hips) for torso height.
* **Visualization:** Draws specific debug lines on the frame:
    * **Yellow Line:** Horizontal axis (Width).
    * **Magenta Line:** Vertical axis (Height).

### `helper_functions.py`

Contains shared mathematical utilities, specifically the `get_distance` implementation used by both tracker classes to ensure consistent multi-axis logic.

### `distance_measure_main.py`

The orchestration layer that fuses data from all trackers.

* **Frame Management:** Creates distinct copies of the video frame to prevent cross-talk between libraries.
    * *Calculation Frame:* Kept pristine for STag marker detection (prevents C++ crashes).
    * *Display Frame:* Used for drawing debug lines, skeletons, and head axes.
* **Priority Logic:** Prioritizes tracking sources in the order: Marker > Face > Body.
* **Hybrid Training:**
    * When the **Marker** is visible, it provides "Ground Truth" depth.
    * It effectively "trains" the Face and Body trackers by teaching them the user's specific physical dimensions (e.g., "This specific user's shoulders are 42cm wide").
* **Output:** Prints aggregated `x, y, yaw` coordinates to `stdout`.
___
## Usage

Run via the root runner script to ensure proper path handling:

```bash
# From project root
python runner.py
```

### Calibration

If `calibration_data.npz` is missing, the system enters **Lens Calibration Mode**.

1.  Print a checkerboard pattern.
2.  Hold it in front of the camera.
3.  Hold still when the indicator turns green until 15 frames are captured.