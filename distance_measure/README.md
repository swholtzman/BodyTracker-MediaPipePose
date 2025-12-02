
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
* **Yaw:** Rotation around the vertical axis (Normalized 0.0 - 1.0).
    * `0.0` / `1.0` = Facing Camera (0 degrees).
    * `0.25` = Facing Left (90 degrees).
    * `0.50` = Facing Away (180 degrees).
    * `0.75` = Facing Right (270 degrees).

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

The system assumes the **minimum** valid distance is the true distance, as rotation can only artificially inflate the distance reading (by shrinking the projected size), not deflate it.

```python
True_Distance = min(
    Focal_Ratio_Width / Current_Width_Px,
    Focal_Ratio_Height / Current_Height_Px
)
```
___
## Core Logic: 360° Rotation Tracking

The system implements a Stateful Rotation Tracker to handle full 360-degree spins, even when the face is not visible.

### 1. The "Synthetic Z" Constraint
MediaPipe's raw Z-axis data is often too noisy for precise rotation measurement. Instead of relying on it directly, we mathematically derive what the depth should be based on the rigid geometry of the shoulders.
- We assume the shoulders form a rigid bar of a fixed length ($Width_{max}$).
- As the user rotates, the visible 2D width ($Width_{current}$) shrinks.
- We calculate the missing depth component:
$$Z_{synthetic} = \pm \sqrt{Width_{max}^2 - Width_{current}^2}$$
- We use the raw MediaPipe Z-data only to determine the sign (Is the left shoulder in front of or behind the right shoulder?).

### 2. The "Profile View" Dead Zone (Coasting)
At exactly 90 degrees (profile view), the visible shoulder width approaches zero, making vector math unstable.
- **Threshold**: When shoulder width drops below 15% of the screen width.
- **Coasting**: The system stops calculating new angles and locks into **Coasting Mode**, applying the last known rotational velocity (momentum). This carries the tracking smoothly through the 90-degree singularity until the back comes into view.

### 3. Drift Correction (Sync)
Body tracking accumulates slight errors over time.
- **Teacher**: The Head Pose Estimator (HPE) is considered the "Ground Truth."
- **Sync**: Whenever the face is detected (0° - 45°), the Body Tracker's internal accumulator is forcibly reset to match the Head Yaw. This eliminates drift.
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
* **Rotational Filter:** Includes a "Speed Limit" (Max Yaw Step) to reject false-positive face detections (e.g., detecting the back of a head as a face) that would cause 180-degree snaps.
* **Output:** Prints aggregated `x, y, yaw` coordinates to `stdout` (throttled to updates >1cm).
* **Visuals:** Draws a dynamic Compass UI to visualize rotation state.

### `fiducial_markers.py`

Handles STag marker detection and pose estimation.

* **Library:** Uses `stag-python` (HD23 family).
* **Crash Prevention:** 
  * Implements a safety border (padding) around image frames before processing to prevent C++ Segmentation Faults (SIGSEGV) caused by edge detection algorithms reading out-of-bounds memory.
  * Implements memory layout enforcement (ascontiguousarray) to prevent known C++ SIGSEGV crashes in the STag library.
* **Calibration:** Includes an automated lens calibration routine using a chessboard pattern to solve for the camera matrix and distortion coefficients.

### `facial_tracker.py`

Wraps MediaPipe FaceMesh.

* **Tracking:** Uses landmarks 145/374 (eyes) for width and 10/152 (forehead/chin) for height.
* **Dynamic Training:** Learns the user's specific facial dimensions relative to the physical marker size during the "Training" phase.

### `body_tracker.py`

Wraps MediaPipe Pose for 360-degree tracking.

* **Synthetic Z**: Implements the geometric constraint logic described above.
* **Auto-Calibration**: Automatically learns the user's max shoulder width, with a slow decay factor to adapt to Z-depth changes without "forgetting" the user during profile views.
* **Visual Debug**:
  * **Points A (Orange) / B (Magenta)**: Visual confirmation of Left/Right shoulder tracking. 
  * **Yellow/Magenta Lines**: Visualizes the horizontal vs vertical measurement axes.

### `helper_functions.py`

Contains shared mathematical utilities, specifically the `get_distance` implementation used by both tracker classes to ensure consistent multi-axis logic.

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