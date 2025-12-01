# Body & Head Tracking Pipeline

This project integrates two distinct computer vision systems into a single pipeline to track a user's 3D position relative to a camera.

1.  **Distance Measure (MediaPipe/STag):** Calculates `X` (Lateral) and `Y` (Depth) in meters.
2.  **Head Pose Estimation (PyTorch):** Calculates `Yaw` (Rotation) in normalized units.

The output is streamed to `stdout` for consumption by external audio control drivers.

## Directory Structure

* **`distance_measure/`**: Logic for depth tracking using Fiducial Markers (Teacher) and Body/Face landmarks (Student).
* **`head_pose_estimation/`**: Logic for determining 3D head orientation using a PyTorch neural network.
* **`runner.py`**: The entry point that initializes the environment and launches the pipeline.

## Installation

**1. Python Environment**
Ensure you are using Python 3.12 (recommended for MediaPipe/Numpy compatibility on macOS).

```bash
# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Dependencies Install the combined requirements for both sub-systems.**

```bash
pip install -r requirements.txt
pip install torch torchvision  # Required for Head Pose
```

##  Usage
Always run the system using the root runner to ensure all submodule paths are loaded correctly.

```bash
python runner.py
```

## Output Format
The system prints CSV-formatted data to standard output whenever a significant movement (>1cm) is detected:

```plaintext
x, y, yaw
```
- x: Lateral distance in meters (Negative = Left, Positive = Right).
- y: Depth distance in meters (Always positive).
- yaw: Head rotation normalized (0.0 - 1.0, where 360 degrees = 1.0).

## Known Issues & Fixes
**1. "Incorrect stag library detected"**
- **Cause**: Numpy 2.0+ breaks the stag-python library.
- **Fix**: Ensure you are running numpy<2. The requirements.txt is pinned to handle this.

**2. SIGSEGV / Crash on Frame Processing**
- **Cause**: The STag library crashes if it detects sharp digital lines (like drawn axes) on the video frame.
- **Fix**: The pipeline strictly separates the Calculation Frame (clean video) from the Display Frame (visual debugs).
