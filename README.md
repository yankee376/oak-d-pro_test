# OAK-D Pro Wide — Vision Projects

Learning computer vision step by step using the **Luxonis OAK-D Pro Wide** camera and the DepthAI v3 Python API.

## About the Camera

The OAK-D Pro Wide is an edge AI camera with:

| Component | Spec |
|---|---|
| RGB sensor | IMX378, 12MP, 120° FOV |
| Stereo cameras | 2× OV9282, 120° FOV |
| VPU (on-device AI) | Intel Myriad X — 4 TOPS |
| IMU | BNO086 (accel + gyro + rotation) |
| IR | Laser dot projector + flood LED |

Everything runs on the camera itself — no GPU needed for inference.

## Setup

```bash
pip install -r requirements.txt
```

Plug in the OAK-D Pro Wide via USB 3.0 (blue port) before running any script.

## Scripts

Run each script directly with Python, or use the launcher:

```bash
python main.py
```

| Script | What you learn |
|---|---|
| `01_rgb_preview.py` | Connecting to the camera, reading frames, displaying with OpenCV |
| `02_depth_preview.py` | Stereo vision fundamentals, disparity maps, colorized depth |
| `03_imu_data.py` | Reading IMU sensor data (accelerometer + gyroscope) |
| `04_rgb_depth_combined.py` | Multi-stream pipelines, frame synchronization, side-by-side display |
| `05_object_detection.py` | On-device object detection (YOLOv6 Nano, COCO classes) |

## Key Concepts

### Stereo Depth
The OAK-D Pro Wide uses two monochrome cameras spaced apart. By comparing the shift (disparity) of the same point between left and right frames, depth (distance) is calculated — the same way human eyes work.

### On-Device AI (VPU)
The Myriad X VPU runs neural networks directly on the camera. Frames never leave the device for inference, making it fast (<10ms) and privacy-friendly.

### DepthAI Pipeline
You define a *pipeline* (a graph of nodes), then start it. The camera runs the pipeline on its processor. Results are streamed to your Python script via output queues.

```
Camera → StereoDepth → [host]
Camera → NeuralNetwork → [host]
```

## Resources

- [DepthAI v3 Docs](https://docs.luxonis.com/software-v3/depthai/)
- [OAK-D Pro Wide product page](https://shop.luxonis.com/products/oak-d-pro-wide)
- [Luxonis GitHub examples](https://github.com/luxonis/oak-examples)
