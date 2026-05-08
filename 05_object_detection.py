#!/usr/bin/env python3
"""
Script 05 - On-Device Object Detection (MobileNet SSD)
-----------------------------------------------------------------------
Hardware used : IMX378 (CAM_A) + Myriad X VPU (on-device AI)
What you learn: How to run a neural network directly on the camera's
                VPU — no GPU, no cloud, no host CPU needed for inference.

                MobileNet SSD is a lightweight object detector trained
                on PASCAL VOC (20 classes). It runs at >30 FPS on the
                Myriad X at <10ms latency.

Network input : 300x300 BGR image
Network output: list of detections (class, confidence, bounding box)

DepthAI v3 pipeline overview:
    [Camera CAM_A] --> requestOutput(300x300) --> DetectionNetwork.input
                   --> requestOutput(640x480) --> display queue
    DetectionNetwork.out --> detection queue
-----------------------------------------------------------------------
First run downloads the .blob model file (~5 MB) via blobconverter.
Press 'q' to quit.
"""

import cv2
import depthai as dai
import blobconverter

# PASCAL VOC labels (index matches the network output class ID)
LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

CONFIDENCE_THRESHOLD = 0.5

# Download MobileNet SSD blob compiled for 6 MyriadX shaves (first run only)
print("Loading MobileNet SSD model...")
blob_path = blobconverter.from_zoo(
    name="mobilenet-ssd",
    shaves=6,
    zoo_type="depthai",
)
print(f"Model ready: {blob_path}\n")

# --- Pipeline setup ---------------------------------------------------
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

# 300x300 feed goes to the neural network
nn_input   = cam.requestOutput((300, 300), dai.ImgFrame.Type.BGR888p)

# Higher-resolution feed for the display window
display_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)

# DetectionNetwork runs MobileNet SSD on the VPU
nn = pipeline.create(dai.node.DetectionNetwork)
nn.setBlobPath(blob_path)
nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
nn.setNumInferenceThreads(2)
nn_input.link(nn.input)

display_queue   = display_out.createOutputQueue()
detection_queue = nn.out.createOutputQueue()

# --- Run --------------------------------------------------------------
with pipeline:
    pipeline.start()
    print("OAK-D Pro Wide | Object Detection  |  press 'q' to quit")

    while pipeline.isRunning():
        frame      = display_queue.get().getCvFrame()
        detections = detection_queue.get().detections

        h, w = frame.shape[:2]

        for det in detections:
            # Bounding box coords are normalized 0..1 — scale to frame size
            x1 = int(det.xmin * w)
            y1 = int(det.ymin * h)
            x2 = int(det.xmax * w)
            y2 = int(det.ymax * h)

            label = LABELS[det.label] if det.label < len(LABELS) else f"#{det.label}"
            conf  = det.confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"{label}  {conf:.0%}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2,
            )

        # Show detection count
        cv2.putText(frame, f"Detections: {len(detections)}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("OAK-D Object Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
