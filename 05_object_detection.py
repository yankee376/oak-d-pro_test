#!/usr/bin/env python3
"""
Script 05 - On-Device Object Detection (YOLOv6 Nano)
-----------------------------------------------------------------------
Runs a pre-trained YOLOv6 Nano object detector on the OAK-D Pro Wide.
The model is fetched from the Luxonis model zoo when the pipeline starts.

Press 'q' to quit.
"""

import cv2
import depthai as dai

CONFIDENCE_THRESHOLD = 0.5

# --- Pipeline setup ---------------------------------------------------
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
nn = pipeline.create(dai.node.DetectionNetwork).build(
    cam, dai.NNModelDescription("yolov6-nano")
)
nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)

labels = nn.getClasses() or []
frame_queue = nn.passthrough.createOutputQueue()
detection_queue = nn.out.createOutputQueue()

# --- Run --------------------------------------------------------------
with pipeline:
    pipeline.start()
    print("OAK-D Pro Wide | YOLOv6 Nano  |  press 'q' to quit")

    while pipeline.isRunning():
        frame_msg = frame_queue.get()
        detection_msg = detection_queue.get()

        frame = frame_msg.getCvFrame()
        height, width = frame.shape[:2]

        for detection in detection_msg.detections:
            x1 = max(0, min(width - 1, int(detection.xmin * width)))
            y1 = max(0, min(height - 1, int(detection.ymin * height)))
            x2 = max(0, min(width - 1, int(detection.xmax * width)))
            y2 = max(0, min(height - 1, int(detection.ymax * height)))

            if 0 <= detection.label < len(labels):
                label = labels[detection.label]
            else:
                label = f"#{detection.label}"

            text = f"{label}  {detection.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Detections: {len(detection_msg.detections)}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.imshow("OAK-D Object Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
