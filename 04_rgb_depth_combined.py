#!/usr/bin/env python3
"""
Script 04 - RGB + Depth Combined
-----------------------------------------------------------------------
Hardware used : IMX378 (CAM_A) + 2x OV9282 (CAM_B, CAM_C)
What you learn: Running multiple nodes in one pipeline, reading from
                two output queues, and compositing frames side by side.

                A real-world use case: knowing WHAT an object is (RGB)
                and HOW FAR it is (depth) at the same time.

DepthAI v3 pipeline overview:
    [Camera CAM_A] --> requestOutput --> queue (rgb)
    [Camera CAM_B] --> requestFullResolutionOutput --> stereo.left
    [Camera CAM_C] --> requestFullResolutionOutput --> stereo.right
    stereo.depth   --> queue (depth)
-----------------------------------------------------------------------
Press 'q' to quit.
"""

import cv2
import numpy as np
import depthai as dai

DISPLAY_W, DISPLAY_H = 640, 400  # size for each panel in the composite view

# --- Pipeline setup ---------------------------------------------------
pipeline = dai.Pipeline()

# RGB
cam_rgb  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
rgb_out  = cam_rgb.requestOutput((DISPLAY_W, DISPLAY_H), dai.ImgFrame.Type.BGR888p)
rgb_queue = rgb_out.createOutputQueue()

# Stereo depth
mono_left  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
mono_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo     = pipeline.create(dai.node.StereoDepth)

mono_left.requestFullResolutionOutput().link(stereo.left)
mono_right.requestFullResolutionOutput().link(stereo.right)

stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

depth_queue = stereo.depth.createOutputQueue()

# --- Run --------------------------------------------------------------
with pipeline:
    pipeline.start()
    print("OAK-D Pro Wide | RGB + Depth  |  press 'q' to quit")

    while pipeline.isRunning():
        rgb_msg   = rgb_queue.get()
        depth_msg = depth_queue.get()

        # RGB frame — already the right size from requestOutput
        rgb = rgb_msg.getCvFrame()

        # Depth frame is in mm (uint16). Normalize and colorize for display.
        depth = depth_msg.getFrame().astype(np.float32)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_color = cv2.resize(depth_color, (DISPLAY_W, DISPLAY_H))

        # Label each panel
        cv2.putText(rgb,         "RGB (IMX378)",      (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_color, "Depth (OV9282 x2)", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Side-by-side composite
        combined = np.hstack([rgb, depth_color])
        cv2.imshow("OAK-D Pro Wide | RGB + Depth", combined)

        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
