#!/usr/bin/env python3
"""
Script 02 - Stereo Depth Preview
-----------------------------------------------------------------------
Hardware used : 2x OV9282 mono cameras (CAM_B = left, CAM_C = right)
What you learn: Stereo vision basics - two cameras see the same scene
                from slightly different angles. The horizontal shift of
                a pixel between the two views is called *disparity*.
                Closer objects have MORE disparity; far objects have LESS.

                The StereoDepth node computes disparity on the VPU in
                real time, no host computation needed.

DepthAI v3 pipeline overview:
    [Camera CAM_B] --> requestFullResolutionOutput --> stereo.left
    [Camera CAM_C] --> requestFullResolutionOutput --> stereo.right
    stereo.disparity --> createOutputQueue --> host Python
-----------------------------------------------------------------------
Key settings:
  setLeftRightCheck   - reject mismatched pixels, reduces noise
  setExtendedDisparity- larger disparity range, better for close objects
  setSubpixel         - sub-pixel interpolation, improves accuracy

Press 'q' to quit.
"""

import cv2
import numpy as np
import depthai as dai

# --- Pipeline setup ---------------------------------------------------
pipeline = dai.Pipeline()

monoLeft  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo    = pipeline.create(dai.node.StereoDepth)

# Link mono outputs to the stereo node inputs
monoLeft.requestFullResolutionOutput().link(stereo.left)
monoRight.requestFullResolutionOutput().link(stereo.right)

stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)   # best for OAK-D Pro Wide's wide FOV
stereo.setSubpixel(True)

disparity_queue = stereo.disparity.createOutputQueue()

# Invalid pixels (disparity 0) are shown as black after applying the colormap.

# --- Run --------------------------------------------------------------
with pipeline:
    pipeline.start()
    print("OAK-D Pro Wide | Stereo Depth  |  press 'q' to quit")
    print("  Color guide: black=no data  blue=far  red=close")

    max_disparity = 1  # tracked dynamically for normalization

    while pipeline.isRunning():
        msg = disparity_queue.get()
        assert isinstance(msg, dai.ImgFrame)

        disp = msg.getFrame()                          # 2D numpy array (uint8/uint16)
        max_disparity = max(max_disparity, np.max(disp))

        # Normalize to 0-255 then apply colormap
        normalized = ((disp / max_disparity) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colored[disp == 0] = (0, 0, 0)

        cv2.imshow("OAK-D Depth", colored)
        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()
            break

cv2.destroyAllWindows()
