#!/usr/bin/env python3
"""
=======================================================================
Script 01 - RGB Camera Preview
=======================================================================
WHAT THIS SCRIPT DOES:
  Opens the OAK-D Pro Wide color camera and shows the live video feed
  in a window on your screen. That's it. Simple but important — this
  is how every camera project begins.

HARDWARE INVOLVED:
  The OAK-D Pro Wide has THREE cameras:
    CAM_A  →  the COLOR (RGB) camera    ← we use this one here
    CAM_B  →  left MONO (black & white) camera
    CAM_C  →  right MONO (black & white) camera

  CAM_A uses a Sony IMX378 sensor, 12 megapixels, with a very wide
  120-degree field of view.

WHAT YOU WILL LEARN:
  - What a "pipeline" means in DepthAI
  - How images are represented as numbers (numpy arrays)
  - How BGR color order differs from RGB
  - How to display a live video stream with OpenCV
=======================================================================
Press 'q' inside the window to quit.
"""

# =======================================================================
# IMPORTS — loading external libraries we need
# =======================================================================

# cv2 is OpenCV — the most popular computer vision library in the world.
# It gives us tools to: display images, draw on them, process them, etc.
# "import cv2" loads the whole library and lets us call it as "cv2"
import cv2

# depthai is the Python library for all Luxonis OAK cameras.
# "import depthai as dai" loads it and gives it the short nickname "dai"
# so we can write  dai.Pipeline()  instead of  depthai.Pipeline()
import depthai as dai


# =======================================================================
# STEP 1 — CREATE A PIPELINE
# =======================================================================
# A Pipeline is the "blueprint" or "recipe" we send to the camera.
# It tells the camera: which sensors to activate, how to process data,
# and how to send results back to our Python script.
#
# Think of it like a factory assembly line:
#   [raw sensor data] → [processing steps] → [final output to you]
#
# We define the pipeline BEFORE starting the camera.
pipeline = dai.Pipeline()


# =======================================================================
# STEP 2 — ADD A CAMERA NODE TO THE PIPELINE
# =======================================================================
# A "node" is one processing block in the pipeline.
# dai.node.Camera is the node type that reads from a physical sensor.
#
# pipeline.create(...)  adds the node to our blueprint.
# .build(dai.CameraBoardSocket.CAM_A)  tells it WHICH sensor to use.
#
# CameraBoardSocket.CAM_A = the RGB color camera (IMX378)
# CameraBoardSocket.CAM_B = left mono camera
# CameraBoardSocket.CAM_C = right mono camera
#
# The result is stored in the variable "cam" so we can configure it next.
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)


# =======================================================================
# STEP 3 — DEFINE WHAT OUTPUT WE WANT FROM THE CAMERA
# =======================================================================
# The camera can produce different sizes and color formats.
# We ask for exactly what we want with requestOutput().
#
#   (1920, 1080)  →  Full HD resolution (width=1920, height=1080 pixels)
#                    That is 1920 × 1080 = ~2 million pixels per frame!
#
#   dai.ImgFrame.Type.BGR888p  →  the color format.
#
#       Images store color as 3 numbers per pixel: one for each channel.
#       Most people know RGB (Red, Green, Blue).
#       OpenCV uses BGR (Blue, Green, Red) — same channels, different order.
#
#       "888"  means 8 bits per channel (values 0–255 each)
#       "p"    means "planar" (each channel stored separately in memory)
#
#       We use BGR888p because OpenCV's getCvFrame() and imshow()
#       expect this format natively — no conversion needed.
#
# frame_output is a "connection point" between the camera node and our code.
frame_output = cam.requestOutput((1920, 1080), dai.ImgFrame.Type.BGR888p)


# =======================================================================
# STEP 4 — CREATE A QUEUE (THE DATA BRIDGE TO OUR PYTHON CODE)
# =======================================================================
# The camera runs on its own processor (the Myriad X VPU chip).
# Our Python script runs on the computer's CPU.
# They are separate — so how does data get from the camera to Python?
#
# Answer: a QUEUE — imagine a conveyor belt between the camera and Python.
#   - The camera pushes frames onto one end of the belt
#   - Our Python code picks frames off the other end
#
# createOutputQueue() creates this bridge and returns it as "queue".
# Later, we call  queue.get()  to grab the next available frame.
queue = frame_output.createOutputQueue()


# =======================================================================
# STEP 5 — START THE PIPELINE AND RUN THE MAIN LOOP
# =======================================================================
# "with pipeline:" is a Python context manager.
# It automatically handles cleanup (stopping the camera, freeing memory)
# when the block finishes — even if something crashes. Good practice.
with pipeline:

    # pipeline.start() powers up the camera and starts running our blueprint.
    # After this line, frames begin flowing through the pipeline.
    pipeline.start()
    print("OAK-D Pro Wide | RGB Preview  |  press 'q' to quit")

    # This is the MAIN LOOP — it runs forever until we press 'q'.
    # pipeline.isRunning() returns True as long as the camera is connected
    # and the pipeline is active.
    while pipeline.isRunning():

        # -------------------------------------------------------------------
        # GRAB THE NEXT FRAME
        # -------------------------------------------------------------------
        # queue.get() WAITS until a new frame arrives, then returns it.
        # It is a "blocking" call — the code pauses here until data comes.
        # The result is a DepthAI message object called ImgFrame.
        msg = queue.get()

        # This line just checks we got the right type of message.
        # "assert" will crash loudly if the check fails — helpful for debugging.
        assert isinstance(msg, dai.ImgFrame)

        # -------------------------------------------------------------------
        # CONVERT THE FRAME TO A NUMPY ARRAY
        # -------------------------------------------------------------------
        # getCvFrame() converts the DepthAI message into a NumPy array.
        #
        # WHAT IS A NUMPY ARRAY?
        #   NumPy is a Python library for numerical data. An "array" is just
        #   a grid of numbers. An image IS a grid of numbers:
        #
        #     frame.shape  →  (height, width, channels)
        #     e.g.          →  (1080, 1920, 3)
        #
        #   So our 1080p image is a 3D array:
        #     1080 rows  (vertical pixels)
        #     1920 columns  (horizontal pixels)
        #     3 numbers per pixel  (Blue, Green, Red, each 0–255)
        #
        #   Pixel at row 0, column 0 (top-left corner):
        #     frame[0, 0]  →  [B, G, R]  e.g. [255, 128, 0]
        #
        # This is the standard format that all OpenCV functions understand.
        frame = msg.getCvFrame()

        # -------------------------------------------------------------------
        # DRAW TEXT OVERLAY ON THE FRAME
        # -------------------------------------------------------------------
        # frame.shape gives (height, width, channels).
        # We unpack just the first two values with [:2].
        h, w = frame.shape[:2]

        # cv2.putText() draws text directly onto the frame array.
        # Arguments explained:
        #   frame              →  the image to draw on (modified in place)
        #   f"{w}x{h} | ..."   →  the text string (f-string fills in values)
        #   (12, 32)           →  position: (x from left, y from top) in pixels
        #   cv2.FONT_HERSHEY_SIMPLEX  →  font style (a clean sans-serif font)
        #   0.8                →  font scale (size multiplier)
        #   (0, 255, 0)        →  text color in BGR: (B=0, G=255, R=0) = GREEN
        #   2                  →  line thickness in pixels
        cv2.putText(
            frame,
            f"{w}x{h}  |  IMX378 (CAM_A)",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),   # green
            2,
        )

        # -------------------------------------------------------------------
        # DISPLAY THE FRAME
        # -------------------------------------------------------------------
        # cv2.imshow() opens a window and draws the frame in it.
        # First argument is the window title.
        # Every call to imshow() updates (redraws) that window with the
        # new frame — this is how we get smooth video from still images.
        cv2.imshow("OAK-D RGB", frame)

        # -------------------------------------------------------------------
        # KEYBOARD INPUT — CHECK IF USER PRESSED 'q'
        # -------------------------------------------------------------------
        # cv2.waitKey(1) does two things:
        #   1. It gives the OS 1 millisecond to refresh/draw the window.
        #      Without this call, the window would never update visually!
        #   2. It checks if any keyboard key was pressed and returns its code.
        #      If no key was pressed, it returns -1.
        #
        # ord("q") converts the character 'q' to its ASCII number (113).
        # So: if waitKey returned 113, the user pressed 'q' → we stop.
        if cv2.waitKey(1) == ord("q"):
            pipeline.stop()   # cleanly tells the camera pipeline to stop
            break             # exit the while loop


# =======================================================================
# STEP 6 — CLEANUP
# =======================================================================
# cv2.destroyAllWindows() closes any OpenCV windows that are still open.
# This runs after the "with pipeline:" block ends.
cv2.destroyAllWindows()


# =======================================================================
# QUICK SUMMARY OF THE FLOW
# =======================================================================
#
#  1. Create pipeline (blueprint)
#  2. Add Camera node → ask for 1920x1080 BGR output
#  3. Create queue (data bridge between camera and Python)
#  4. Start pipeline (power on camera, begin streaming)
#  5. Loop:
#       a. Wait for next frame  (queue.get)
#       b. Convert to numpy array  (getCvFrame)
#       c. Draw text on it  (putText)
#       d. Show it in a window  (imshow)
#       e. Check for 'q' key  (waitKey)
#  6. Stop pipeline and close windows
#
# Each iteration of the loop = one frame = one image.
# At 30 FPS, this loop runs ~30 times per second.
# =======================================================================
