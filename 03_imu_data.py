#!/usr/bin/env python3
"""
Script 03 - IMU Data Stream
-----------------------------------------------------------------------
Hardware used : BNO086 IMU (built into OAK-D Pro)
What you learn: How to read raw sensor data (accelerometer + gyroscope)
                from the on-board Inertial Measurement Unit.

IMU sensors:
  Accelerometer - measures linear acceleration in m/s^2 along X/Y/Z.
                  At rest, Z will read ~9.8 (gravity).
  Gyroscope     - measures rotational velocity in rad/s around X/Y/Z.
                  At rest, all values should be near 0.

DepthAI v3 pipeline overview:
    [IMU node] --> out.createOutputQueue --> host Python
-----------------------------------------------------------------------
Try tilting and rotating the camera while watching the values.
Press Ctrl+C to quit.
"""

import depthai as dai

# --- Pipeline setup ---------------------------------------------------
pipeline = dai.Pipeline()

imu = pipeline.create(dai.node.IMU)

# Enable sensors with desired sample rates (Hz).
# BNO086 max: accelerometer 500 Hz, gyroscope 400 Hz.
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)

# How many readings to batch before sending to host.
# 1 = send immediately after each reading (low latency).
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

imu_queue = imu.out.createOutputQueue()

# --- Run --------------------------------------------------------------
with pipeline:
    pipeline.start()
    print("OAK-D Pro Wide | IMU Data Stream  |  Ctrl+C to quit\n")
    print(f"{'Accel X':>10} {'Accel Y':>10} {'Accel Z':>10} |"
          f" {'Gyro X':>10} {'Gyro Y':>10} {'Gyro Z':>10}  (units: m/s² | rad/s)")
    print("-" * 80)

    try:
        while pipeline.isRunning():
            imu_data = imu_queue.get()

            for packet in imu_data.packets:
                accel = packet.acceleroMeter
                gyro  = packet.gyroscope

                print(
                    f"{accel.x:>10.3f} {accel.y:>10.3f} {accel.z:>10.3f} |"
                    f" {gyro.x:>10.4f} {gyro.y:>10.4f} {gyro.z:>10.4f}"
                )

    except KeyboardInterrupt:
        print("\nStopped.")
