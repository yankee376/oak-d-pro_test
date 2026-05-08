#!/usr/bin/env python3
"""
OAK-D Pro Wide - Project Launcher
Run this file to pick a script interactively, or run each script directly.
"""
import subprocess
import sys

SCRIPTS = {
    "1": ("01_rgb_preview.py",        "RGB Camera Preview"),
    "2": ("02_depth_preview.py",      "Stereo Depth Preview"),
    "3": ("03_imu_data.py",           "IMU Data Stream"),
    "4": ("04_rgb_depth_combined.py", "RGB + Depth Combined"),
    "5": ("05_object_detection.py",   "On-Device Object Detection"),
}


def main():
    print("\n" + "=" * 48)
    print("    OAK-D Pro Wide - Vision Projects")
    print("=" * 48)
    for key, (_, name) in SCRIPTS.items():
        print(f"  [{key}]  {name}")
    print("  [q]  Quit")
    print("=" * 48)

    choice = input("\nChoose a script: ").strip().lower()
    if choice in SCRIPTS:
        script, name = SCRIPTS[choice]
        print(f"\nStarting: {name}\n")
        subprocess.run([sys.executable, script])
    elif choice == "q":
        print("Bye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
