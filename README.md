# AI Kinematic Physics Sandbox ⚖️🖐️

A real-time custom physics engine driven entirely by computer vision hand-tracking. Built using Python, OpenCV, and Google's MediaPipe framework, this simulation models conservation of momentum and coefficients of restitution based on object mass.

## Features
* 🖐️ **Biometric Tracking:** Maps and tracks hand landmark coordinates in real-time.
* 🧮 **Mass-Based Collision:** Light particles (Cyan) retain high kinetic energy for hyper-bounces, while high-mass particles (Red) undergo heavy energy absorption and damping on impact.
* 🚀 **Zero-Glitch Environment:** Includes instant positional correction to prevent particle clipping on fast hand movements.

## Tech Stack
* Python 3
* OpenCV
* MediaPipe
* NumPy

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

   python main.py
