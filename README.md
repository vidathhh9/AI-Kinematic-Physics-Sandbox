# Computer Vision Physics Sandbox & Industrial QA Pipeline

An interactive, real-time computer vision application built with Python, OpenCV, and MediaPipe. The project demonstrates real-time biometric tracking, custom kinematics, and state-machine logic applied to a simulated industrial environment.

## 🛠️ Features & Architecture

The repository contains two distinct phases of development:

### 1. Basic Sandbox (`main.py`)
* Real-time 21-point hand skeleton tracking.
* Dynamic rigid-body collisions where hand bounds act as physical deflectors.
* Variable mass kinematics (high-impulse bouncy particles vs. heavy sliding particles).

### 2. Advanced QA Sorting Pipeline (`advanced_sorting.py`)
* **In-Line Sensor Simulation:** Fruits (Apples & Oranges) drop through a top-screen inspection band where their density/weight is automatically evaluated and flagged as `Good` or `Defect`.
* **Biometric Gesture Actuation:** Uses high-precision individual finger extension tracking. 
  * **Open Palm:** Acts as an open routing gate, automatically guiding premium grade fruits smoothly into the left-side **Passed Line** bucket.
  * **Closed Fist:** Functions as an automated crushing mechanism. Intercepting a defective fruit with a fist shatters it into fragments, routing it directly into the right-side **Defects Scrap** bucket.
* **Graphical Telemetry Bins:** Real-time production statistics are dynamically rendered inside color-coded collection buckets at the bottom of the display.

---

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
python -m pip install opencv-python numpy mediapipe pycaw comtypes
python advanced_sorting.py
