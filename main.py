import cv2
import numpy as np
import random
import os

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(0)
particles = []
MAX_PARTICLES = 100  # Lowered slightly for cleaner tracking visibility

# High contrast colors for different weights
NEON_COLORS = [
    (0, 255, 255),   # Yellow/Cyan - Light & Hyper-bouncy
    (255, 0, 255),   # Magenta - Medium
    (0, 0, 255)      # Pure Red - Heavy & Dead weight
]

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Create a fresh, solid dark background every single frame (No more trails!)
        display_frame = (frame * 0.2).astype(np.uint8)

        # Process MediaPipe tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        hand_obstacles = []

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                hand_obstacles.extend(pts)

                # Draw clean biometric skeleton lines
                connections = [
                    (0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8),
                    (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
                    (17,18), (18,19), (19,20), (0,5), (5,9), (9,13), (13,17), (0,17)
                ]
                for start, end in connections:
                    if start < len(pts) and end < len(pts):
                        cv2.line(display_frame, pts[start], pts[end], (255, 255, 255), 1, cv2.LINE_AA)

        # --- SPAWN PARTICLES ---
        if len(particles) < MAX_PARTICLES:
            radius = random.choice([2, 4, 8])  # Extreme sizes: 2 (tiny), 4 (mid), 8 (huge)
            
            # Map colors distinctly to size groups
            if radius == 2:
                color = (0, 255, 255)   # Cyan (Hyper)
            elif radius == 4:
                color = (0, 255, 0)     # Green (Mid)
            else:
                color = (0, 0, 255)     # Red (Heavy Lead)

            particles.append({
                "x": random.randint(20, w - 20),
                "y": random.randint(-10, 0),
                "vx": random.uniform(-1, 1),
                "vy": random.uniform(2, 4),
                "radius": radius,
                "color": color
            })

        # --- HIGH CONTRAST PHYSICS LOOP ---
        for p in particles[:]:
            # Uniform Gravity
            p["vy"] += 0.25  
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Collision matching against hand coordinates
            for (hx, hy) in hand_obstacles:
                distance = np.hypot(p["x"] - hx, p["y"] - hy)
                
                if distance < 25:
                    dx = p["x"] - hx
                    dy = p["y"] - hy
                    norm = np.hypot(dx, dy)
                    if norm == 0: norm = 1
                    
                    # --- INTENSE MASS DISPARITY MECHANICS ---
                    if p["radius"] == 2:
                        # Tiny balls retain massive energy + bonus impulse speed
                        bounce_velocity = 11.0 
                    elif p["radius"] == 4:
                        bounce_velocity = 6.0
                    else:
                        # Huge red balls lose almost all energy on impact (Dampened slide)
                        bounce_velocity = 1.5 
                    
                    # Apply vector force updates
                    p["vx"] = (dx / norm) * bounce_velocity
                    p["vy"] = (dy / norm) * bounce_velocity
                    
                    # Instant position adjustment to completely remove sticky glitches
                    p["x"] += (dx / norm) * 4
                    p["y"] += (dy / norm) * 4

            # Draw crisp shapes without any trails
            cv2.circle(display_frame, (int(p["x"]), int(p["y"])), p["radius"], p["color"], -1)

            # Clear out-of-bounds metrics
            if p["y"] > h or p["x"] < 0 or p["x"] > w:
                particles.remove(p)

        # Clean HUD
        cv2.putText(display_frame, "MASS CONSERVATION LAB", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "Cyan (Light) = Hyper Bounce | Red (Heavy) = Dead Drop", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        cv2.imshow("Physics Display Grid", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()