import cv2
import numpy as np
import random

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

window_name = "Automated Multi-Stage QA Pipeline"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fruits = []
MAX_FRUITS = 6  

telemetry = {
    "Passed_Total": 0,
    "Defects_Total": 0
}

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape


        display_frame = (frame * 0.65).astype(np.uint8)

        # --- DRAW TOP SENSOR BAR (INSPECTION ZONE) ---
        sensor_top = 50
        sensor_bottom = 110
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, sensor_top), (w, sensor_bottom), (80, 40, 0), -1)
        cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, dst=display_frame)
        cv2.line(display_frame, (0, sensor_top), (w, sensor_top), (255, 128, 0), 1, cv2.LINE_AA)
        cv2.line(display_frame, (0, sensor_bottom), (w, sensor_bottom), (255, 128, 0), 1, cv2.LINE_AA)
        cv2.putText(display_frame, "IN-LINE WEIGHT & DENSITY SENSOR SYSTEM", (30, sensor_bottom - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 100), 1, cv2.LINE_AA)

        # --- GESTURE PROCESSING ENGINE ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        hand_obstacles = []
        is_palm_open = False

        if detection_result.hand_landmarks and detection_result.handedness:
            for hand_landmarks in detection_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                hand_obstacles.extend(pts)
                
                fingers_extended = [
                    pts[8][1] < pts[6][1],    
                    pts[12][1] < pts[10][1],  
                    pts[16][1] < pts[14][1],  
                    pts[20][1] < pts[18][1]   
                ]
                
                if sum(fingers_extended) >= 3:
                    is_palm_open = True

                skeleton_color = (0, 255, 0) if is_palm_open else (0, 0, 255)
                
                connections = [
                    (0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8),
                    (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
                    (17,18), (18,19), (19,20), (0,5), (5,9), (9,13), (13,17), (0,17)
                ]
                for start, end in connections:
                    if start < len(pts) and end < len(pts):
                        cv2.line(display_frame, pts[start], pts[end], skeleton_color, 2, cv2.LINE_AA)

        # --- RAW UNASSIGNED FRUIT SPAWNER ---
        if len(fruits) < MAX_FRUITS and random.random() < 0.02: 
            fruit_type = random.choice(["Apple", "Orange"])
            radius = 16 if fruit_type == "Orange" else 12
            
            fruits.append({
                "x": random.randint(w // 4, 3 * w // 4),
                "y": -30,
                "vx": random.uniform(-0.3, 0.3),
                "vy": random.uniform(1.0, 1.6),  
                "radius": radius,
                "type": fruit_type,
                "color": (200, 200, 200),  
                "status": "Scanning",
                "crushed": False
            })

        # --- MAIN SORTING KINEMATICS ENGINE ---
        for f in fruits[:]:
            f["vy"] += 0.06  
            f["x"] += f["vx"]
            f["y"] += f["vy"]

            # --- SENSOR INTERCEPT TRIGGER LOGIC ---
            if f["status"] == "Scanning" and (sensor_top <= f["y"] <= sensor_bottom):
                if random.random() < 0.35:  
                    f["status"] = "Defect"
                    f["color"] = (100, 140, 140)  
                else:
                    f["status"] = "Good"
                    f["color"] = (0, 165, 255) if f["type"] == "Orange" else (0, 0, 255)

            # --- HAND INTERACTIVE COLLISION ZONE ---
            for (hx, hy) in hand_obstacles:
                distance = np.hypot(f["x"] - hx, f["y"] - hy)
                if distance < (f["radius"] + 15):
                    if is_palm_open:
                        # --- DYNAMIC VECTOR ROUTING FOR QUALITY FRUIT ---
                        if f["status"] == "Good":
                            f["vx"] = -3.5  # Forces the good fruit left into the green bucket area
                    else:
                        # Closed Hand: If it's a defect, crush it!
                        if f["status"] == "Defect" and not f["crushed"]:
                            f["crushed"] = True
                            f["color"] = (50, 50, 50)  
                            f["radius"] = 5
                            # Send crushed fragments sliding over towards the right side red bucket
                            f["vx"] = random.uniform(2.5, 4.5)  
                            f["vy"] = 2.0
                        elif f["status"] == "Good":
                            dx, dy = f["x"] - hx, f["y"] - hy
                            norm = np.hypot(dx, dy) or 1
                            # Regular collision bouncing if you close fist on a good fruit
                            f["vx"] = (dx / norm) * 5.0
                            f["vy"] = (dy / norm) * 3.0
                    break

            cv2.circle(display_frame, (int(f["x"]), int(f["y"])), f["radius"], f["color"], -1)
            
            if f["crushed"]:
                lbl = "CRUSHED 💥"
                lbl_color = (120, 120, 255)
            elif f["status"] == "Scanning":
                lbl = "WEIGHING..."
                lbl_color = (200, 200, 200)
            else:
                lbl = f"{f['type']}"
                lbl_color = (0, 255, 0) if f["status"] == "Good" else (0, 0, 255)

            cv2.putText(display_frame, lbl, (int(f["x"]) - 35, int(f["y"]) - f["radius"] - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, lbl_color, 1, cv2.LINE_AA)

            # --- PHYSICAL BIN BOUNDARY SORTING ASSESSMENT ---
            bin_h = 100
            if f["y"] >= (h - bin_h):
                if f["x"] < w // 2:
                    if f["status"] == "Good":
                        telemetry["Passed_Total"] += 1
                    fruits.remove(f)
                else:
                    if f["status"] == "Defect" or f["crushed"]:
                        telemetry["Defects_Total"] += 1
                    fruits.remove(f)
            elif f["x"] < 0 or f["x"] > w:
                if f["x"] < 0 and f["status"] == "Good":
                    telemetry["Passed_Total"] += 1
                elif f["x"] > w and (f["status"] == "Defect" or f["crushed"]):
                    telemetry["Defects_Total"] += 1
                fruits.remove(f)

        # --- GRAPHICAL SORTING BINS OVERLAY (BOTTOM OF SCREEN) ---
        bin_h = 100
        mid_point = w // 2
        
        # Left Bin - GOOD PRODUCT LINE
        cv2.rectangle(display_frame, (0, h - bin_h), (mid_point - 5, h), (20, 40, 20), -1)
        cv2.rectangle(display_frame, (0, h - bin_h), (mid_point - 5, h), (0, 255, 0), 2)
        cv2.putText(display_frame, "PASSED LINE (GOOD)", (20, h - bin_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"TOTAL: {telemetry['Passed_Total']}", (20, h - bin_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Right Bin - DEFECT SCRAP LINE
        cv2.rectangle(display_frame, (mid_point + 5, h - bin_h), (w, h), (20, 20, 40), -1)
        cv2.rectangle(display_frame, (mid_point + 5, h - bin_h), (w, h), (0, 0, 255), 2)
        cv2.putText(display_frame, "DEFECTS SCRAP (CRUSHED)", (mid_point + 20, h - bin_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(display_frame, f"TOTAL: {telemetry['Defects_Total']}", (mid_point + 20, h - bin_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        mode_str = "OPEN GATE" if is_palm_open else "CRUSHER ARMED"
        mode_color = (0, 255, 0) if is_palm_open else (0, 0, 255)
        cv2.putText(display_frame, f"ACTUATOR: {mode_str}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

        cv2.imshow(window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()