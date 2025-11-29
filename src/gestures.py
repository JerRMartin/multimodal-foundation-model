import cv2
import config as C
from src.helpers import count_oscillations, show_delay
from collections import deque
import pygame
from src.xbox_controller import XboxController
import os

# --------- State ----------
xs = deque(maxlen=C.WINDOW_SIZE)
ys = deque(maxlen=C.WINDOW_SIZE)
detected_gestures = set()
nod_delay, shake_delay = (C.DETECT_DELAY,)*2
gaze_delay = C.GAZE_DELAY

# Load OpenCV's built-in Haar cascade for faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions/device index.")

print("[o] Running head movement detection. Press 'q' to quit.")

# Initialize mixer ONCE before webcam loop (this esnures no delays on sound playback) 
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Initialize Xbox Controller for Haptic Feedback
if os.getenv("CONTROL", "0").strip() == "True":
    controller = XboxController(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xbox Controller Polling
    if os.getenv("CONTROL", "0").strip() == "True":
        controller.poll()

    # Flip for natural webcam view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (we use the first one)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    # Default status
    status_text = "No Gaze Detected"
    status_color = C.COLORS['RED']

    if len(faces) > 0: # Face detected
        status_text = "Detecting Gaze..."
        status_color = C.COLORS["YELLOW"]
        # Xbox Controller Rumble while face detected
        if os.getenv("CONTROL", "0").strip() == "True":
            controller.rumble(C.LEFT_VIBE, C.RIGHT_VIBE, duration=0.01)
        
        (x, y, w, h) = faces[0]
        cx = x + w // 2
        cy = y + h // 2

        xs.append(cx)
        ys.append(cy)

        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), C.COLORS["YELLOW"], 2)
        cv2.circle(frame, (cx, cy), 4, C.COLORS["YELLOW"], -1)

        if len(xs) == C.WINDOW_SIZE:
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)

            osc_x = count_oscillations(xs)
            osc_y = count_oscillations(ys)

            # Vertical → nod
            if dy > C.NOD_THRESHOLD and dy > dx and osc_y >= C.MIN_OSCILLATIONS and nod_delay == C.DETECT_DELAY and gaze_delay <= 0:
                detected_gestures.add("NOD")
                pygame.mixer.Sound(C.NOD_WAV).play()
            # Horizontal → shake
            elif dx > C.SHAKE_THRESHOLD and dx > dy and osc_x >= C.MIN_OSCILLATIONS and shake_delay == C.DETECT_DELAY and gaze_delay <= 0:
                detected_gestures.add("SHAKE")
                pygame.mixer.Sound(C.SHAKE_WAV).play()

        gaze_delay -= 1
    else:
        gaze_delay = C.GAZE_DELAY

    # Show status
    cv2.putText( frame, status_text, (350, 30), C.FONT_TYPE, 1, status_color, 2, cv2.LINE_AA)

# ============================== DEBUG INFO ======================================
    cv2.putText(frame, "Cues Detected:", (10, 30), C.FONT_TYPE, 1, C.COLORS["BLUE"], 2, cv2.LINE_AA)

# ---------------- Nod delay handling ----------------
    if len(detected_gestures) > 0 and "NOD" in detected_gestures and nod_delay > 0:
        # Show nod timeout
        show_delay(frame, "NOD", nod_delay, (10, 60))
        nod_delay -= 1
    elif nod_delay <= 0: # Reset nod detection
        detected_gestures.remove("NOD")
        nod_delay = C.DETECT_DELAY

    # ---------------- Shake delay handling ----------------
    if len(detected_gestures) > 0 and "SHAKE" in detected_gestures and shake_delay > 0: 
        # Show shake timeout
        show_delay(frame, "SHAKE", shake_delay, (10, 90))
        shake_delay -= 1
    elif shake_delay <= 0: # Reset shake detection
        detected_gestures.remove("SHAKE")
        shake_delay = C.DETECT_DELAY
    
    cv2.imshow("Head movement detection (OpenCV only)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
