import cv2
import src.configs as C
from src.helpers import count_oscillations, show_delay
from collections import deque
import pygame

# --------- State ----------
xs = deque(maxlen=C.WINDOW_SIZE)
ys = deque(maxlen=C.WINDOW_SIZE)
detected_gestures = set()
nod_delay, shake_delay = (C.DETECT_DELAY,)*2

# Load OpenCV's built-in Haar cascade for faces
#face_cascade = cv2.CascadeClassifier(
#    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
#)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions/device index.")

print("[o] Running head movement detection. Press 'q' to quit.")

# Initialize mixer ONCE before webcam loop (this esnures no delays on sound playback) 
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    status_text = "No Gaze Detected"
    status_color = C.COLOR_RED

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cx = x + w // 2
        cy = y + h // 2

        xs.append(cx)
        ys.append(cy)

        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), C.COLOR_YELLOW, 2)
        cv2.circle(frame, (cx, cy), 4, C.COLOR_YELLOW, -1)

        if len(xs) == C.WINDOW_SIZE:
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)

            osc_x = count_oscillations(xs)
            osc_y = count_oscillations(ys)


            # TODO: Do we want to add a cooldown after gaze is detected, to prevent false positives?
            status_text = "Detecting Gaze..."
            status_color = C.COLOR_YELLOW
            # Vertical → nod
            if dy > C.NOD_THRESHOLD and dy > dx and osc_y >= C.MIN_OSCILLATIONS and nod_delay == C.DETECT_DELAY:
                detected_gestures.add("NOD")
                pygame.mixer.Sound(C.NOD_WAV).play()
            # Horizontal → shake
            elif dx > C.SHAKE_THRESHOLD and dx > dy and osc_x >= C.MIN_OSCILLATIONS and shake_delay == C.DETECT_DELAY:
                detected_gestures.add("SHAKE")
                pygame.mixer.Sound(C.SHAKE_WAV).play()

    # Show status
    cv2.putText(
        frame,
        status_text,
        (350, 30),
        C.FONT_TYPE,
        1,
        status_color,
        2,
        cv2.LINE_AA,
    )

# ============================== DEBUG INFO ======================================
    cv2.putText(
        frame,
        "Cues Detected:",
        (10, 30),
        C.FONT_TYPE,
        1,
        C.COLOR_BLUE,
        2,
        cv2.LINE_AA,
        )

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
