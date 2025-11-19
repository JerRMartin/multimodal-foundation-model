import cv2
import numpy as np
from collections import deque

# --------- Configuration Thresholds ----------
WINDOW_SIZE = 15          # regarding frames in history
NOD_THRESHOLD = 5        # regarding vertical movement in pixels
SHAKE_THRESHOLD = 5      # regarding horizontal movement in pixels
MIN_OSCILLATIONS = 3      # how many direction changes to call a gesture

# --------- Constants ----------
DETECT_DELAY = 100        # seconds to wait after a detection

# ---- Colors ----
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

# --------- Helpers ----------
def count_oscillations(signal):
    """
    Count how many times the movement changes direction.
    signal: iterable of numbers (x or y positions over time).
    """
    if len(signal) < 3:
        return 0

    diffs = np.diff(signal).astype(float)
    
    # Filter out tiny jitters
    diffs[np.abs(diffs) < 1.0] = 0.0

    signs = np.sign(diffs)
    signs = signs[signs != 0]  # remove zeros (no movement)

    if len(signs) < 2:
        return 0

    # Count sign changes
    return int(np.sum(signs[1:] * signs[:-1] < 0))

def show_delay(frame, text, delay, position):
    cv2.putText(
        frame,
        f'{text}: {str(delay)}',
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR_GREEN,
        2,
        cv2.LINE_AA,
    )

# --------- State ----------
xs = deque(maxlen=WINDOW_SIZE)
ys = deque(maxlen=WINDOW_SIZE)
detected_gestures = set()
nod_delay, shake_delay = (DETECT_DELAY,)*2



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
    status_color = COLOR_RED

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cx = x + w // 2
        cy = y + h // 2

        xs.append(cx)
        ys.append(cy)

        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_YELLOW, 2)
        cv2.circle(frame, (cx, cy), 4, COLOR_YELLOW, -1)

        if len(xs) == WINDOW_SIZE:
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)

            osc_x = count_oscillations(xs)
            osc_y = count_oscillations(ys)


            # TODO: Do we want to add a cooldown after gaze is detected, to prevent false positives?
            status_text = "Detecting Gaze..."
            status_color = COLOR_YELLOW
            # Vertical → nod
            if dy > NOD_THRESHOLD and dy > dx and osc_y >= MIN_OSCILLATIONS and nod_delay == DETECT_DELAY:
                detected_gestures.add("NOD")
            # Horizontal → shake
            elif dx > SHAKE_THRESHOLD and dx > dy and osc_x >= MIN_OSCILLATIONS and shake_delay == DETECT_DELAY:
                detected_gestures.add("SHAKE")


    # Show status
    cv2.putText(
        frame,
        status_text,
        (350, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
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
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        COLOR_BLUE,
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
        nod_delay = DETECT_DELAY

    # ---------------- Shake delay handling ----------------
    if len(detected_gestures) > 0 and "SHAKE" in detected_gestures and shake_delay > 0: 
        # Show shake timeout
        show_delay(frame, "SHAKE", shake_delay, (10, 90))
        shake_delay -= 1
    elif shake_delay <= 0: # Reset shake detection
        detected_gestures.remove("SHAKE")
        shake_delay = DETECT_DELAY
    
    cv2.imshow("Head movement detection (OpenCV only)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
