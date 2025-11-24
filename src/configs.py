# configs.py
import cv2
import simpleaudio as sa

DICT_EMO = {
    0: 'Neutral', 
    1: 'Happiness', 
    2: 'Sadness', 
    3: 'Surprise', 
    4: 'Fear', 
    5: 'Disgust', 
    6: 'Anger'
}
# ---- Colors ----
COLORS = {
    'RED': (0, 0, 255),
    'YELLOW': (0, 255, 255),
    'GREEN': (0, 255, 0),
    'BLUE': (255, 0, 0),
    'GRAY': (128, 128, 128),
    'WHITE': (255, 255, 255),
    'PURPLE': (255, 0, 255),
    'BLACK': (0, 0, 0),
}

# ---- Typeface ----
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

# --------- Constants ----------
DETECT_DELAY = 100        # seconds to wait AFTER a gesture detection
GAZE_DELAY = 15           # seconds to wait BEFORE detecting gestures

# --------- Configuration Thresholds ----------
WINDOW_SIZE = 15          # regarding frames in history
NOD_THRESHOLD = 15        # regarding vertical movement in pixels
SHAKE_THRESHOLD = 15      # regarding horizontal movement in pixels
MIN_OSCILLATIONS = 2      # how many direction changes to call a gesture

# ---- PyTorch Model Path ----
PYTORCH_MODEL_PATH = 'models/torchscript_model_0_66_49_wo_gl.pth'

# --------- Audio Objects ----------
NOD_WAV = 'audio/Villager_Nod.wav'
SHAKE_WAV = 'audio/Villager_Shake.wav'

# ------- Controller Constants -------
LEFT_VIBE = 0.25
RIGHT_VIBE = 0.20