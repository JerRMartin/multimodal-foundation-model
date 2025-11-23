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
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GRAY = (128, 128, 128)
COLOR_WHITE = (255, 255, 255)
COLOR_PURPLE = (255, 0, 255)
COLOR_BLACK = (0, 0, 0)

# ---- Typeface ----
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

# --------- Constants ----------
DETECT_DELAY = 100        # seconds to wait after a detection

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
