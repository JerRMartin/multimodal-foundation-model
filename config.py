# configs.py
import cv2
from dataclasses import dataclass

DICT_EMO = {
    0: 'Neutral',
    1: 'Happiness', # haptics
    2: 'Sadness', # haptics
    3: 'Surprise', # audio cues
    6: 'Anger' # audio cues
}
# ---- Colors ----
@dataclass
class COLORS():
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    GRAY = (128, 128, 128)
    WHITE = (255, 255, 255)
    PURPLE = (255, 0, 255)
    BLACK = (0, 0, 0)

# ---- Typeface ----
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

# --------- Constants ----------
DETECT_DELAY = 100        # seconds to wait AFTER a gesture detection
GAZE_DELAY = 15           # seconds to wait BEFORE detecting gestures

# --------- Configuration Thresholds ----------
WINDOW_SIZE = 15          # regarding frames in history
NOD_THRESHOLD = 7        # regarding vertical movement in pixels
SHAKE_THRESHOLD = 2      # regarding horizontal movement in pixels
MIN_OSCILLATIONS = 3      # how many direction changes to call a gesture

# ---- PyTorch Model Path ----
PYTORCH_MODEL_PATH = 'models/torchscript_model_0_66_49_wo_gl.pth'

class _Clip():
    def __init__(self, file, volume):
        self.file: str = file
        self.volume: float = volume

@dataclass
class AUDIO():
    NOD: _Clip = _Clip('audio/Nod.mp3', 0.25)
    SHAKE: _Clip = _Clip('audio/Shake.mp3', 0.35)
    SURPRISE: _Clip = _Clip('audio/Surprise.mp3', 0.50)
    ANGER: _Clip = _Clip('audio/Anger.mp3', 0.50)

# ------- Controller Constants -------
LEFT_VIBE = 0.25
RIGHT_VIBE = 0.20