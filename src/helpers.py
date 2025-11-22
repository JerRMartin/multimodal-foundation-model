import cv2
import math
import numpy as np
import configs as C

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
        C.FONT_TYPE,
        1,
        C.COLOR_GREEN,
        2,
        cv2.LINE_AA,
    )

def normalize_coordinates(normalized_x, normalized_y, image_width, image_height):
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    
    return x_px, y_px

def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = normalize_coordinates(landmark.x, landmark.y, w, h)

        if landmark_px:
            idx_to_coors[idx] = landmark_px

    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])

    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
    return startX, startY, endX, endY