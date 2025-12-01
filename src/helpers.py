import cv2
import math
import numpy as np
import config as C
import torch 
from torchvision import transforms
from PIL import Image

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
        C.COLORS.GREEN,
        2,
        cv2.LINE_AA,
    )

def normalize_coordinates(normalized_x, normalized_y, image_width, image_height):
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    
    return x_px, y_px

def get_box(fl, w, h):
    """Get bounding box coordinates from face landmarks and convert them to the original frame's coordinate system."""
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = normalize_coordinates(landmark.x, landmark.y, w, h)
        if landmark_px:
            idx_to_coors[idx] = landmark_px
    
    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])
    
    # Convert coordinates from flipped space back to original frame space
    # Since the frame was flipped horizontally, we need to mirror the x-coordinates
    startX_flipped = max(0, x_min)
    endX_flipped = min(w - 1, endX)
    
    # Mirror the x-coordinates to match the original frame
    startX = w - endX_flipped      # Left edge becomes right edge after mirroring
    endX = w - startX_flipped    # Right edge becomes left edge after mirroring
    startY = max(0, y_min)       # Y coordinates remain the same
    endY = min(h - 1, endY)
    
    return startX, startY, endX, endY

def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001)  # line width
    thickness = max(int(line_width / 3), 1)  # font thickness

    font_scale = thickness / 1.5

    t_w, t_h = cv2.getTextSize(text, C.FONT_TYPE, font_scale, None)[0]

    margin_n = int(t_h * margin)
    sub_img = img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
              img_w - t_w - margin_n - int(2 * t_h * box_scale): img_w - margin_n]

    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    img[0 + margin_n: 0 + margin_n + t_h + int(2 * t_h * box_scale),
    img_w - t_w - margin_n - int(2 * t_h * box_scale):img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, .5,
                                                                                          1.0)

    cv2.putText(img=img,
                text=text,
                org=(img_w - t_w - margin_n - int(2 * t_h * box_scale) // 2,
                     0 + margin_n + t_h + int(2 * t_h * box_scale) // 2),
                fontFace=C.FONT_TYPE,
                fontScale=font_scale,
                color=C.COLORS.BLACK,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)

    return img

def display_EMO_PRED(img, box, label='', color=C.COLORS.GRAY, txt_color=C.COLORS.WHITE, line_width=2, ):
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, C.COLORS.PURPLE, thickness=lw, lineType=cv2.LINE_AA)

    ft = max(lw - 1, 1)
    text_width_2, text_height_2 = cv2.getTextSize(label, C.FONT_TYPE, lw / 3, ft)
    text_width_2 = text_width_2[0] + round(((p2[0] - p1[0]) * 10) / 360)
    center_face = p1[0] + round((p2[0] - p1[0]) / 2)

    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), C.FONT_TYPE,
                lw / 3, C.COLORS.BLACK, thickness=ft, lineType=cv2.LINE_AA)
    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), C.FONT_TYPE,
                lw / 3, C.COLORS.PURPLE, thickness=ft, lineType=cv2.LINE_AA)
    return img

def pth_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img):
        
        ttransform = transforms.Compose([
            transforms.PILToTensor(),
            PreprocessInput()
        ])
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = ttransform(img)
        img = torch.unsqueeze(img, 0).to('cuda')
        return img
    return get_img_torch(fp)