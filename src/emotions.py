import cv2
import time
import torch
import mediapipe as mp
import numpy as np
import src.configs as C
from PIL import Image
from torchvision import transforms
from src.helpers import get_box

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

def display_EMO_PRED(img, box, label='', color=C.COLORS["GRAY"], txt_color=C.COLORS["WHITE"], line_width=2, ):
    lw = line_width or max(round(sum(img.shape) / 2 * 0.003), 2)

    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, C.COLORS["PURPLE"], thickness=lw, lineType=cv2.LINE_AA)

    ft = max(lw - 1, 1)
    text_width_2, text_height_2 = cv2.getTextSize(label, C.FONT_TYPE, lw / 3, ft)
    text_width_2 = text_width_2[0] + round(((p2[0] - p1[0]) * 10) / 360)
    center_face = p1[0] + round((p2[0] - p1[0]) / 2)

    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), C.FONT_TYPE,
                lw / 3, C.COLORS["BLACK"], thickness=ft, lineType=cv2.LINE_AA)
    cv2.putText(img, label,
                (center_face - round(text_width_2 / 2), p1[1] - round(((p2[0] - p1[0]) * 20) / 360)), C.FONT_TYPE,
                lw / 3, C.COLORS["PURPLE"], thickness=ft, lineType=cv2.LINE_AA)
    return img

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
                color=C.COLORS["BLACK"],
                thickness=thickness,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)

    return img

mp_face_mesh = mp.solutions.face_mesh

torch_model = torch.jit.load(C.PYTORCH_MODEL_PATH).to('cuda')
torch_model.eval()

cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = np.round(cap.get(cv2.CAP_PROP_FPS))

path_save_video = 'result.mp4'
vid_writer = cv2.VideoWriter(path_save_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
with mp_face_mesh.FaceMesh(
max_num_faces=1,
refine_landmarks=False,
min_detection_confidence=0.5,
min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        t1 = time.time()
        success, frame = cap.read()
        if frame is None: break

        frame_copy = frame.copy()
        frame_copy.flags.writeable = False
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_copy)
        frame_copy.flags.writeable = True
        
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                startX, startY, endX, endY  = get_box(fl, w, h)
                cur_face = frame_copy[startY:endY, startX: endX]

                cur_face = pth_processing(Image.fromarray(cur_face))
                output = torch.nn.functional.softmax(torch_model(cur_face), dim=1).cpu().detach().numpy()
                
                cl = np.argmax(output)
                label = C.DICT_EMO[cl]
                frame = display_EMO_PRED(frame, (startX, startY, endX, endY), label, line_width=3)

        t2 = time.time()

        frame = display_FPS(frame, 'FPS: {0:.1f}'.format(1 / (t2 - t1)), box_scale=.5)

        vid_writer.write(frame)
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()

