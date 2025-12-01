import cv2
import time
import torch
import mediapipe as mp
import numpy as np
import pygame
from collections import deque
from src.helpers import get_box, show_delay, count_oscillations, display_EMO_PRED, display_FPS, pth_processing
from src.xbox_controller import XboxController
import os
import config as C
from PIL import Image

# Global setup
mp_face_mesh = mp.solutions.face_mesh
torch_model = torch.jit.load(C.PYTORCH_MODEL_PATH).to('cuda')
torch_model.eval()

class HeadGestureDetector:
    def __init__(self, control = False):
        # Initialize state variables
        self.position_history_x = deque(maxlen=C.WINDOW_SIZE)
        self.position_history_y = deque(maxlen=C.WINDOW_SIZE)
        self.detected_gestures = set()        # For NOD and SHAKE
        self.nod_delay = C.DETECT_DELAY
        self.shake_delay = C.DETECT_DELAY
        self.gaze_delay = C.GAZE_DELAY
        
        # Emotion delay system
        self.detected_emotions = set()        # Currently active emotions
        self.emotion_delays = {}              # e.g., {"Happy": 30, "Sad": 15}
        self.last_detected_emotion = None     # To prevent repeat triggers
        
        # Store emotion detection bounding box for movement tracking
        self.current_emo_box = None
        
        # Initialize components
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # Initialize controller only if explicitly requested
        self.controller = None
        if control:
            try:
                self.controller = XboxController(0)
            except Exception as e:
                print(f"Warning: Failed to initialize Xbox controller: {e}")
                self.controller = None

    def process_frame(self, frame, face_mesh):
        """Process a single frame for emotion detection and return the processed frame."""
        frame_copy = cv2.flip(frame.copy(), 1)
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        frame_copy.flags.writeable = False
        results = face_mesh.process(frame_copy)
        frame_copy.flags.writeable = True
        
        self.current_emo_box = None
        current_emotion_label = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                startX, startY, endX, endY = get_box(face_landmarks, frame.shape[1], frame.shape[0])
                face_region = frame_copy[startY:endY, startX:endX]
                processed_face = pth_processing(Image.fromarray(face_region))
                output = torch.nn.functional.softmax(torch_model(processed_face), dim=1).cpu().detach().numpy()
                
                emotion_class = np.argmax(output)
                emotion_label = C.DICT_EMO[emotion_class]
                current_emotion_label = emotion_label
                
                frame = display_EMO_PRED(frame, (startX, startY, endX, endY), emotion_label, line_width=3)
                self.current_emo_box = (startX, startY, endX, endY)
        
        # Trigger emotion with delay logic
        if current_emotion_label and current_emotion_label != self.last_detected_emotion:
            if current_emotion_label not in self.emotion_delays or self.emotion_delays[current_emotion_label] <= 0:
                self.detected_emotions.add(current_emotion_label)
                self.emotion_delays[current_emotion_label] = C.DETECT_DELAY
                self.last_detected_emotion = current_emotion_label
                
                # Play emotion-specific sound if it exists in config
                emo_audio = getattr(C.AUDIO, current_emotion_label.upper(), None)
                if emo_audio:
                    audio = pygame.mixer.Sound(emo_audio.file)
                    audio.set_volume(emo_audio.volume)
                    audio.play()
        
        return frame, results
    
    def update_gaze_tracking(self):
        if self.current_emo_box is not None:
            startX, startY, endX, endY = self.current_emo_box
            center_x = startX + (endX - startX) // 2
            center_y = startY + (endY - startY) // 2
            self.position_history_x.append(center_x)
            self.position_history_y.append(center_y)
            return True
        return False
    
    def detect_gestures(self):
        if len(self.position_history_x) == C.WINDOW_SIZE:
            x_range = max(self.position_history_x) - min(self.position_history_x)
            y_range = max(self.position_history_y) - min(self.position_history_y)
            x_osc = count_oscillations(self.position_history_x)
            y_osc = count_oscillations(self.position_history_y)
            
            if (y_range > C.NOD_THRESHOLD and y_range > x_range and
                y_osc >= C.MIN_OSCILLATIONS and
                self.nod_delay == C.DETECT_DELAY and self.gaze_delay <= 0):
                self.trigger_gesture("Nod", C.AUDIO.NOD)
            
            elif (x_range > C.SHAKE_THRESHOLD and x_range > y_range and
                  x_osc >= C.MIN_OSCILLATIONS and
                  self.shake_delay == C.DETECT_DELAY and self.gaze_delay <= 0):
                self.trigger_gesture("Shake", C.AUDIO.SHAKE)
    
    def trigger_gesture(self, gesture_type, audio_config):
        self.detected_gestures.add(gesture_type)
        audio = pygame.mixer.Sound(audio_config.file)
        audio.set_volume(audio_config.volume)
        audio.play()
    
    def update_delays(self, frame):
        # --- Update Nod / SHAKE delays ---
        for gesture in list(self.detected_gestures):
            delay = self.nod_delay if gesture == "Nod" else self.shake_delay
            y_pos = 60 if gesture == "Nod" else 90
            if delay > 0:
                show_delay(frame, gesture, delay, (10, y_pos))
                if gesture == "Nod":
                    self.nod_delay -= 1
                else:
                    self.shake_delay -= 1
            else:
                self.detected_gestures.discard(gesture)
                if gesture == "Nod":
                    self.nod_delay = C.DETECT_DELAY
                else:
                    self.shake_delay = C.DETECT_DELAY
        
       # Update EMOTION delays (excluding happiness and neutral)
        emotions_to_delay = {"happiness", "neutral"}
        y_offset = 120
        
        for emotion in list(self.detected_emotions):
            # Only process emotions that should have delays
            if emotion.lower() in emotions_to_delay:
                # Remove happiness and neutral from the delay system if they were somehow added
                self.detected_emotions.discard(emotion)
                self.emotion_delays.pop(emotion, None)
                continue
                
            delay = self.emotion_delays.get(emotion, 0)
            if delay > 0:
                show_delay(frame, emotion, delay, (10, y_offset))
                self.emotion_delays[emotion] = delay - 1
                y_offset += 30
            else:
                self.detected_emotions.discard(emotion)
                self.emotion_delays.pop(emotion, None)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame, results = self.process_frame(frame, face_mesh)
                faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                
                if len(faces) > 0:
                    status_text = "Detecting Gaze..."
                    status_color = C.COLORS.YELLOW
                    self.gaze_delay -= 1
                    
                    if self.controller:
                        self.controller.poll()
                        self.controller.rumble(C.LEFT_VIBE, C.RIGHT_VIBE, duration=0.01)
                    
                    if self.current_emo_box is not None:
                        self.update_gaze_tracking()
                        self.detect_gestures()
                else:
                    status_text = "No Gaze Detected"
                    status_color = C.COLORS.RED
                    self.gaze_delay = C.GAZE_DELAY
                
                self.update_delays(frame)
                cv2.putText(frame, status_text, (350, 30), C.FONT_TYPE, 1, status_color, 2, cv2.LINE_AA)
                cv2.putText(frame, "Cues Detected:", (10, 30), C.FONT_TYPE, 1, C.COLORS.BLUE, 2, cv2.LINE_AA)
                
                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()