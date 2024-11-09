import streamlit as st
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import time
from pathlib import Path
import vlc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the trained model (from .h5)
model = tf.keras.models.load_model('HandModel.h5')

# Set confidence threshold
confidence_threshold = 0.8  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Labels for hand gestures
labels_dict = {0: 'A', 1: 'B', 2: 'N-1', 3: 'N-2', 4: 'Lund', 5: 'Fuck'}  
prediction_sequence = []

sequence_actions = {
    ('A', 'B'): "Alphabet",
    ('N-1', 'N-2'): "Namaste",
    ('Fuck', 'Lund'): "sex"
}

# Function to play sound
def play_sound(file_path):
    audio_path = Path(file_path)
    if not audio_path.is_file():
        print("Error: File does not exist.")
        return

    try:
        player = vlc.MediaPlayer(str(audio_path))
        player.play()
        time.sleep(2)
        while player.is_playing():
            time.sleep(0.1)
    except Exception as e:
        print(f"Error playing sound: {e}")


# Streamlit UI setup
st.title('Hand Gesture Recognition')
st.write("This app recognizes hand gestures using your webcam.")

# Custom video transformer class for Streamlit WebRTC
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.consecutive_frames = 0
        self.last_predicted_character = None

    def transform(self, frame):
        # Convert frame to RGB
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand = []
        right_hand = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                landmarks_normalized = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    landmarks_normalized.append(x - min(x_))
                    landmarks_normalized.append(y - min(y_))

                hand_label = results.multi_handedness[hand_idx].classification[0].label
                if hand_label == 'Left':
                    left_hand = landmarks_normalized
                else:
                    right_hand = landmarks_normalized

                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        if not left_hand:
            left_hand = [0] * 42
        if not right_hand:
            right_hand = [0] * 42

        data_aux = left_hand + right_hand

        if len(data_aux) == 84:  # If the hand landmarks are ready
            data_aux = np.asarray(data_aux).reshape(1, -1)  # Reshape for model input
            prediction = model.predict(data_aux)  # Predict using the deep learning model
            predicted_label = np.argmax(prediction, axis=1)[0]  # Get the index with the highest probability
            confidence = np.max(prediction)  # Get the confidence of the prediction

            if confidence >= confidence_threshold and predicted_label in labels_dict:
                predicted_character = labels_dict[predicted_label]

                if predicted_character == self.last_predicted_character:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 1

                if self.consecutive_frames >= 2:
                    if not prediction_sequence or prediction_sequence[-1] != predicted_character:
                        prediction_sequence.append(predicted_character)

                        if len(prediction_sequence) >= 2:
                            last_two = tuple(prediction_sequence[-2:])
                            if last_two in sequence_actions:
                                action = sequence_actions[last_two]
                                st.write(f"Detected gesture sequence: {action}")
                                threading.Thread(target=play_sound, args=(f'VoiceCacheMemory/{action}.mp3',)).start()

                self.last_predicted_character = predicted_character

                if x_ and y_:
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return img


# Start the Streamlit WebRTC video stream
webrtc_streamer(key="hand-gesture-recognition", video_transformer_factory=HandGestureTransformer)
