from flask import (
    Flask, render_template
    )
import cv2
import mediapipe as mp

# Flask app
app = Flask(
    __name__,
)

# Video and hand tracking setup
# video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Views 
@app.route("/home", methods = ['GET'])
def home_page():
    return render_template("detect.html")


# Get frame from camera 
def gen(video):
    while True: # Infinite loop
        succes, frame = video.read()
        if not succes:
            continue  # Ignore empty camera frame
        # Inference
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        # Do nothing if hand not detected
        if not results.multi_hand_landmarks:
            continue
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())



