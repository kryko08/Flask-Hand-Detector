from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    url_for
    )
import cv2
import mediapipe as mp

# Flask app
app = Flask(
    __name__,
)

# Video and hand tracking setup
video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

is_winner = 0

# Get frame from camera 
def gen_f(video):
    while True: # Infinite loop
        succes, frame = video.read()
        height, width, channels = frame.shape
        if not succes:
            continue  # Ignore empty camera frame

        # Inference
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        # Do nothing if hand not detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand point with connections 
                mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1)) # Flip the image horizontally
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Views 
@app.route('/video_feed')
def video_feed():
    return Response(gen_f(video=video), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/home", methods = ['GET'])
def home_page():
    return render_template("detect.html")
