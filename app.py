from flask import Flask, render_template, Response
import cv2
from drowsiness_detection import DrowsinessDetector

app = Flask(__name__)

detector = DrowsinessDetector()

def generate_frames():
    vs = cv2.VideoCapture(0)
    
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = detector.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    vs.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
