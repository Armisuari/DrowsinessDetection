from flask import Flask, render_template, Response
import cv2
from drowsiness_detection import DrowsinessDetector
from threading import Thread, Lock
import time

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = Lock()
        self.frame = None
        self.stopped = False
        self.detector = DrowsinessDetector()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.video.read()
            if not ret:
                continue
            
            # Reduce the frame size for better performance
            frame = cv2.resize(frame, (320, 240))
            processed_frame = self.detector.process_frame(frame)
            
            with self.lock:
                self.frame = processed_frame
            
            time.sleep(0.033)  # Limit to ~30 FPS

    def read(self):
        with self.lock:
            frame = self.frame.copy()
        return frame

    def __del__(self):
        self.stopped = True
        self.video.release()

app = Flask(__name__)
camera = VideoCamera()

def generate_frames():
    while True:
        frame = camera.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
