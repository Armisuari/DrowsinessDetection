#!/usr/bin/env python3

import cv2
from drowsiness_detection import DrowsinessDetector
from threading import Thread, Lock
import time
import RPi.GPIO as GPIO  # Import RPi.GPIO for GPIO control

# GPIO setup
BUZZER_PIN = 17  # GPIO pin connected to the buzzer
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off initially

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
            # frame = cv2.resize(frame, (320, 240))
            frame = cv2.resize(frame, (720, 480))
            processed_frame, alert = self.detector.process_frame(frame)
            
            with self.lock:
                self.frame = processed_frame
                
            # Trigger buzzer if alert is true
            if alert:
                GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer on
            else:
                GPIO.output(BUZZER_PIN, GPIO.LOW)   # Turn buzzer off
            
            time.sleep(0.033)  # Limit to ~30 FPS

    def read(self):
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()
            else:
                frame = None
        return frame

    def __del__(self):
        self.stopped = True
        self.video.release()
        GPIO.cleanup()  # Clean up GPIO settings

camera = VideoCamera()

def display_frames():
    while True:
        frame = camera.read()
        if frame is not None:
            cv2.imshow('Drowsiness Detection', frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    try:
        display_frames()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        del camera
        GPIO.cleanup()  # Ensure GPIO is cleaned up on exit
