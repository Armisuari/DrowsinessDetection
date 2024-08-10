#!/usr/bin/env python3

import cv2
from drowsiness_detection import DrowsinessDetector
from threading import Thread, Lock
import time
import RPi.GPIO as GPIO

# GPIO setup for buzzer
BUZZER_PIN = 17  # GPIO pin connected to the buzzer
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off initially

# GPIO setup for servo
SERVO_PIN = 24  # GPIO pin connected to the servo
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
servo.start(7.5)  # Neutral position

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = Lock()
        self.frame = None
        self.stopped = False
        self.detector = DrowsinessDetector()
        self.buzzer_blinking = False  # Flag to control buzzer blinking
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.servo_angle = 90  # Initialize servo angle at the center

    def update(self):
        while not self.stopped:
            ret, frame = self.video.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (720, 480))
            processed_frame, alert, face_position = self.detector.process_frame(frame)
            
            with self.lock:
                self.frame = processed_frame
                
            # Trigger buzzer if alert is true
            if alert and not self.buzzer_blinking:
                self.buzzer_blinking = True
                Thread(target=self.blink_buzzer).start()
            elif not alert:
                self.buzzer_blinking = False
                GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer off
            
            # Adjust servo based on face position
            if face_position is not None:
                self.adjust_servo(face_position[0], frame.shape[1])
            
            time.sleep(0.033)  # Limit to ~30 FPS

    def blink_buzzer(self):
        while self.buzzer_blinking:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.1)  # Buzzer on for 100 ms
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.1)  # Buzzer off for 100 ms

    def adjust_servo(self, face_x, frame_width):
        # Calculate the deviation of the face from the center
        error = face_x - frame_width / 2
        
        # Proportional control: Adjust the servo angle based on the error
        kp = 0.1  # Proportional gain (tune this value as needed)
        adjustment = kp * error
        self.servo_angle -= adjustment
        
        # Clamp the servo angle to the valid range (0 to 180 degrees)
        self.servo_angle = max(0, min(180, self.servo_angle))
        
        # Convert the angle to duty cycle
        duty_cycle = 2.5 + (self.servo_angle / 18)
        servo.ChangeDutyCycle(duty_cycle)

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
        servo.stop()
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
