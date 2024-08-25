#!/usr/bin/env python3

import cv2
from drowsiness_detection import DrowsinessDetector
from threading import Thread, Lock
import time
import RPi.GPIO as GPIO

# Pengaturan GPIO untuk buzzer
BUZZER_PIN = 17  # Pin GPIO yang terhubung ke buzzer
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)  # Pastikan buzzer mati pada awalnya

# Pengaturan GPIO untuk servo
SERVO_PIN = 24  # Pin GPIO yang terhubung ke servo
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # Frekuensi PWM 50Hz
servo.start(7.5)  # Posisi netral servo

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.lock = Lock()  # Kunci untuk memastikan thread aman saat mengakses frame
        self.frame = None
        self.stopped = False
        self.detector = DrowsinessDetector()  # Inisialisasi detektor kantuk
        self.buzzer_blinking = False  # Flag untuk mengontrol blinking buzzer
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True  # Set thread sebagai daemon agar berhenti saat program berhenti
        self.thread.start()
        self.servo_angle = 90  # Inisialisasi sudut servo pada posisi tengah

    def update(self):
        while not self.stopped:
            ret, frame = self.video.read()  # Membaca frame dari kamera
            if not ret:
                continue
            
            frame = cv2.resize(frame, (720, 480))  # Resize frame ke resolusi 720x480
            processed_frame, alert, face_position = self.detector.process_frame(frame)
            
            with self.lock:
                self.frame = processed_frame  # Menyimpan frame yang sudah diproses
            
            # Mengaktifkan buzzer jika alert bernilai True
            if alert and not self.buzzer_blinking:
                self.buzzer_blinking = True
                Thread(target=self.blink_buzzer).start()
            elif not alert:
                self.buzzer_blinking = False
                GPIO.output(BUZZER_PIN, GPIO.LOW)  # Mematikan buzzer
            
            # Menyesuaikan posisi servo berdasarkan posisi wajah
            if face_position is not None:
                self.adjust_servo(face_position[0], frame.shape[1])
            
            time.sleep(0.033)  # Membatasi kecepatan pembaruan frame ke ~30 FPS

    def blink_buzzer(self):
        while self.buzzer_blinking:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(0.1)  # Buzzer menyala selama 100 ms
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(0.1)  # Buzzer mati selama 100 ms

    def adjust_servo(self, face_x, frame_width):
        # Menghitung deviasi wajah dari pusat frame
        error = face_x - frame_width / 2
        
        # Kontrol proporsional: Sesuaikan sudut servo berdasarkan error
        kp = 0.1  # Gain proporsional (atur nilai ini sesuai kebutuhan)
        adjustment = kp * error
        self.servo_angle -= adjustment
        
        # Batasi sudut servo pada rentang yang valid (0 hingga 180 derajat)
        self.servo_angle = max(0, min(180, self.servo_angle))
        
        # Konversi sudut menjadi duty cycle
        duty_cycle = 2.5 + (self.servo_angle / 18)
        servo.ChangeDutyCycle(duty_cycle)

    def read(self):
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()  # Mengambil salinan frame yang ada
            else:
                frame = None
        return frame

    def __del__(self):
        self.stopped = True
        self.video.release()  # Membebaskan sumber daya video
        servo.stop()  # Menghentikan PWM servo
        GPIO.cleanup()  # Membersihkan pengaturan GPIO

camera = VideoCamera()

def display_frames():
    while True:
        frame = camera.read()
        if frame is not None:
            cv2.imshow('Drowsiness Detection', frame)  # Menampilkan frame di jendela
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Keluar dari loop jika tombol 'q' ditekan

if __name__ == '__main__':
    try:
        display_frames()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()  # Menutup semua jendela OpenCV
        del camera  # Menghapus objek camera dan membersihkan GPIO
        GPIO.cleanup()  # Pastikan GPIO dibersihkan saat keluar dari program
