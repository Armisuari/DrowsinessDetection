import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import time

class DrowsinessDetector:
    def __init__(self):
        # Threshold dan frame limit untuk mendeteksi kantuk
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 5
        self.COUNTER = 0  # Counter untuk menghitung berapa kali EAR di bawah threshold
        self.COUNTER_AFTER_LIMIT = 15  # Limit setelah berapa banyak frame alarm menyala
        self.COUNTER_AFTER = 0  # Counter setelah kondisi kantuk terdeteksi
        self.COUNTER_OPEN = 0  # Counter untuk menghitung berapa kali EAR di atas threshold
        self.ALARM_ON = False  # Status apakah alarm menyala
        self.last_frame_time = time.time()
        self.frame_interval = 0.1  # Interval untuk memproses frame setiap 100ms (10 FPS)

        print("[INFO] loading facial landmark predictor...")
        # Memuat detektor wajah dan predictor landmark wajah
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/home/comitup/DrowsinessDetection/shape_predictor_68_face_landmarks.dat")
        
        # Mendapatkan indeks untuk mata kiri dan kanan
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        # Menghitung rasio aspek mata (EAR) berdasarkan jarak Euclidean antara titik landmark
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)  # Rasio aspek mata
        return ear

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return frame, False, None  # Mengembalikan None untuk posisi wajah jika frame dilewati

        self.last_frame_time = current_time

        # Konversi frame ke skala abu-abu
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Terapkan equalization histogram untuk meningkatkan kontras
        gray = cv2.equalizeHist(gray)

        # Deteksi wajah
        rects = self.detector(gray, 0)
        face_position = None  # Inisialisasi posisi wajah sebagai None

        for rect in rects:
            # Dapatkan bentuk wajah (landmark)
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Ekstraksi koordinat mata kiri dan kanan
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # Hitung rata-rata EAR antara mata kiri dan kanan
            ear = (leftEAR + rightEAR) / 2.0

            # Cek apakah EAR di bawah threshold
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                self.COUNTER_OPEN = 0  # Reset counter untuk mata terbuka
            else:
                self.COUNTER = 0  # Reset counter untuk mata tertutup
                self.COUNTER_OPEN += 1
                
            # Jika counter mencapai batas untuk frame berturut-turut, tambahkan counter setelahnya
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.COUNTER_AFTER += 1
                
            # Jika mata terbuka lebih dari 15 frame berturut-turut, reset counter setelahnya dan matikan alarm
            if self.COUNTER_OPEN >= 15:
                self.COUNTER_AFTER = 0
                self.ALARM_ON = False
                
            # Jika counter setelahnya mencapai batas, nyalakan alarm
            if self.COUNTER_AFTER >= self.COUNTER_AFTER_LIMIT:
                self.ALARM_ON = True

            # Dapatkan koordinat X dari posisi wajah
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face_position = (x + w // 2, y + h // 2)  # Pusat wajah

            # Tampilkan informasi EAR dan lainnya untuk debugging
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"THRESH: {self.EYE_AR_THRESH:.2f}", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, f"COUNTER: {self.COUNTER}", (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"COUNTER AFTER: {self.COUNTER_AFTER}", (10, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self.ALARM_ON:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Menggambar kontur mata kiri dan kanan
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame, self.ALARM_ON, face_position  # Mengembalikan posisi wajah dengan koordinat X, Y
