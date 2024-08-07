import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import time

class DrowsinessDetector:
    def __init__(self):
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 5
        self.COUNTER = 0
        self.COUNTER_AFTER_LIMIT = 15
        self.COUNTER_AFTER = 0
        self.COUNTER_OPEN = 0
        self.ALARM_ON = False
        self.last_frame_time = time.time()
        self.frame_interval = 0.1  # Process a frame every 100ms (10 FPS)

        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/home/comitup/DrowsinessDetection/shape_predictor_68_face_landmarks.dat")
        
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return frame

        self.last_frame_time = current_time

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve the contrast
        gray = cv2.equalizeHist(gray)

        # Detect faces
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                self.COUNTER_OPEN = 0
            else:
                self.COUNTER = 0
                self.COUNTER_OPEN += 1
                # self.ALARM_ON = False
                
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.COUNTER_AFTER += 1
                
            if self.COUNTER_OPEN >= 15:
                self.COUNTER_AFTER = 0
                self.ALARM_ON = False
                
            if self.COUNTER_AFTER >= self.COUNTER_AFTER_LIMIT:
                self.ALARM_ON = True

            # Display EAR value in red color
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Display EYE_AR_THRESH in blue color
            cv2.putText(frame, f"THRESH: {self.EYE_AR_THRESH:.2f}", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display EYE_AR_CONSEC_FRAMES counter in green color
            cv2.putText(frame, f"COUNTER: {self.COUNTER}", (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display COUNTE_AFTER counter in green color
            cv2.putText(frame, f"COUNTER AFTER: {self.COUNTER_AFTER}", (10, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self.ALARM_ON:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Draw a rectangle around the face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame, self.ALARM_ON