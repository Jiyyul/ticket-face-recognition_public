import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# 1. 얼굴 탐지 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 입력 얼굴 데이터 로드
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise Exception("No face detected in input image")
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (100, 100))  # Resize for feature comparison

input_face = preprocess_image('image2.jpg')

# 3. 특징 추출 함수
def extract_features(face_image):
    return cv2.calcHist([face_image], [0], None, [256], [0, 256]).flatten()

input_features = extract_features(input_face)

# 4. 팝업창 표시 함수
def show_popup():
    root = tk.Tk()
    root.withdraw()  # Tkinter 기본 창 숨김
    messagebox.showinfo("Face Recognition", "Same Person Detected!")

# 5. 현재 웹캠에서 얼굴 탐지 및 비교
cap = cv2.VideoCapture(0)

popup_shown = False  # 팝업 중복 방지

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        current_face = gray[y:y+h, x:x+w]
        current_face = cv2.resize(current_face, (100, 100))
        current_features = extract_features(current_face)

        # Cosine similarity 비교
        similarity = cosine_similarity([input_features], [current_features])[0][0]

        # 결과 출력
        if similarity > 0.9:  # 임계값 0.9
            text = "Same Person"
            color = (0, 255, 0)  # Green
            if not popup_shown:
                show_popup()  # 팝업창 표시
                popup_shown = True  # 팝업 표시된 상태로 설정
        else:
            text = "Different Person"
            color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
