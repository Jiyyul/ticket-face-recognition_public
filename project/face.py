import cv2
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# 1. 얼굴 탐지 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 데이터베이스 시뮬레이션 (데이터1,2 추가)
database = {
    "image1.jpg": {
        "name": "person1",
        "birth_date": "2004.04.11.",
        "ticket_info": "2024 플레이오프 1차전 삼성 vs LG",
        "ticket_number": "T3013825783",
        "seat_number": "MR-3구역 12-3"
    },
    "image2.jpg": {
        "name": "person2",
        "birth_date": "2003.02.05.",
        "ticket_info": None,
        "ticket_number": None,
        "seat_number": None
    }
}

# 2. 입력 얼굴 데이터 로드
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise Exception("No face detected in input image")
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (100, 100))  # Resize for feature comparison

# 3. 특징 추출 함수
def extract_features(face_image):
    return cv2.calcHist([face_image], [0], None, [256], [0, 256]).flatten()

# 4. 팝업창 표시 함수
def show_popup(message):
    root = tk.Tk()
    root.withdraw()  # Tkinter 기본 창 숨김
    messagebox.showinfo("Face Recognition", message)

# 데이터베이스와 얼굴 비교 함수
def compare_with_database(current_features):
    max_similarity = 0
    matched_user = None

    for image_path, user_data in database.items():
        reference_image = preprocess_image(image_path)
        reference_features = extract_features(reference_image)
        similarity = cosine_similarity([current_features], [reference_features])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            matched_user = user_data

    return max_similarity, matched_user

# 5. 현재 웹캠에서 얼굴 탐지 및 비교
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        current_face = gray[y:y+h, x:x+w]
        current_face = cv2.resize(current_face, (100, 100))
        current_features = extract_features(current_face)

        # 데이터베이스와 비교
        similarity, user_data = compare_with_database(current_features)
        print(f"유사도: {similarity}")

        if similarity > 0.9 and user_data:
            if user_data["ticket_info"]:
                message = (f"입장하십시오.\n"
                           f"이름 : {user_data['name']}\n"
                           f"생년월일 : {user_data['birth_date']}\n"
                           f"예매내역 : {user_data['ticket_info']}\n"
                           f"예매번호 : {user_data['ticket_number']}\n"
                           f"좌석번호 : {user_data['seat_number']}")
            else:
                message = (f"입장할 수 없습니다.\n"
                           f"이름 : {user_data['name']}\n"
                           f"생년월일 : {user_data['birth_date']}\n"
                           f"예매내역 : 없음")
        else:
            message = "등록되지 않은 사용자입니다"

        show_popup(message)

        # 결과 출력
        color = (0, 255, 0) if similarity > 0.9 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
