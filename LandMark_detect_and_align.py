import cv2
import dlib
import numpy as np
from imutils import face_utils
import os
import sys

# 현재 스크립트 파일의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# align_dlib.py의 상대 경로
relative_path = 'openface-master'

# 상대 경로 지정하여 sys.path에 추가
sys.path.append(os.path.join(current_dir, relative_path))

# align_dlib 모듈을 import
from openface.align_dlib import AlignDlib

# 생성할 디렉토리의 상대 경로
photos_relative_path = 'photos/dohyeon_landmark'

# 생성할 디렉토리의 절대 경로
absolute_path = os.path.join(current_dir, photos_relative_path)

# 디렉토리 생성
os.makedirs(absolute_path, exist_ok=True)

# 얼굴 감지 모델 로드
detector = dlib.get_frontal_face_detector()

# 얼굴 감지 모델 파일의 상대 경로
model_relative_path = 'models/shape_predictor_68_face_landmarks.dat'

# 얼굴 감지 모델 파일의 절대 경로
model_absolute_path = os.path.join(current_dir, model_relative_path)

# 얼굴 랜드마크 예측 모델 로드
predictor = dlib.shape_predictor(model_absolute_path)

# AlignDlib 초기화
face_aligner = AlignDlib(model_absolute_path)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 이미지 저장을 위한 카운트 변수
count = 0

# 무시할 얼굴의 최소 크기 설정
min_face_size = 150  # 필요에 따라 조절

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 얼굴 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # 얼굴 크기 계산
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_size = max(w, h)

        # 얼굴 크기가 설정한 최소 크기보다 작으면 무시
        if face_size < min_face_size:
            continue
        
        # 얼굴 랜드마크 예측
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # 얼굴 영역을 추출
        face_roi = frame[y:y+h, x:x+w]

        # align_dlib.align 함수를 사용하여 얼굴 정렬
        # 옵션들
        # landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE
        # INNER_EYES_AND_BOTTOM_LIP
        # OUTER_EYES_AND_NOSE
        aligned_face = face_aligner.align(160, frame, bb=dlib.rectangle(x, y, x+w, y+h), landmarks=shape, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

        # 정렬된 얼굴 이미지 저장
        output_image_path = os.path.join(current_dir, photos_relative_path, f"aligned_face_{count}.jpg")
        cv2.imwrite(output_image_path, aligned_face)
        print(f"Aligned face saved: {output_image_path}")

        # 화면에 얼굴과 랜드마크 그리기
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 카운트 변수 증가
        count += 1

    # 화면에 결과 표시
    cv2.imshow("Face Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
