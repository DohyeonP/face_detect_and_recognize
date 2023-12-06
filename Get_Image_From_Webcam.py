## 웹캠으로부터 이미지 얻기
import dlib
import cv2
import os
from imutils import face_utils

class get_image_from_webcam():

    def __init__(self, name):
        self.name = name

        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

        # 웹캠으로부터 저장될 이미지의 경로
        self.train_image_path = os.path.join(self.current_path, 'train_photos', name)

    def get_image(self):

        # 디렉토리 생성
        os.makedirs(self.train_image_path, exist_ok=True)

        #얼굴 감지 모델 로드
        detector = dlib.get_frontal_face_detector()

        # 웹캠 열기
        cap = cv2.VideoCapture(0)

        # 이미지 저장을 위한 카운트 변수
        count = 0

        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if ret:
                faces = detector(frame)

                for face in faces :
                    (x, y, w, h) = face_utils.rect_to_bb(face)

                    face_roi = frame[y:y+h, x:x+w]
                    resized_frame = cv2.resize(face_roi, (160, 160))
                
                if resized_frame is not None:
                    file_path = os.path.join(self.train_image_path, f"{self.name}_{count}.jpg")
                    cv2.imwrite(file_path, resized_frame)
                    # 카운트 변수 증가
                    count += 1

                # 화면에 결과 표시
                cv2.imshow("Face Detection", face_roi)

                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 종료
        cap.release()
        cv2.destroyAllWindows()