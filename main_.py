import os
import cv2

class main_process():
    def __init__(self):
        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    # 웹캠으로부터 학습할 이미지 가져오기
    def get_train_image(self, name):
        import Get_Image_From_Webcam
        
        # image 객체 생성
        image = Get_Image_From_Webcam.get_image_from_webcam(name)

        # 웹캠 실행 및 이미지 저장
        image.get_image()

    # 학습할 이미지에서 랜덤하게 20가지의 테스트 이미지 가져오기
    def get_test_image(self, num_images=20):
        import random
        import shutil
        from pathlib import Path
        
        # 학습할 이미지를 가져올 디렉토리
        source_directory = os.path.join(self.current_path, 'train_photos')

        # 테스트 이미지를 저장할 디렉토리
        dest_directory = os.path.join(self.current_path, 'test_photos')

        # 이미지가 저장된 최상위 디렉토리에서 하위 디렉토리 목록 가져오기
        subdirectories = [d for d in Path(source_directory).iterdir() if d.is_dir()]

        for subdir in subdirectories:
            subdir_name = subdir.name
            dest_subdir = Path(dest_directory) / subdir_name

            # 목적지 디렉토리 생성
            dest_subdir.mkdir(parents=True, exist_ok=True)

            # 이미지를 목적지 디렉토리로 복사
            image_files = [f for f in subdir.glob('*.jpg') if f.is_file()]
            selected_images = random.sample(image_files, min(num_images, len(image_files)))

            for image_path in selected_images:
                dest_path = dest_subdir / image_path.name
                shutil.copy(image_path, dest_path)

    # 학습할 데이터와 테스트할 데이터를 구축
    def set_data(self):
        import Data_Setting
       # data 객체 생성
        data = Data_Setting.data_setting()

        # 학습할 데이터와 테스트할 데이터 압축
        data.compressing_all_dataset()
    
    # 구축한 데이터를 가지고 embedding 하기
    def set_embeddings_data(self):
        import Get_Embeddings_From_Data_Set

        obj = Get_Embeddings_From_Data_Set.Embeddings()

        obj.process_and_save_embeddings()

    # embedding 한 데이터를 가지고 얼굴 분류하기
    def face_classification_using_embedding_data(self):
        import Face_Classification

        obj = Face_Classification.face_classification()

        obj.run_face_classification()

    def run_predict(self):
        import dlib
        from imutils import face_utils
        
        # 웹캠 열기
        cap = cv2.VideoCapture(0)

        #얼굴 감지 모델 로드
        detector = dlib.get_frontal_face_detector()

        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if ret:
                faces = detector(frame)

                for face in faces :
                    (x, y, w, h) = face_utils.rect_to_bb(face)

                    face_roi = frame[y:y+h, x:x+w]
                    resized_frame = cv2.resize(face_roi, (160, 160))
                
                # 화면에 결과 표시
                cv2.imshow("Face Detection", frame)

                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    run_main = main_process()
    #run_main.get_train_image('aa')
    #run_main.get_test_image()
    #run_main.set_data()
    #run_main.set_embeddings_data()
    #run_main.face_classification_using_embedding_data()
    run_main.run_predict()