## 데이터 전처리
import os
from numpy import asarray, savez_compressed
from keras.preprocessing.image import load_img, img_to_array

class data_setting():
    def __init__(self):
        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

        # 학습할 데이터의 경로
        self.train_image_path = os.path.join(self.current_path, 'train_photos')

        # 테스트할 데이터의 경로
        self.test_image_path = os.path.join(self.current_path, 'test_photos')

    def get_data(self, path):
        x, y = list(), list()

        for subdir in os.listdir(path):
            subdir_path = os.path.join(path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    # 이미지 파일 경로 생성
                    img_path = os.path.join(subdir_path, filename)

                    # 이미지 불러오기
                    img = load_img(img_path)

                    # 이미지를 NumPy 배열로 변환
                    img_array = img_to_array(img)

                    # x에 이미지 배열 추가
                    x.append(img_array)

                    # 파일명에서 영어 부분 추출하여 y에 라벨로 추가
                    label = subdir
                    y.append(label)

        return asarray(x), asarray(y)

    def get_data_from_train_data_set(self):
        return self.get_data(self.train_image_path)

    def get_data_from_test_data_set(self):
        return self.get_data(self.test_image_path)
    
    def compressing_all_dataset(self):
        trainx, trainy = self.get_data_from_train_data_set()
        testx, testy = self.get_data_from_test_data_set()

        # 저장할 디렉토리 지정
        save_dir = os.path.join(self.current_path, 'data_set')
        os.makedirs(save_dir, exist_ok=True)

        # 저장할 파일 경로 지정
        save_path = os.path.join(save_dir, 'faces-dataset.npz')

        # 데이터 저장
        savez_compressed(save_path, trainx, trainy, testx, testy)
