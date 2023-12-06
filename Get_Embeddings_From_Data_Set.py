import os
import numpy as np
from keras.models import load_model
from numpy import expand_dims, asarray, savez_compressed

class Embeddings:
    def __init__(self) -> None:
        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

        # Facenet 모델을 로드하여 self.model에 할당
        self.model = self.load_facenet_model()

    #face net 모델 로드
    def load_facenet_model(self):
        # 모델 파일의 경로를 직접 전달
        model = load_model(os.path.join(self.current_path, 'models', 'facenet_keras.h5'))
        return model
    
    # 얼굴 데이터셋 로드
    def load_data_set(self):
        data = np.load(os.path.join(self.current_path, 'data_set', 'faces-dataset.npz'))

        trainx, trainy, testx, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        return trainx, trainy, testx, testy

    # 얼굴 이미지의 임베딩 추출
    def get_embedding(self, face_pixels):
        # 픽셀 값의 척도
        face_pixels = face_pixels.astype('int32')
        # 채널 간 픽셀값 표준화(전역에 걸쳐)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # 얼굴을 하나의 샘플로 변환
        samples = expand_dims(face_pixels, axis=0)
        # 임베딩을 갖기 위한 예측 생성
        yhat = self.model.predict(samples)
        return yhat[0]

    # 얼굴 데이터셋을 임베딩을 변환하고 저장
    def process_and_save_embeddings(self):
        # 얼굴 데이터셋 불러오기
        trainX, trainy, testX, testy = self.load_data_set()
        print('불러오기: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
        
        # 훈련 셋에서 각 얼굴을 임베딩으로 변환하기
        newTrainX = [self.get_embedding(face_pixels) for face_pixels in trainX]
        newTrainX = asarray(newTrainX)
        print(newTrainX.shape)
        
        # 테스트 셋에서 각 얼굴을 임베딩으로 변환하기
        newTestX = [self.get_embedding(face_pixels) for face_pixels in testX]
        newTestX = asarray(newTestX)
        print(newTestX.shape)
        
        # 저장할 경로 지정
        save_path = os.path.join(self.current_path, 'embeddings_data_set', 'faces-embeddings.npz')
        
        # 배열을 하나의 압축 포맷 파일로 저장
        savez_compressed(save_path, newTrainX, trainy, newTestX, testy)
