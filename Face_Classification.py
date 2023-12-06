import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import cv2

class face_classification():
    def __init__(self) -> None:
        # 현재 디렉토리 경로
        self.current_path = os.path.dirname(os.path.abspath(__file__))

    # 임베딩 데이터셋 불러오기
    def load_embedding_data_set(self):
        data = np.load(os.path.join(self.current_path, 'embeddings_data_set', 'faces-embeddings.npz'))
        trainx, trainy, testx, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        print('데이터셋: 훈련 %d개, 테스트 %d개' % (trainx.shape[0], testx.shape[0]))
        return trainx, trainy, testx, testy
    
    # 입력 벡터를 정규화하기
    def normalize_vectors(self, trainX, testX):
        in_encoder = Normalizer(norm='l2')
        trainx = in_encoder.transform(trainX)
        testx = in_encoder.transform(testX)
        return trainx, testx

    # 목표 레이블을 인코딩하기
    def encode_labels(self, trainy, testy):
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        trainy = out_encoder.transform(trainy)
        testy = out_encoder.transform(testy)
        return trainy, testy

    # 분류 모델을 맞추기(적합시키기)
    def fit_model(self, trainX, trainy):
        model = SVC(kernel='linear')
        model.fit(trainX, trainy)
        return model

    # 훈련된 모델을 사용하여 예측하기
    def make_predictions(self, model, trainX, testX):
        yhat_train = model.predict(trainX)
        yhat_test = model.predict(testX)
        return yhat_train, yhat_test

    # 정확도 점수를 계산하고 출력하기
    def calculate_accuracy(self, trainy, testy, yhat_train, yhat_test):
        score_train = accuracy_score(trainy, yhat_train)
        score_test = accuracy_score(testy, yhat_test)
        print('정확도: 훈련=%.3f, 테스트=%.3f' % (score_train * 100, score_test * 100))
    
    # 얼굴 분류 하기
    def run_face_classification(self):
        # 데이터셋을 불러오고
        trainx, trainy, testx, testy = self.load_embedding_data_set()

        # 입력 벡터 정규화
        trainx, testx = self.normalize_vectors(trainx, testx)

        # 목표 레이블 인코딩
        trainy, testy = self.encode_labels(trainy, testy)

        # 분류 모델 훈련
        model = self.fit_model(trainx, trainy)

        # 예측 수행
        yhat_train, yhat_test = self.make_predictions(model, trainx, testx)

        # 정확도 계산 및 출력
        self.calculate_accuracy(trainy, testy, yhat_train, yhat_test)
