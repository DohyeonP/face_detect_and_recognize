# 5명의 유명인사 얼굴 데이터셋의 분류기 개발
from random import choice
import numpy as np
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import os
import cv2

# 현재 디렉토리 경로
current_path = os.path.dirname(os.path.abspath(__file__))

face_data = np.load(os.path.join(current_path, 'embeddings_data_set', 'faces-embeddings.npz'))

# 얼굴 불러오기
data = np.load(os.path.join(current_path, 'data_set', 'faces-dataset.npz'))
testX_faces = data['arr_2']

# 얼굴 임베딩 불러오기
data = np.load(os.path.join(current_path, 'embeddings_data_set', 'faces-embeddings.npz'))
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# 입력 벡터 일반화
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# 목표 레이블 암호화
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# 모델 적합
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# 테스트 데이터셋에서 임의의 예제에 대한 테스트 모델
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# 얼굴 예측
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

# 이름 얻기
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('예상: %s (%.3f)' % (predict_names[0], class_probability))
print('추측: %s' % random_face_name[0])


# ADD
# OpenCV를 사용하여 이미지를 로드하고 색상 채널 변경
random_face_pixels = cv2.cvtColor(random_face_pixels, cv2.COLOR_BGR2RGB)

# 재미삼아 그리기
pyplot.imshow((random_face_pixels * 255).astype(np.uint8))
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
