import numpy as np
import os

# 스크립트 파일의 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# npz 파일 로딩
data = np.load(os.path.join(script_dir, 'embeddings_data_set', 'faces-embeddings.npz'))

# 파일 내용 출력
for key in data.files:
    print(key, data[key])