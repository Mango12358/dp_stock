import json
import sys

import h5py
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

columns = ['close', 'high', 'low', 'vol', 'ma10']
index = 0
NPS, NFS = 40, 4

'''Get Input Datas'''
with h5py.File(''.join(['data/code.h5']), 'r') as hf:
    code_validate_inputs = hf['code_validate_inputs'].value
    code_validate_real = hf['code_validate_real'].value
    code_org_datas = hf['code_original_datas'].value
with h5py.File(''.join(['data/index.h5']), 'r') as hf:
    index_validate_inputs = hf['index_validate_inputs'].value

'''Setup Model'''
file = open("data/model.json",'r')
model_config = json.load(file)
file.close()
model =  Model.from_config(model_config)
weight = 'data/weights/functional-10-0.00171.hdf5'
if len(sys.argv) > 1:
    weight = sys.argv[1]
model.load_weights(weight)
model.compile(loss='mse', optimizer='adam', loss_weights=[1])


predicted = model.predict([code_validate_inputs])

predicted_inverted = []
scaler = MinMaxScaler()
print(code_org_datas[:,index])
for i in range(code_org_datas.shape[1]):
    i = index
    scaler.fit(code_org_datas[:, i].reshape(-1, 1))
    predicted_inverted.append(scaler.inverse_transform(predicted.reshape(-1, 1)))
    break
print(np.array(predicted_inverted).shape)
predicted_inverted = np.array(predicted_inverted).reshape(-1)
# print(predicted_inverted)

time_stamps = np.load('data/code_validate_timestamps.npy')

ground_true = pd.DataFrame(code_validate_real).ix[:, index].values.reshape(-1)
print(ground_true.shape)

from tools import ChartsJsonParser

parser = ChartsJsonParser(time_stamps.reshape(-1).tolist(), ground_true.tolist(),
                          predicted_inverted.tolist())
parser.save_data('data/result.json')
