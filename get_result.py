import json
import sys

import h5py
import numpy as np
import pandas as pd
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

columns = ['close', 'high', 'low', 'vol', 'ma10']
index = 6
NPS, NFS = 10, 1

'''Get Input Datas'''
with h5py.File(''.join(['data/code.h5']), 'r') as hf:
    code_validate_inputs = hf['code_validate_inputs'].value
    code_validate_real = hf['code_validate_real'].value
    code_org_datas = hf['code_original_datas'].value
with h5py.File(''.join(['data/index.h5']), 'r') as hf:
    index_validate_inputs = hf['index_validate_inputs'].value

'''Setup Model'''
file = open("data/model.json", 'r')
model_config = json.load(file)
file.close()
model = Model.from_config(model_config)
weight = 'data/weights/functional-10-0.04118.hdf5'
if len(sys.argv) > 1:
    weight = sys.argv[1]
model.load_weights(weight)
model.compile(loss='mse', optimizer='adam', loss_weights=[1])

predicted = model.predict([code_validate_inputs[:, :, :5]])

predicted_inverted = []
print(predicted)
'''Predict Data Of MinMaxScaler'''
# scaler = MinMaxScaler()
# print(code_org_datas[:,index])
# for i in range(code_org_datas.shape[1]):
#     i = index
#     scaler.fit(code_org_datas[:, i].reshape(-1, 1))
#     predicted_inverted.append(scaler.inverse_transform(predicted.reshape(-1, 1)))
#     break
# print(np.array(predicted_inverted).shape)
# predicted_inverted = np.array(predicted_inverted).reshape(-1)
'''Predict Data Of MinMaxScaler End'''

''' Predict By Change  '''
change_lv1 = np.percentile(code_org_datas[:, -1], 30)
change_lv2 = np.percentile(code_org_datas[:, -1], 50)
change_lv3 = np.percentile(code_org_datas[:, -1], 70)

for i in range(predicted.shape[0]):
    max=predicted[i].max()
    if predicted[i][0] == max:
        predicted_inverted.append(change_lv1)
    elif predicted[i][1] == max:
        predicted_inverted.append(change_lv2)
    elif predicted[i][2] == max:
        predicted_inverted.append(change_lv3)
    else:
        predicted_inverted.append(10)
predicted_inverted = np.array(predicted_inverted)
'''Predict By Change  END'''

'''Predict Data Of One Hot'''

# for i in range(predicted.shape[0]):
#     tmp_index = i * NFS - 1
#     if tmp_index < 0:
#         tmp_index = 0
#     base_low = code_validate_real[tmp_index,index]
#     base_up = code_validate_real[tmp_index,index]
#     for j in range(predicted.shape[1]):
#         change = 0
#         for k in range(predicted.shape[2]):
#             if predicted[i][j][k] == 1:
#                 change = k - 10
#                 break
#         base_low = base_low * (1 + change / 100.0)
#         base_up = base_up * (1 + change / 100.0)
#         if change != 0 :
#             print(change)
#         predicted_inverted.append(base_low)
# predicted_inverted = np.array(predicted_inverted).reshape(-1)
# print(predicted_inverted.shape)
'''Predict Data Of One Hot END'''

''' Predict Data of true'''
# predicted_inverted = np.array(predicted.reshape(-1,1))
''' Predict Data of true End'''

time_stamps = np.load('data/code_validate_timestamps.npy')

ground_true = pd.DataFrame(code_validate_real).ix[:, index].values.reshape(-1)
print(ground_true.shape)

from tools import ChartsJsonParser

parser = ChartsJsonParser(time_stamps.reshape(-1).tolist(), ground_true.tolist(),
                          predicted_inverted.tolist()[1:])
parser.save_data('data/result.json')
