import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.layers import LSTM, LeakyReLU, Input, Dense
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
import keras
import sys
from sklearn.preprocessing import MinMaxScaler

'''Get Input Datas'''
with h5py.File(''.join(['data/code.h5']), 'r') as hf:
    code_input_datas = hf['code_inputs'].value
    code_input_labels = hf['code_outputs'].value
    code_org_datas = hf['code_original_datas'].value
with h5py.File(''.join(['data/index.h5']), 'r') as hf:
    index_input_datas = hf['index_inputs'].value
    index_input_labels = hf['index_outputs'].value
    index_org_datas = hf['index_original_datas'].value

# split training validation
code_training_size = int(0.8 * code_input_datas.shape[0])
code_training_datas = code_input_datas[:code_training_size, :]
code_training_labels = code_input_labels[:code_training_size, :, 2]
code_validation_datas = code_input_datas[code_training_size:, :]
code_validation_labels = code_input_labels[code_training_size:, :, 2]

if index_input_datas.shape[0] > code_input_datas.shape[0]:
    start_index = index_input_datas.shape[0] - code_input_datas.shape[0]
    index_input_datas = index_input_datas[start_index:]
    index_input_labels = index_input_labels[start_index:]
index_training_size = int(0.8 * index_input_datas.shape[0])
index_training_datas = index_input_datas[:code_training_size, :]
index_training_labels = index_input_labels[:code_training_size, :, 2]
index_validation_datas = index_input_datas[code_training_size:, :]
index_validation_labels = index_input_labels[code_training_size:, :, 2]

'''Setup Model'''
step_size = code_training_datas.shape[1]
index_step_size = index_training_datas.shape[1]
units = 32
second_units = 32
batch_size = 8
nb_features = code_training_datas.shape[2]
index_nb_features = index_training_datas.shape[2]
epochs = 10
output_size = 1
output_file_name_prefix = 'LSTM_'
# build model
code_input = Input(shape=(step_size, nb_features), name='code_input')
lstm_out = LSTM(units=units, activation='tanh', input_shape=(step_size, nb_features), use_bias=True)(
    code_input)

index_output = Dense(1, activation="sigmoid", name='index_output')(lstm_out)
index_input = Input(shape=(index_step_size, index_nb_features), name='index_input')
index_lstm = LSTM(units=units, activation='tanh', input_shape=(index_step_size, index_nb_features), use_bias=True)(index_input)
merged = keras.layers.concatenate([lstm_out, index_lstm])
# merged_lstm = LSTM(units=second_units, activation='tanh')(merged)
merged_drop = Dropout(0.8)(merged)
# merged_dense = Dense(second_units, activation='relu')(merged_drop)
# merged_dense = Dense(16, activation='relu')(merged_dense)
merged_dense = Dense(1)(merged_drop)
code_output = LeakyReLU()(merged_dense)

model = Model(inputs=[code_input, index_input], outputs=[code_output, index_output])
weight = 'data/weights/LSTM_-49-0.00029.hdf5'
if len(sys.argv) > 1:
    weight = sys.argv[1]
model.load_weights(weight)
model.compile(loss='mse', optimizer='adam', loss_weights=[1, 0.5])

predicted = model.predict([code_validation_datas, index_validation_datas])

predicted_inverted = []
scaler = MinMaxScaler()
# print(predicted.shape)
# print(np.array(predicted).shape)
# print(code_org_datas)

for i in range(code_org_datas.shape[1]):
    i=2
    scaler.fit(code_org_datas[:, i].reshape(-1, 1))
    predicted_inverted.append(scaler.inverse_transform(predicted[0].reshape(-1, 1)))
    break
print(np.array(predicted_inverted).shape)
predicted_inverted = np.array(predicted_inverted).reshape(-1)
# print(predicted_inverted)

data_count = 200

pre_start_index = predicted_inverted.shape[0] - data_count - output_size
predicted_inverted = predicted_inverted[pre_start_index:]

ground_true = code_org_datas[:,2].reshape(-1)
print(code_org_datas[:,2].shape)
start_ground = ground_true.shape[0] - data_count
ground_true = ground_true[start_ground:]

from tools import ChartsJsonParser

time_stamps = np.load('data/code_timestamps.npy')
start_index = time_stamps.shape[0] - data_count

parser = ChartsJsonParser(time_stamps.reshape(-1)[start_index - output_size:].tolist(), ground_true.tolist(),
                          predicted_inverted.tolist())
parser.save_data('data/result.json')
