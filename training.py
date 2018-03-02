import pandas as pd
import numpy as numpy
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
from keras.layers.core import Reshape
import keras
import numpy as np
import json

config = tf.ConfigProto()
set_session(tf.Session(config=config))
columns = ['close', 'high', 'low', 'vol', 'ma10']
index = 0
NPS, NFS = 40, 4

'''Get Input Datas'''
with h5py.File(''.join(['data/code.h5']), 'r') as hf:
    code_input_datas = hf['code_inputs'].value
    code_validate_inputs = hf['code_validate_inputs'].value
    code_input_labels = hf['code_outputs'].value
with h5py.File(''.join(['data/index.h5']), 'r') as hf:
    index_input_datas = hf['index_inputs'].value
    index_input_labels = hf['index_outputs'].value

# split training validation
code_training_size = int(0.8 * code_input_datas.shape[0])
code_training_datas = code_input_datas[:code_training_size, :]
code_training_labels = code_input_labels[:code_training_size, :, index]
code_validation_datas = code_input_datas[code_training_size:, :]
code_validation_labels = code_input_labels[code_training_size:, :, index]

if index_input_datas.shape[0] > code_input_datas.shape[0]:
    start_index = index_input_datas.shape[0] - code_input_datas.shape[0]
    index_input_datas = index_input_datas[start_index:]
    index_input_labels = index_input_labels[start_index:]
index_training_size = int(0.8 * index_input_datas.shape[0])
index_training_datas = index_input_datas[:code_training_size, :]
index_training_labels = index_input_labels[:code_training_size, :, index]
index_validation_datas = index_input_datas[code_training_size:, :]
index_validation_labels = index_input_labels[code_training_size:, :, index]

for i in range(3):
    code_training_datas = np.append(code_training_datas, code_training_datas, axis=0)
    code_training_labels = np.append(code_training_labels, code_training_labels, axis=0)
    index_training_datas = np.append(index_training_datas, index_training_datas, axis=0)
    index_training_labels = np.append(index_training_labels, index_training_labels, axis=0)

'''Setup Model'''
step_size = code_training_datas.shape[1]
index_step_size = index_training_datas.shape[1]
nb_features = code_training_datas.shape[2]
index_nb_features = index_training_datas.shape[2]
units = 32
second_units = 32
batch_size = 8
epochs = 10
output_size = NFS

''' build model'''
code_input = Input(shape=(step_size, nb_features), name='code_input')
lstm_out = LSTM(units=output_size, activation='tanh', input_shape=(step_size, nb_features), use_bias=False)(
    code_input)

noise=keras.layers.noise.GaussianNoise(0.1)(lstm_out)

# index_output = Dense(output_size, activation="relu", name='index_output')(lstm_out)
# index_input = Input(shape=(index_step_size, index_nb_features), name='index_input')
# index_lstm = LSTM(units=second_units, activation='tanh', input_shape=(index_step_size, index_nb_features),
#                   use_bias=True)(index_input)
# merged = keras.layers.concatenate([lstm_out, index_lstm])
# print(merged.shape)
# merged = Reshape((units + second_units, 1))(merged)
# print(merged.shape)
# merged_drop = Dropout(0.8)(merged)
# conv1d1 = Conv1D(activation='relu', strides=3, filters=4, kernel_size=8)(merged_drop)
# conv1d1 = Dropout(0.5)(conv1d1)
# conv1d2 = Conv1D(activation='relu', strides=3, filters=4, kernel_size=8)(conv1d1)
# conv1d2 = Dropout(0.5)(conv1d2)
# conv1d2 = Reshape((-1,))(conv1d2)
# code_output = Dense(output_size, use_bias=True,activation="relu")(merged_drop)
code_output = LeakyReLU(output_size)(lstm_out)
print(code_output.shape)
model = Model(inputs=[code_input], outputs=[code_output])

'''Save Model To File'''
model_config = model.get_config()
data = json.dumps(model_config)
print(data)
file = open("data/model.json", 'w')
file.writelines(data)
file.close()

print(code_validation_labels.shape)

model.compile(loss='mse', optimizer='adam', loss_weights=[1])
output_file_name = 'functional'
model.fit([code_training_datas], [code_training_labels],
          batch_size=batch_size,
          validation_data=(
              [code_validation_datas], [code_validation_labels]),
          epochs=epochs, callbacks=[CSVLogger('data/' + output_file_name + '.csv', append=True),
                                    ModelCheckpoint(
                                        'data/weights/' + output_file_name + '-{epoch:02d}-{val_loss:.5f}.hdf5',
                                        monitor='val_loss', verbose=1, mode='min')])
