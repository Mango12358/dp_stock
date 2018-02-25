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
import keras

config = tf.ConfigProto()
set_session(tf.Session(config=config))

'''Get Input Datas'''
with h5py.File(''.join(['data/code.h5']), 'r') as hf:
    code_input_datas = hf['code_inputs'].value
    code_input_labels = hf['code_outputs'].value
with h5py.File(''.join(['data/index.h5']), 'r') as hf:
    index_input_datas = hf['index_inputs'].value
    index_input_labels = hf['index_outputs'].value

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
index_lstm = LSTM(units=units, activation='tanh', input_shape=(index_step_size, index_nb_features), use_bias=True)(
    index_input)
merged = keras.layers.concatenate([lstm_out, index_lstm])
# merged_lstm = LSTM(units=second_units, activation='tanh')(merged)
merged_drop = Dropout(0.8)(merged)
# merged_dense = Dense(second_units, activation='relu')(merged_drop)
# merged_dense = Dense(16, activation='relu')(merged_dense)
merged_dense = Dense(1)(merged_drop)
code_output = LeakyReLU()(merged_dense)

model = Model(inputs=[code_input, index_input], outputs=[code_output, index_output])

print(code_validation_labels.shape)

model.compile(loss='mse', optimizer='adam', loss_weights=[1, 0.5])
output_file_name = 'functional'
model.fit([code_training_datas, index_training_datas], [code_training_labels, index_training_labels],
          batch_size=batch_size,
          validation_data=(
              [code_validation_datas, index_validation_datas], [code_validation_labels, index_validation_labels]),
          epochs=epochs, callbacks=[CSVLogger('data/' + output_file_name + '.csv', append=True),
                                    ModelCheckpoint(
                                        'data/weights/' + output_file_name + '-{epoch:02d}-{val_loss:.5f}.hdf5',
                                        monitor='val_loss', verbose=1, mode='min')])
