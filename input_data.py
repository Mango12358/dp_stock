import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tools import PastSampler
import h5py

'''Code Data Input'''
code_columns = ['close', 'vol']
code_df = pd.read_csv("data/code.csv")
code_time_stamps = code_df['datetime']
code_time_stamps = np.array(code_time_stamps)[:, None, None]
code_org_df = code_df.loc[:, code_columns]
code_org_df['avg'] = code_df['amount'] / code_df['vol'] / 100

scaler = MinMaxScaler()
code_sample_df = code_df.loc[:, code_columns]
code_sample_df['avg'] = code_df['amount'] / code_df['vol'] / 100
for c in code_columns:
    code_sample_df[c] = scaler.fit_transform(code_sample_df[c].values.reshape(-1, 1))
code_sample_df['avg'] = scaler.fit_transform(code_sample_df['avg'].values.reshape(-1, 1))

A = np.array(code_sample_df)[:, None, :]
original_A = np.array(code_sample_df)[:, None, :]
# Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 256, 1  # Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
validate_B, validate_Y = ps.transform(A)
original_B, original_Y = ps.transform(original_A)

validate_start = int(validate_B.shape[0] * 0.8)
print(validate_start)

file_name = 'data/code.h5'
with h5py.File(file_name, 'w') as f:
    f.create_dataset("code_inputs", data=B)
    f.create_dataset('code_outputs', data=Y)
    f.create_dataset('code_validate_inputs', data=validate_B[validate_start:])
    f.create_dataset("code_original_datas", data=np.array(code_org_df))
    f.create_dataset('code_original_inputs', data=original_B)
    f.create_dataset('code_original_outputs', data=original_Y)

np.save('data/code_timestamps', code_time_stamps)
np.save('data/code_validate_timestamps', code_time_stamps[validate_start:])

'''Index Data Input'''
index_columns = ['close', 'vol']
index_df = pd.read_csv("data/index.csv")
index_time_stamps = index_df['datetime']
index_time_stamps = np.array(index_time_stamps)[:, None, None]
index_org_df = index_df.loc[:, index_columns]

scaler = MinMaxScaler()
index_sample_df = index_df.loc[:, index_columns]
index_sample_df['avg'] = index_df['amount'] / index_df['vol'] / 100
for c in index_columns:
    index_sample_df[c] = scaler.fit_transform(index_sample_df[c].values.reshape(-1, 1))
index_sample_df['avg'] = scaler.fit_transform(index_sample_df['avg'].values.reshape(-1, 1))

A = np.array(index_sample_df)[:, None, :]
original_A = np.array(index_sample_df)[:, None, :]
# Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 256, 1  # Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
validate_B, validate_Y = ps.transform(A)
original_B, original_Y = ps.transform(original_A)

validate_start = int(validate_B.shape[0] * 0.8)
print(validate_start)
print(B.shape)

file_name = 'data/index.h5'
with h5py.File(file_name, 'w') as f:
    f.create_dataset("index_inputs", data=B)
    f.create_dataset('index_outputs', data=Y)
    f.create_dataset('index_validate_inputs', data=validate_B[validate_start:])
    f.create_dataset("index_original_datas", data=np.array(index_org_df))
    f.create_dataset('index_original_inputs', data=original_B)
    f.create_dataset('index_original_outputs', data=original_Y)

np.save('data/index_timestamps', index_time_stamps)
np.save('data/index_validate_timestamps', index_time_stamps[validate_start:])
