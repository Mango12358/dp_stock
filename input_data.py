import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tools import PastSampler
import h5py

'''Code Data Input'''
code_columns = ['close', 'high', 'low', 'vol', 'ma10']
code_df = pd.read_csv("data/code.csv").fillna(0)
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
print(A.shape)
# Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 80, 4  # Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
original_B, original_Y = ps.transform(original_A)

ps = PastSampler(NPS, NFS, sliding_window=True)
validate_B, validate_Y = ps.transform(A)
validate_count = (int(validate_B.shape[0] * 0.2)) * NFS
validate_B_start = int(validate_B.shape[0] * 0.2)
print(validate_B_start)
print(validate_count)

np.save('data/code_timestamps', code_time_stamps)
validate_timestamps_start = code_time_stamps.shape[0] - validate_count
np.save('data/code_validate_timestamps', code_time_stamps[validate_timestamps_start:])
validate_real = code_df.set_index('datetime', drop=False).ix[
                code_time_stamps[validate_timestamps_start:].reshape(-1).tolist(), :]
validate_real['avg'] = validate_real['amount'] / validate_real['vol'] / 100
validate_columns = np.append(code_columns, 'avg')

file_name = 'data/code.h5'
with h5py.File(file_name, 'w') as f:
    f.create_dataset("code_inputs", data=B)
    f.create_dataset('code_outputs', data=Y)
    f.create_dataset('code_validate_inputs', data=validate_B[validate_B_start:])
    f.create_dataset('code_validate_real', data=np.array(validate_real.ix[:, validate_columns]))
    f.create_dataset("code_original_datas", data=np.array(code_org_df))
    f.create_dataset('code_original_inputs', data=original_B)
    f.create_dataset('code_original_outputs', data=original_Y)




'''Index Data Input'''
index_columns = ['close', 'high', 'low', 'vol', 'ma10']
index_df = pd.read_csv("data/index.csv").fillna(0)
index_df = index_df.set_index('datetime', drop=False)
index_df = index_df.ix[code_df['datetime'].values.tolist()]
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
print(A.shape)
# Make samples of temporal sequences of pricing data (channel)
# NPS, NFS = 80, 4  # Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
original_B, original_Y = ps.transform(original_A)

ps = PastSampler(NPS, NFS, sliding_window=True)
validate_B, validate_Y = ps.transform(A)
validate_count = (int(validate_B.shape[0] * 0.2)) * NFS
validate_B_start = int(validate_B.shape[0] * 0.2)
print(validate_B_start)

file_name = 'data/index.h5'
with h5py.File(file_name, 'w') as f:
    f.create_dataset("index_inputs", data=B)
    f.create_dataset('index_outputs', data=Y)
    f.create_dataset('index_validate_inputs', data=validate_B[validate_B_start:])
    f.create_dataset("index_original_datas", data=np.array(index_org_df))
    f.create_dataset('index_original_inputs', data=original_B)
    f.create_dataset('index_original_outputs', data=original_Y)

np.save('data/index_timestamps', index_time_stamps)
validate_timestamps_start = code_time_stamps.shape[0] - validate_count
print(validate_count)
np.save('data/index_validate_timestamps', index_time_stamps[validate_timestamps_start:])
