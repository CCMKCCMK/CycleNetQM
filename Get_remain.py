import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from models import CycleNet
from utils.timefeatures import time_features

scale = True
url = r'dataset/electricity.csv'
setting = 'Electricity_720_720_CycleNet_custom_ftM_sl720_pl720_cycle168_linear_seed1024'
_seq_len = 720
_set_type = 2 # 0: train, 1: vali, 2: test
_cycle = 168
_label_len = 0
_pred_len = 720
index = 0
flag = 'all'
_scaler = StandardScaler()
df_raw = pd.read_csv(url)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
df_raw.columns: ['date', ...(other features), target feature]
'''
cols = list(df_raw.columns)
cols.remove('OT')
cols.remove('date')
df_raw = df_raw[['date'] + cols + ['OT']] # move the target feature to the last column

print(df_raw.shape)
num_train = int(len(df_raw) * 0.7) # 70% of the data is used for training
num_test = int(len(df_raw) * 0.2) # 20% of the data is used for testing
num_vali = len(df_raw) - num_train - num_test # 10% of the data is used for validation
border1s = [0, num_train - _seq_len, len(df_raw) - num_test - _seq_len] # 0, 70% of the data - seq_len, 80% of the data - seq_len
border2s = [num_train, num_train + num_vali, len(df_raw)] # 70% of the data, 80% of the data, 100% of the data
border1 = border1s[_set_type] # in test case, border1 = 80% of the data - seq_len
border2 = border2s[_set_type] # in test case, border2 = 100% of the data

cols_data = df_raw.columns[1:] # all columns except the date column
df_data = df_raw[cols_data] # all columns except the date column

# if scale:
train_data = df_data[border1s[0]:border2s[0]] 
_scaler.fit(train_data.values) # fit the scaler to the training data
# print(_scaler.mean_)
# exit()
data = _scaler.transform(df_data.values) # scale the data
print(data.shape)
# else:
#     data = df_data.values

df_stamp = df_raw[['date']] # only the date column
df_stamp['date'] = pd.to_datetime(df_stamp.date) # convert the date column to datetime
data_stamp = time_features(pd.to_datetime(df_stamp['date'].values)) # get the time features
data_stamp = data_stamp.transpose(1, 0) # transpose the time features to match the shape of the data
_data_y = data # all data

def inverse_transform(data):
    return _scaler.inverse_transform(data) # inverse the scaling

class Configs:
    def __init__(self, seq_len, pred_len, enc_in, cycle, model_type, d_model, use_revin):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle = cycle
        self.model_type = model_type
        self.d_model = d_model
        self.use_revin = use_revin

configs = Configs(
    seq_len=_seq_len,
    pred_len=_pred_len,
    enc_in=321,
    cycle=_cycle,
    model_type='linear',
    d_model=512,
    use_revin=True
)

_model = CycleNet.Model(configs=configs).float()

print('loading model')
_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

folder_path = './test_results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

Q = _model.cycleQueue.data.detach().cpu().numpy()  # 一个周期的数据
print(Q.shape)
trues_last = _data_y[:,:]#[0, :, -1].reshape(-1)

# 计算需要多少个完整周期
Q_len = Q.shape[0]  # 一个周期的长度
total_len = len(trues_last)
num_cycles = (total_len + Q_len - 1) // Q_len  # 向上取整得到需要的周期数

# 将Q重复扩展到足够的长度
Q_repeated = np.tile(Q, [num_cycles,1])[:total_len]

# 计算remain
trues_remain = trues_last - Q_repeated

Q = inverse_transform(Q)

trues_remain = inverse_transform(trues_remain)

# 保留小数
remain = pd.DataFrame(trues_remain).round(6)
remain.to_csv(folder_path + 'remain.csv', index=False)

# 保存Q
Q = pd.DataFrame(Q).round(6)
Q.to_csv(folder_path + 'Q.csv', index=False)
print('done')
