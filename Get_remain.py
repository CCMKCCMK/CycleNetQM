import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

scale = True
url = r''

def __read_data__(self):
    _scaler = StandardScaler()
    df_raw = pd.read_csv(url)

    '''
    df_raw.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_raw.columns)
    cols.remove('OT')
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + ['OT']]
    # print(cols)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - _seq_len, len(df_raw) - num_test - _seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[_set_type]
    border2 = border2s[_set_type]

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    if scale:
        train_data = df_data[border1s[0]:border2s[0]]
        _scaler.fit(train_data.values)
        # print(_scaler.mean_)
        # exit()
        data = _scaler.transform(df_data.values)
    else:
        data = df_data.values

    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if _timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values
    elif _timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values))
        data_stamp = data_stamp.transpose(1, 0)

    _data_x = data[border1:border2]
    _data_y = data[border1:border2]
    _data_stamp = data_stamp

    # add cycle
    _cycle_index = (np.arange(len(data)) % _cycle)[border1:border2]

def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + _seq_len
    r_begin = s_end - _label_len
    r_end = r_begin + _label_len + _pred_len

    seq_x = _data_x[s_begin:s_end]
    seq_y = _data_y[r_begin:r_end]
    seq_x_mark = _data_stamp[s_begin:s_end]
    seq_y_mark = _data_stamp[r_begin:r_end]

    cycle_index = torch.tensor(_cycle_index[s_end])

    return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

def __len__(self):
    return len(_data_x) - _seq_len - _pred_len + 1

def inverse_transform(self, data):
    return _scaler.inverse_transform(data)
