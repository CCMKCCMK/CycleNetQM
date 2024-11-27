import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from torch.multiprocessing import freeze_support

warnings.filterwarnings('ignore')


class Dataset_Electricity(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', residual_csv_path='trans_remain.csv',
                 cycle_csv_path='Q.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=24):  # 默认周期为24小时
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 7  # 一周的数据作为输入
            self.label_len = 24  # 一天的数据作为标签长度
            self.pred_len = 24 * 7  # 预测一周
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        self.root_path = root_path
        self.residual_csv_path = residual_csv_path
        self.cycle_csv_path = cycle_csv_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.residual_csv_path, self.cycle_csv_path))

        # 计算数据集边界
        num_hours = len(df_raw)
        # 分割比例：前60%训练，中间20%验证，后20%测试
        train_end = int(num_hours * 0.6)
        val_end = int(num_hours * 0.8)

        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_hours]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:-1]  # 除去date列和最后一列OT
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # 添加周期索引（基于小时）
        self.cycle_index = (df_stamp.date.apply(lambda x: x.hour).values) % self.cycle

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
