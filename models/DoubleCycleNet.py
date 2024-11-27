# import torch
# import torch.nn as nn
#
# # RecurrentCycle: 周期建模核心模块
# class RecurrentCycle(torch.nn.Module):
#     def __init__(self, cycle_len, channel_size):
#         super(RecurrentCycle, self).__init__()
#         self.cycle_len = cycle_len
#         self.channel_size = channel_size
#         self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)
#
#     def forward(self, index, length):
#         # 动态计算索引并提取周期模式
#         gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
#         return self.data[gather_index]
#
#
# # DoubleCycleNet 模型
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         # 配置参数
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.cycle_len = configs.cycle
#         self.model_type = configs.model_type
#         self.d_model = configs.d_model
#         self.use_revin = configs.use_revin
#
#         # 第一层周期建模
#         self.cycleQueue1 = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         # 第二层周期建模
#         self.cycleQueue2 = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         # 第二层预测模型
#         assert self.model_type in ['linear', 'mlp']
#         if self.model_type == 'linear':
#             self.model2 = nn.Linear(self.seq_len, self.pred_len)  # 第二层直接处理第一层残差
#         elif self.model_type == 'mlp':
#             self.model2 = nn.Sequential(
#                 nn.Linear(self.seq_len, self.d_model),
#                 nn.ReLU(),
#                 nn.Linear(self.d_model, self.pred_len)
#             )
#
#     def forward(self, x, cycle_index):
#         # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
#
#         # 实例归一化 (RevIN)
#         if self.use_revin:
#             seq_mean = torch.mean(x, dim=1, keepdim=True)  # 按时间步求均值
#             seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # 按时间步求方差
#             x = (x - seq_mean) / torch.sqrt(seq_var)  # 标准化输入
#
#         # ========================= 第一层 =========================
#         # 提取数据的主要周期性成分
#         data_cycle = self.cycleQueue1(cycle_index, self.seq_len)  # 第一层提取周期性成分
#         # 去周期性，得到第一层残差
#         res1 = x - data_cycle  # 第一层的输出是去掉主要周期性后的残差
#
#         # ========================= 第二层 =========================
#         # 提取第一层残差中的次级周期性成分
#         res1_cycle = self.cycleQueue2(cycle_index, self.seq_len)  # 第二层提取次级周期性成分
#         # 去次级周期性，得到最终残差
#         res2 = res1 - res1_cycle  # 第二层的输出是去掉次级周期性后的残差
#
#         # 使用第二层预测模型对最终残差进行建模
#         final_residual = self.model2(res2.permute(0, 2, 1)).permute(0, 2, 1)
#
#         # ========================= 最终输出 =========================
#         # 最终输出 = 第一层周期性成分 + 第二层周期性成分 + 最终残差
#         y = data_cycle + res1_cycle + final_residual
#
#         # 实例反归一化 (RevIN)
#         if self.use_revin:
#             y = y * torch.sqrt(seq_var) + seq_mean  # 反归一化恢复原始数据分布
#
#         return y

# import torch
# import torch.nn as nn
#
# class RecurrentCycle(torch.nn.Module):
#     # Thanks for the contribution of wayhoww.
#     # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
#     # while the original implementation manually rolls and repeats the data through looping.
#     # It achieves a significant speed improvement (2x ~ 3x acceleration).
#     # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
#     def __init__(self, cycle_len, channel_size):
#         super(RecurrentCycle, self).__init__()
#         self.cycle_len = cycle_len
#         self.channel_size = channel_size
#         self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)
#
#     def forward(self, index, length):
#         gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
#         return self.data[gather_index]
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.cycle_len = configs.cycle
#         self.model_type = configs.model_type
#         self.d_model = configs.d_model
#         self.use_revin = configs.use_revin
#
#         self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         assert self.model_type in ['linear', 'mlp']
#         if self.model_type == 'linear':
#             self.model = nn.Linear(self.seq_len, self.pred_len)
#         elif self.model_type == 'mlp':
#             self.model = nn.Sequential(
#                 nn.Linear(self.seq_len, self.d_model),
#                 nn.ReLU(),
#                 nn.Linear(self.d_model, self.pred_len)
#             )
#
#     def forward(self, x, cycle_index):
#         # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
#
#         # instance norm
#         if self.use_revin:
#             seq_mean = torch.mean(x, dim=1, keepdim=True)
#             seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
#             x = (x - seq_mean) / torch.sqrt(seq_var)
#
#
#         # remove the cycle of the input data
#         x = x - self.cycleQueue(cycle_index, self.seq_len)
#
#         # forecasting with channel independence (parameters-sharing)
#         y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
#
#         # add back the cycle of the output data
#         y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
#
#         # instance denorm
#         if self.use_revin:
#             y = y * torch.sqrt(seq_var) + seq_mean
#
#         return y

import torch
import torch.nn as nn
import pandas as pd

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # This implementation achieves 2x ~ 3x acceleration compared to the original.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        # RecurrentCycle for residuals
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # Prediction model: Linear or MLP
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, residual_csv_path, cycle_csv_path, cycle_index):
        """
        Args:
            residual_csv_path (str): Path to the CSV file containing the residuals.
            cycle_csv_path (str): Path to the CSV file containing the first layer's cycle.
            cycle_index (Tensor): Tensor of shape (batch_size,) containing the cycle indices.
        """
        # Load residuals and first layer's cycle from CSV
        residual_data = pd.read_csv(residual_csv_path)
        cycle_data = pd.read_csv(cycle_csv_path)

        # Assume the CSV files are formatted as flattened tensors
        # Convert CSV data to PyTorch tensors
        batch_size = len(cycle_index)
        residual = torch.tensor(residual_data.values, dtype=torch.float32).view(batch_size, self.seq_len, self.enc_in)
        first_layer_cycle = torch.tensor(cycle_data.values, dtype=torch.float32).view(batch_size, self.seq_len, self.enc_in)

        # Move tensors to the same device as the model
        residual = residual.to(next(self.parameters()).device)
        first_layer_cycle = first_layer_cycle.to(next(self.parameters()).device)

        # ========================= RevIN (Instance Normalization) =========================
        if self.use_revin:
            seq_mean = torch.mean(residual, dim=1, keepdim=True)
            seq_var = torch.var(residual, dim=1, keepdim=True) + 1e-5
            residual = (residual - seq_mean) / torch.sqrt(seq_var)

        # ========================= Residual Processing =========================
        # Extract the cycle component of the residual
        residual_cycle = self.cycleQueue(cycle_index, self.seq_len)

        # Remove the cycle component from the residual
        residual_without_cycle = residual - residual_cycle

        # Predict the next steps of the residual
        residual_forecast = self.model(residual_without_cycle.permute(0, 2, 1)).permute(0, 2, 1)

        # Add back the cycle component to the predicted residual
        residual_forecast_with_cycle = residual_forecast + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # ========================= Final Output =========================
        # Final output = first layer's cycle + residual cycle + residual forecast
        y = first_layer_cycle + residual_cycle + residual_forecast_with_cycle

        # Reverse normalization (denormalize)
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y