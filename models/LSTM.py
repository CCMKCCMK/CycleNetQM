# import torch
# import torch.nn as nn
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.hidden_size = configs.d_model
#
#         # LSTM层
#         self.lstm = nn.LSTM(
#             input_size=self.enc_in,
#             hidden_size=self.hidden_size,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.1
#         )
#
#         # 全连接层 - 确保输出维度与输入特征维度相同
#         self.fc = nn.Linear(self.hidden_size, self.enc_in)
#
#     def forward(self, x):
#         # x shape: [batch_size, seq_len, enc_in]
#         batch_size = x.size(0)
#
#         # LSTM forward
#         lstm_out, (h_n, c_n) = self.lstm(x)
#         # lstm_out shape: [batch_size, seq_len, hidden_size]
#
#         # 初始化预测序列存储
#         predictions = torch.zeros((batch_size, self.pred_len, self.enc_in)).to(x.device)
#
#         # 使用最后一个时间步的输出作为初始输入
#         current_input = lstm_out[:, -1, :]
#
#         # 生成预测序列
#         for i in range(self.pred_len):
#             # 通过全连接层生成预测
#             current_pred = self.fc(current_input)  # [batch_size, enc_in]
#             predictions[:, i, :] = current_pred
#
#             # 更新current_input (可选：使用预测结果更新输入)
#             current_input = self.lstm(current_pred.unsqueeze(1))[0][:, -1, :]
#
#         return predictions

import torch
import torch.nn as nn

class RecurrentCycle(torch.nn.Module):
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

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.hidden_size = getattr(configs, 'hidden_size', 512)  # 默认值为 512
        self.num_layers = getattr(configs, 'num_layers', 2)  # 默认值为 2
        self.use_revin = configs.use_revin

        # 可学习的周期性组件
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # LSTM 模型
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        # 全连接层，将 LSTM 的输出映射到预测长度
        self.fc = nn.Linear(self.hidden_size, self.enc_in)

    def forward(self, x, cycle_index):
        # 数据预处理：去周期性分量
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x - self.cycleQueue(cycle_index, self.seq_len)  # 去除周期性分量

        # LSTM 学习残差分量
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        print(f"lstm_out shape: {lstm_out.shape}")  # 打印调试

        # 确保 lstm_out 的时间维度足够长
        if lstm_out.size(1) >= self.pred_len:
            residual = self.fc(lstm_out[:, -self.pred_len:, :])  # (batch_size, pred_len, enc_in)
        else:
            residual = self.fc(lstm_out)  # 如果时间维度不足，则直接用现有输出

        # 加回周期性分量
        cycle_output = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
        print(f"residual shape: {residual.shape}, cycle_output shape: {cycle_output.shape}")  # 打印调试

        # 对齐时间维度
        if cycle_output.size(1) > residual.size(1):
            cycle_output = cycle_output[:, :residual.size(1), :]  # 裁剪周期分量
        elif cycle_output.size(1) < residual.size(1):
            residual = torch.nn.functional.pad(residual, (0, 0, 0, cycle_output.size(1) - residual.size(1)))  # 扩展残差

        residual = residual + cycle_output

        # 数据后处理：复原归一化
        if self.use_revin:
            residual = residual * torch.sqrt(seq_var) + seq_mean

        return residual