import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_size = configs.d_model

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 全连接层 - 确保输出维度与输入特征维度相同
        self.fc = nn.Linear(self.hidden_size, self.enc_in)

    def forward(self, x):
        # x shape: [batch_size, seq_len, enc_in]
        batch_size = x.size(0)

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: [batch_size, seq_len, hidden_size]

        # 初始化预测序列存储
        predictions = torch.zeros((batch_size, self.pred_len, self.enc_in)).to(x.device)

        # 使用最后一个时间步的输出作为初始输入
        current_input = lstm_out[:, -1, :]

        # 生成预测序列
        for i in range(self.pred_len):
            # 通过全连接层生成预测
            current_pred = self.fc(current_input)  # [batch_size, enc_in]
            predictions[:, i, :] = current_pred

            # 更新current_input (可选：使用预测结果更新输入)
            current_input = self.lstm(current_pred.unsqueeze(1))[0][:, -1, :]

        return predictions