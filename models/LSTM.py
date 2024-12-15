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

# import torch
# import torch.nn as nn
#
# class RecurrentCycle(torch.nn.Module):
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
#         self.hidden_size = getattr(configs, 'hidden_size', 512)  # 默认值为 512
#         self.num_layers = getattr(configs, 'num_layers', 2)  # 默认值为 2
#         self.use_revin = configs.use_revin
#
#         # 可学习的周期性组件
#         self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         # LSTM 模型
#         self.lstm = nn.LSTM(
#             input_size=self.enc_in,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             batch_first=True
#         )
#         # 全连接层，将 LSTM 的输出映射到预测长度
#         self.fc = nn.Linear(self.hidden_size, self.enc_in)
#
#     def forward(self, x, cycle_index):
#         # 数据预处理：去周期性分量
#         if self.use_revin:
#             seq_mean = torch.mean(x, dim=1, keepdim=True)
#             seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
#             x = (x - seq_mean) / torch.sqrt(seq_var)
#
#         # 如果输入序列长度不足 pred_len，则扩展输入序列
#         if x.size(1) < self.pred_len:
#             padding = torch.zeros(x.size(0), self.pred_len - x.size(1), x.size(2), device=x.device)
#             x = torch.cat((x, padding), dim=1)
#
#         # 去除周期性分量
#         x = x - self.cycleQueue(cycle_index, self.seq_len)
#
#         # LSTM 学习残差分量
#         lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
#         print(f"lstm_out shape: {lstm_out.shape}")  # 打印调试
#
#         # 如果 LSTM 输出的时间维度不足 pred_len，则补零扩展
#         if lstm_out.size(1) < self.pred_len:
#             padding = torch.zeros(lstm_out.size(0), self.pred_len - lstm_out.size(1), lstm_out.size(2),
#                                   device=lstm_out.device)
#             lstm_out = torch.cat((lstm_out, padding), dim=1)
#
#         # 取最后 pred_len 步的残差预测
#         residual = self.fc(lstm_out[:, -self.pred_len:, :])
#
#         # 加回周期性分量
#         cycle_output = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
#         print(f"residual shape: {residual.shape}, cycle_output shape: {cycle_output.shape}")
#
#         # 对齐时间维度
#         if cycle_output.size(1) > residual.size(1):
#             cycle_output = cycle_output[:, :residual.size(1), :]
#         elif cycle_output.size(1) < residual.size(1):
#             padding = torch.zeros(cycle_output.size(0), residual.size(1) - cycle_output.size(1), cycle_output.size(2),
#                                   device=cycle_output.device)
#             cycle_output = torch.cat((cycle_output, padding), dim=1)
#
#         residual = residual + cycle_output
#
#         # 数据后处理：复原归一化
#         if self.use_revin:
#             residual = residual * torch.sqrt(seq_var) + seq_mean
#
#         return residual

# import torch
# import torch.nn as nn
#
# class RecurrentCycle(torch.nn.Module):
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
#         self.use_revin = configs.use_revin
#
#         # RecurrentCycle for cycle data
#         self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         # 默认值设置
#         self.hidden_size = 512  # 默认隐藏层大小
#         self.num_layers = 2  # 默认层数
#         self.bidirectional = False  # 默认不使用双向 LSTM
#
#         self.lstm = nn.LSTM(
#             input_size=self.enc_in,  # Feature size of each step
#             hidden_size=self.hidden_size,  # Size of hidden state
#             num_layers=self.num_layers,  # Number of layers
#             batch_first=True,  # Input shape is (batch_size, seq_len, input_size)
#             bidirectional=self.bidirectional  # Whether to use bidirectional LSTM
#         )
#
#         # Linear layer to map LSTM output to the desired shape
#         lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
#         self.fc = nn.Linear(lstm_output_size, self.enc_in)  # Map hidden state to `enc_in`
#
#     def forward(self, x, cycle_index):
#         # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
#
#         # Instance normalization (Revin)
#         if self.use_revin:
#             seq_mean = torch.mean(x, dim=1, keepdim=True)
#             seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
#             x = (x - seq_mean) / torch.sqrt(seq_var)
#
#         # Remove the cycle of the input data
#         x = x - self.cycleQueue(cycle_index, self.seq_len)
#
#         # Pass through LSTM
#         # Output of LSTM: (batch_size, seq_len, hidden_size * num_directions)
#         lstm_out, _ = self.lstm(x)  # Obtain LSTM output
#
#         # Pass LSTM output through a fully connected layer to project to `enc_in`
#         y = self.fc(lstm_out)  # (batch_size, seq_len, enc_in)
#
#         # Add back the cycle of the output data
#         y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.seq_len)
#
#         # Instance denormalization (Revin)
#         if self.use_revin:
#             y = y * torch.sqrt(seq_var) + seq_mean
#
#         return y

# import torch
# import torch.nn as nn
#
# class RecurrentCycle(torch.nn.Module):
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
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, bidirectional):
#         super(Encoder, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )
#         self.hidden_size = hidden_size
#         self.num_directions = 2 if bidirectional else 1
#
#     def forward(self, x):
#         # x: (batch_size, seq_len, input_size)
#         output, (hidden, cell) = self.lstm(x)  # hidden/cell: (num_layers * num_directions, batch_size, hidden_size)
#         return output, hidden, cell
#
#
# class Decoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
#         super(Decoder, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )
#         lstm_output_size = hidden_size * (2 if bidirectional else 1)
#         self.fc = nn.Linear(lstm_output_size, output_size)  # Map LSTM output to target size
#
#     def forward(self, x, hidden, cell):
#         # x: (batch_size, 1, input_size) - decoding one step at a time
#         output, (hidden, cell) = self.lstm(x, (hidden, cell))  # output: (batch_size, 1, hidden_size * num_directions)
#         output = self.fc(output)  # (batch_size, 1, output_size)
#         return output, hidden, cell
#
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.cycle_len = configs.cycle
#         self.use_revin = configs.use_revin
#
#         # RecurrentCycle for cycle data
#         self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
#
#         # Encoder and Decoder
#         self.hidden_size = 512  # Default hidden size for LSTM
#         self.num_layers = 2  # Default number of layers
#         self.bidirectional = False  # Default bidirectional setting
#
#         self.encoder = Encoder(
#             input_size=self.enc_in,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             bidirectional=self.bidirectional
#         )
#         self.decoder = Decoder(
#             input_size=self.enc_in,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             output_size=self.enc_in,
#             bidirectional=self.bidirectional
#         )
#
#     def forward(self, x, cycle_index):
#         # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
#
#         # Instance normalization (Revin)
#         if self.use_revin:
#             seq_mean = torch.mean(x, dim=1, keepdim=True)  # Mean along the sequence
#             seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # Variance along the sequence
#             x = (x - seq_mean) / torch.sqrt(seq_var)
#
#         # Remove the cycle of the input data
#         x = x - self.cycleQueue(cycle_index, self.seq_len)
#
#         # Pass input through the encoder
#         encoder_output, hidden, cell = self.encoder(x)
#
#         # Prepare decoder initial input (start token)
#         decoder_input = torch.zeros(x.size(0), 1, self.enc_in, device=x.device)  # (batch_size, 1, enc_in)
#
#         # Decoder initial hidden and cell states are taken from the encoder
#         decoder_hidden, decoder_cell = hidden, cell
#
#         # Store decoder outputs
#         outputs = []
#
#         # Decode step-by-step
#         for t in range(self.pred_len):
#             decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
#             outputs.append(decoder_output)  # Collect current time step output
#             decoder_input = decoder_output  # Use current output as next input
#
#         # Concatenate all outputs
#         y = torch.cat(outputs, dim=1)  # (batch_size, pred_len, enc_in)
#
#         # Add back the cycle of the output data
#         y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
#
#         # Instance denormalization (Revin)
#         if self.use_revin:
#             y = y * torch.sqrt(seq_var) + seq_mean
#
#         return y

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)  # hidden/cell: (num_layers * num_directions, batch_size, hidden_size)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, output_size)  # Map LSTM output to target size

    def forward(self, x, hidden, cell):
        # x: (batch_size, 1, input_size) - decoding one step at a time
        output, (hidden, cell) = self.lstm(x, (hidden, cell))  # output: (batch_size, 1, hidden_size * num_directions)
        output = self.fc(output)  # (batch_size, 1, output_size)
        return output, hidden, cell

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # Encoder and Decoder
        self.hidden_size = 512  # Default hidden size for LSTM
        self.num_layers = 2  # Default number of layers
        self.bidirectional = False  # Default bidirectional setting

        self.encoder = Encoder(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional
        )
        self.decoder = Decoder(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.enc_in,
            bidirectional=self.bidirectional
        )

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)

        # Pass input through the encoder
        encoder_output, hidden, cell = self.encoder(x)

        # Prepare decoder initial input (start token)
        decoder_input = torch.zeros(x.size(0), 1, self.enc_in, device=x.device)  # (batch_size, 1, enc_in)

        # Decoder initial hidden and cell states are taken from the encoder
        decoder_hidden, decoder_cell = hidden, cell

        # Store decoder outputs
        outputs = []

        # Decode step-by-step
        for t in range(self.pred_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            outputs.append(decoder_output)  # Collect current time step output
            decoder_input = decoder_output  # Use current output as next input

        # Concatenate all outputs
        y = torch.cat(outputs, dim=1)  # (batch_size, pred_len, enc_in)

        return y