import torch
import torch.nn as nn

# RecurrentCycle: 用于周期建模的核心模块
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        # 动态计算索引并提取周期模式
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


# 双层 CycleNet 模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 配置参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        # 第一层周期建模
        self.cycleQueue1 = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        # 第二层周期建模
        self.cycleQueue2 = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # 第一层预测模型
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model1 = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model1 = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

        # 第二层预测模型
        if self.model_type == 'linear':
            self.model2 = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model2 = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # 实例归一化 (RevIN)
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 第一层：去除输入数据的周期性成分
        x1 = x - self.cycleQueue1(cycle_index, self.seq_len)
        # 第一层：预测残差
        y1 = self.model1(x1.permute(0, 2, 1)).permute(0, 2, 1)
        # 第一层：添加周期性成分
        y1 = y1 + self.cycleQueue1((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # 第二层：对第一层的残差建模
        x2 = x - y1  # 第一层残差作为第二层的输入
        x2 = x2 - self.cycleQueue2(cycle_index, self.seq_len)
        # 第二层：预测残差
        y2 = self.model2(x2.permute(0, 2, 1)).permute(0, 2, 1)
        # 第二层：添加周期性成分
        y2 = y2 + self.cycleQueue2((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # 最终输出
        y = y1 + y2

        # 实例反归一化 (RevIN)
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y