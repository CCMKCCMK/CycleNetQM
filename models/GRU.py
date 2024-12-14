import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, configs):
        """
        configs参数说明
        - enc_in: 输入特征维度
        - seq_len: 输入序列长度
        - pred_len: 预测序列长度
        - hidden_size: GRU隐藏层大小
        - num_layers: GRU层数
        """
        super(GRUModel, self).__init__()
        
        # 基础参数
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = getattr(configs, 'hidden_size', 512)
        self.num_layers = getattr(configs, 'num_layers', 2)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1 if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 预测层
        self.fc = nn.Linear(self.hidden_size, self.enc_in)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, batch_x_mark=None, dec_inp=None, batch_y_mark=None):
        """
        输入:
            x: [Batch, seq_len, enc_in]
        输出:
            [Batch, pred_len, enc_in]
        """
        batch_size = x.size(0)
        
        # GRU编码
        _, hidden = self.gru(x)  # hidden: [num_layers, batch_size, hidden_size]
        
        # 使用最后一个时间步的隐藏状态进行预测
        hidden_last = hidden[-1]  # [batch_size, hidden_size]
        
        # 生成预测序列
        predictions = []
        current_input = hidden_last
        
        # 自回归预测
        for _ in range(self.pred_len):
            # 通过全连接层生成预测
            pred = self.fc(current_input)  # [batch_size, enc_in]
            predictions.append(pred)
            
            # 更新当前输入（用于下一步预测）
            _, hidden = self.gru(pred.unsqueeze(1), hidden.contiguous())
            current_input = hidden[-1]
        
        # 将预测结果拼接
        predictions = torch.stack(predictions, dim=1)  # [batch_size, pred_len, enc_in]
        
        return predictions