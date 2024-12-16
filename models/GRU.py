import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = getattr(configs, 'hidden_size', 512)
        self.num_layers = getattr(configs, 'num_layers', 2)
        
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1 if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_size, self.enc_in)
        
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
        batch_size = x.size(0)
        
        _, hidden = self.gru(x)  # hidden: [num_layers, batch_size, hidden_size]
        
        hidden_last = hidden[-1]  # [batch_size, hidden_size]
        
        predictions = []
        current_input = hidden_last
        
        for _ in range(self.pred_len):
            pred = self.fc(current_input)  # [batch_size, enc_in]
            predictions.append(pred)
            
            _, hidden = self.gru(pred.unsqueeze(1), hidden.contiguous())
            current_input = hidden[-1]
        
        predictions = torch.stack(predictions, dim=1)  # [batch_size, pred_len, enc_in]
        
        return predictions, predictions