import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        
        # Define both linear and MLP models
        self.model = None
        self.modelL = nn.Linear(self.seq_len, self.pred_len)
        self.modelM = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len)
        )

        # Single CycleQueue
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        
        # Seasonal scaler
        self.seasonal_scaler = nn.Linear(self.enc_in, self.enc_in)

    def forward(self, x, cycle_index, train_step=0):
        assert train_step in [0, 1, 2]
        if train_step == 1:
            self.model = self.modelL
        else:
            self.model = self.modelM

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Handle cycle components
        if train_step == 2:
            with torch.no_grad():
                Q = self.cycleQueue(cycle_index, self.seq_len)
                Q = self.seasonal_scaler(Q)
        else:
            Q = self.cycleQueue(cycle_index, self.seq_len)
            Q = self.seasonal_scaler(Q)
        
        # Remove cycle from input
        x = x - Q

        # Forecasting
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add back cycle for output
        if train_step == 2:
            with torch.no_grad():
                Q_out = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
                Q_out = self.seasonal_scaler(Q_out)
        else:
            Q_out = self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)
            Q_out = self.seasonal_scaler(Q_out)

        y = y + Q_out

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y, x