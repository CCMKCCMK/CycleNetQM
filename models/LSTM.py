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

        return y, y
