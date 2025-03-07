"""
References:
1. Tonekaboni, S., Eytan, D., & Goldenberg, A. (2021). Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding. International Conference on Learning Representations. https://openreview.net/forum?id=8qDwejCuCN
2. https://openreview.net/forum?id=8qDwejCuCN

Acknowledgements:
- https://github.com/sanatonek/TNC_representation_learning?tab=readme-ov-file
- https://seunghan96.github.io/cl/ts/(CL_code3)TNC/
"""

import torch
import torch.nn as nn


class RnnEncoder(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, cell_type='GRU', num_layers=1, device='cpu', dropout=0, bidirectional=True):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device

        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size*(int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        if cell_type=='GRU':
            self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)

        elif cell_type=='LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                                    batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

    def forward(self, x):
        x = x.permute(2,0,1)
        if self.cell_type=='GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        elif self.cell_type=='LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x.to(self.device), past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))
        return encodings


class StateClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(StateClassifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = torch.nn.BatchNorm1d(self.input_size)
        self.nn = torch.nn.Linear(self.input_size, self.output_size)
        torch.nn.init.xavier_uniform_(self.nn.weight)

    def forward(self, x):
        x = self.normalize(x)
        logits = self.nn(x)
        return logits


class E2EStateClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, dropout_rate=0.5, 
                 cell_type='GRU', num_layers=1, bidirectional=True, device='cpu'):
        super(E2EStateClassifier, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.device = device

        if cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM}')

        self.fc = nn.Sequential(
            nn.Linear(100, input_size),  # ここで入力サイズを 100 に修正
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            #nn.Dropout(dropout)
        )
        self.nn = nn.Linear(input_size, output_size)

        self.to(device)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out, _ = self.rnn(x)  # x shape = [batch_size, seq_len, in_channel]
        encodings = self.fc(out[:, -1])  # encoding shape: (batch_size, hidden_size)
        return self.nn(encodings)
    
