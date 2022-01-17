class DASALC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(DASALC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(16*hidden_dim, 16)
        self.relu = torch.nn.ReLU()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.mhsa_fc = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        batch_dim, class_dim, t_dim = x.shape[:3]
        x = x.reshape((batch_dim*class_dim, t_dim, -1))
        out, (h_n, h_c) = self.rnn(x, None)
        MHSA_input = out.reshape(batch_dim, class_dim, t_dim, -1)
        MHSA_input = torch.transpose(out, 1, 2).reshape(batch_dim*t_dim, class_dim, -1)
        MHSA = self.transformer_encoder(MHSA_input)
        MHSA = MHSA.reshape(batch_dim, t_dim, class_dim, -1)
        MHSA = self.mhsa_fc(MHSA)
        MHSA = self.softmax(MHSA)

        out = self.fc(out)
        out = self.relu(self.dropout(out))

        out = out.reshape(batch_dim, class_dim, t_dim, -1)
        out = torch.transpose(out, 1, 2)
        
        out = out*(1+MHSA)
        out = out.reshape(batch_dim, t_dim, -1)
        out = self.out(out)

        return out