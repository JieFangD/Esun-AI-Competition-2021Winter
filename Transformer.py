class Transformer(nn.Module):
    def __init__(self,feature_size=7,num_layers=3,dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size*2, nhead=2, dim_feedforward=256, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size*2,16)
        self.softmax = torch.nn.Softmax(dim=2)
        self.init_weights()
        pe = torch.zeros(1, 24, feature_size)
        for pos in range(24):
            for i in range(0, feature_size, 2):
                pe[0, pos, i] = np.sin(pos / (12 ** ((2 * i)/feature_size)))
            for i in range(1, feature_size, 2):
                pe[0, pos, i] = np.cos(pos / (12 ** ((2 * i)/feature_size)))
        self.pe = pe.cuda()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = src + self.pe
        mask = self._generate_square_subsequent_mask(src.shape[1]).cuda()
        output = self.transformer_encoder(src,mask)
        output = self.decoder(output)
        output = self.softmax(output)
        return output