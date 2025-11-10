class PositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to the input.
    Inspired by NeRF-style encoding: [x, sin(2^0 x), cos(2^0 x), ..., sin(2^L x), cos(2^L x)]
    """
    def __init__(self, in_features, num_frequencies=10, include_input=True):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Create frequency bands
        self.freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)

    def forward(self, x):
        out = [x] if self.include_input else []

        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)


class PosEncMLP(nn.Module):
    """
    ReLU-based MLP with positional encoding and Xavier initialization
    """
    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 num_encoding_freqs=10, include_input=True, outermost_linear=True):
        super().__init__()

        self.pos_enc = PositionalEncoding(in_features, num_frequencies=num_encoding_freqs, include_input=include_input)
        pe_out_dim = in_features * (2 * num_encoding_freqs + (1 if include_input else 0))

        self.net = []
        self.net.append(nn.Linear(pe_out_dim, hidden_features))
        self.net.append(nn.ReLU(inplace=True))

        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))

        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.net)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords):
        encoded = self.pos_enc(coords)
        return self.net(encoded)


f1 = PosEncMLP(in_features=in_features_f1, out_features=3, hidden_features=256,
               hidden_layers=3, num_encoding_freqs=15, include_input=True, outermost_linear=True)