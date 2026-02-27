import torch
import torch.nn as nn

class RGFeedForward(nn.Module):

    def __init__(self, dim, k):
        super().__init__()

        self.expand = nn.Linear(dim, 4*dim)
        self.contract = nn.Linear(4*dim, dim)

        self.rg = SyntheticRGLayer(dim, k)
        self.activation = nn.GELU()

    def forward(self, x):

        # nonlinear mixing (local interaction)
        h = self.activation(self.expand(x))
        h = self.contract(h)

        # Fisher geometric normalization
        h_white, _ = fisher_whiten(h)

        # RG coarse graining
        h_rg = self.rg(h_white)

        # identity persistence
        return x + h_rg