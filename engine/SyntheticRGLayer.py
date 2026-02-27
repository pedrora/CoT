import torch.nn as nn

class SyntheticRGLayer(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.dim = dim
        self.k = k

        self.U = nn.Parameter(torch.randn(dim, k))
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.activation = nn.GELU()

    def forward(self, x):

        # orthonormal relevant subspace
        U = torch.linalg.qr(self.U).Q[:, :self.k]

        # coarse-graining projection
        x_proj = x @ U @ U.T

        alpha = torch.exp(self.log_alpha)

        return self.activation(alpha * x_proj)