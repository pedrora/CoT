class CoTaCore(nn.Module):

    def __init__(self, dim, k):
        super().__init__()

        self.dim = dim
        self.rg = SyntheticRGLayer(dim, k)

        self.input_adapter = nn.Linear(dim, dim)

    def forward(self, state, experience):

        # integrate experience
        x = state + self.input_adapter(experience)

        # Fisher geometric normalization
        x_white, cov = fisher_whiten(x)

        # renormalization (fractal refinement)
        new_state = self.rg(x_white)

        return new_state, cov