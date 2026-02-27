class ConceptField(nn.Module):
    def __init__(self, dim, num_concepts):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_concepts, dim))

    def energy(self, state):
        d = torch.cdist(state, self.centers)
        return -torch.logsumexp(-d, dim=1)