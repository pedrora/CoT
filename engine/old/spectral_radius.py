def spectral_radius(module, state, experience):

    def f(s):
        out, _ = module(s, experience)
        return out

    J = torch.autograd.functional.jacobian(f, state)

    J = J.reshape(state.numel(), state.numel())

    eigvals = torch.linalg.eigvals(J)
    rho = eigvals.abs().max().real

    return rho.item()