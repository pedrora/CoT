import torch

def fisher_whiten(x, eps=1e-5):
    """
    Fisher metric approximation via covariance whitening.
    """
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean

    C = (xc.T @ xc) / (x.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(
        C + eps * torch.eye(C.shape[0], device=x.device)
    )

    C_inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T
    x_white = xc @ C_inv_sqrt

    return x_white, C