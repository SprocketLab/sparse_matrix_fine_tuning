import torch
from einops import rearrange


def low_rank_project(M, rank, reverse=False):
    """Supports batches of matrices as well.
    Returns left and right singular vectors multiplied by singular values
    """
    U, S, Vt = torch.linalg.svd(M)
    if reverse:
        S_sqrt_rev = S[..., rank:].sqrt()
        U_rev = U[..., rank:] * rearrange(S_sqrt_rev, "... rank -> ... 1 rank")
        Vt_rev = rearrange(S_sqrt_rev, "... rank -> ... rank 1") * Vt[..., rank:, :]
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, "... rank -> ... 1 rank")
    Vt = rearrange(S_sqrt, "... rank -> ... rank 1") * Vt[..., :rank, :]

    if reverse:
        return U, Vt, U_rev, Vt_rev
    return U, Vt
