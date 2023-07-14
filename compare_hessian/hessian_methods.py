import torch

"""Includes all methods to compute hessian, except autograd. Code for autograd is part of the
main pipeline and is included in compute_hessian.py"""

# functions in this file are not used anymore, but included to avoid import errors


def hessian_finite_diff(cfg, crit, decoder, z, label, hessian_wrt, inputs=None):
    # inspired by https://rh8liuqy.github.io/Finite_Difference.html
    if hessian_wrt == 'autoencoder':
        in_data = inputs
    elif hessian_wrt == 'classifier':
        in_data = label
    else:
        raise NotImplemented('Valid options to compute hessian w.r.t are: "autoencoder" or "classifier"')

    hessian = torch.zeros_like(z)
    epsilon = cfg.EPS * torch.mean(z)
    eps = epsilon * torch.eye(cfg.MODEL.EMBED_DIM).to(cfg.DEVICE)
    for j in range(cfg.MODEL.EMBED_DIM):
        for i in range(cfg.MODEL.EMBED_DIM):
            fij = torch.mean(crit(decoder(z + eps[i, :] + eps[j, :]), in_data), dim=1)
            f_ij = torch.mean(crit(decoder(z - eps[i, :] + eps[j, :]), in_data), dim=1)
            fi_j = torch.mean(crit(decoder(z + eps[i, :] - eps[j, :]), in_data), dim=1)
            f_i_j = torch.mean(crit(decoder(z - eps[i, :] - eps[j, :]), in_data), dim=1)
            hessian[:, i, j] = (fij - f_ij - fi_j + f_i_j) / (4 * epsilon * epsilon)
    return hessian
