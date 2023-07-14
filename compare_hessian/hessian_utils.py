import torch

from compute_hessian import compute_hessian_loss

# functions in this file are not used


def compare_hessians(cfg, model, criterion_cl, criterion_ae, alpha_off_diag, alpha_diag, inputs, outputs, reconstr_im,
                     latent, labels):
    with torch.no_grad:
        _, _, _, _, hess_mean_FD = compute_hessian_loss(cfg,
            crit_cl=criterion_cl, crit_ae=criterion_ae, outputs=outputs, reconstr_im=reconstr_im, inputs=inputs,
            labels=labels, latent=latent, model=model, alpha_off_diag=alpha_off_diag, alpha_diag=alpha_diag,
            norm=True, hessian_wrt=cfg.REG.WRT, method='finite_diff')
        hess_mean_N = 0.0
        return hess_mean_FD, hess_mean_N


def calculate_rss(estimated, ground_truth):
    err = ground_truth - estimated
    return err.T * err


def rss_hess_per_itr(cfg, model, criterion_cl, criterion_ae, alpha_off_diag, alpha_diag, inputs, outputs, reconstr_im,
                     latent, labels):
    hess_FD, hess_N = compare_hessians(cfg, model, criterion_cl, criterion_ae, alpha_off_diag, alpha_diag, inputs, outputs, reconstr_im,
                     latent, labels)
    hess_analyt = torch.zeros_like(hess_FD)  # todo: calculate analytical hessian
    rss_FD = calculate_rss(hess_FD, hess_analyt)
    rss_N = calculate_rss(hess_N, hess_analyt)
    return rss_FD, rss_N
