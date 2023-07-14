import torch
from functools import partial


def loss_func(L, g, data, latent):
    return torch.mean(L(g(latent), data))


def get_sum_of_gradients(L, g, data, latent):
    # parts of code taken from https://discuss.pytorch.org/t/can-the-new-functional-autograd-take-batches-also-is-it-more-efficient-to-compute-a-hessian-with-the-new-functional-autograd-than-it-is-using-the-old-autograd/78848/17
    loss = torch.mean(L(g(latent), data))
    return torch.autograd.grad(loss, latent, create_graph=True)[0].sum(0)


def hessian_autograd(crit, decoder, z, label, hessian_wrt, inputs=None):
    # parts of code taken from https://discuss.pytorch.org/t/can-the-new-functional-autograd-take-batches-also-is-it-more-efficient-to-compute-a-hessian-with-the-new-functional-autograd-than-it-is-using-the-old-autograd/78848/17
    if hessian_wrt == 'autoencoder':
        in_matrix = inputs
    elif hessian_wrt == 'classifier':
        in_matrix = label
    else:
        print('Invalid argument')
        exit(0)

    partial_sum_of_gradients = partial(get_sum_of_gradients, crit, decoder, in_matrix)
    hessian_auto = torch.autograd.functional.jacobian(partial_sum_of_gradients, z,
                                                      vectorize=True, create_graph=True).swapaxes(0, 1)

    return z.shape[0] * hessian_auto  # multiply by batch size


def hessian_jacobian(crit, decoder, z, label, hessian_wrt, inputs=None):
    # parts of code taken from https://discuss.pytorch.org/t/can-the-new-functional-autograd-take-batches-also-is-it-more-efficient-to-compute-a-hessian-with-the-new-functional-autograd-than-it-is-using-the-old-autograd/78848/17
    if hessian_wrt == 'autoencoder':
        in_matrix = inputs
    elif hessian_wrt == 'classifier':
        in_matrix = label
    else:
        print('Invalid argument')
        exit(0)

    partial_loss_func = partial(loss_func, crit, decoder, in_matrix)
    jacobian = torch.autograd.functional.jacobian(partial_loss_func, z, create_graph=True, strict=True)
    return torch.matmul(jacobian.unsqueeze(2), jacobian.unsqueeze(1))


def loss_jacobian(crit, decoder, z, label, hessian_wrt, inputs=None):
    # parts of code taken from https://discuss.pytorch.org/t/can-the-new-functional-autograd-take-batches-also-is-it-more-efficient-to-compute-a-hessian-with-the-new-functional-autograd-than-it-is-using-the-old-autograd/78848/17
    if hessian_wrt == 'autoencoder':
        in_matrix = inputs
    elif hessian_wrt == 'classifier':
        in_matrix = label
    else:
        print('Invalid argument')
        exit(0)

    partial_loss_func = partial(loss_func, crit, decoder, in_matrix)
    jacobian = torch.autograd.functional.jacobian(partial_loss_func, z, create_graph=True, strict=True)
    return jacobian


def get_hessian(cfg, crit, decoder, z, labels, hessian_wrt, inputs=None, test=False, norm=True, method='autograd'):
    b, n = z.shape
    if method == 'autograd':
        hessian = hessian_autograd(crit=crit, decoder=decoder, z=z, label=labels, inputs=inputs,
                                   hessian_wrt=hessian_wrt)

    elif method == 'finite_diff':   # not used anymore
        pass

    assert norm is False
    if norm:
        norm_val = torch.linalg.norm(hessian)
        if norm_val > 0.001:
            hessian_norm = torch.div(hessian, (torch.linalg.norm(hessian, dim=(1, 2)).unsqueeze(1).unsqueeze(2)))
        else:
            hessian_norm = hessian
    else:
        hessian_norm = hessian

    assert not torch.isnan(torch.sum(hessian_norm)) is True, f'Minibatch Hessian: {hessian} \n Labels: {labels}'
    if not test:
        hessian_norm = torch.abs(hessian_norm)
    hessian_mean = torch.zeros((cfg.MODEL.C, n, n)).to(cfg.DEVICE)  # sum hessians of each class

    if cfg.MODEL.C == 1:  # for classless disentanglement datasets like celeba, dsprites etc
        hessian_per_class = hessian_norm[:, :, :]
        hessian_mean[0, :, :] = torch.mean(hessian_per_class if hessian_per_class.numel() != 0 else torch.zeros((n, n)),
                                           dim=0)
    else:
        for c in range(cfg.MODEL.C):
            if len(labels == c) == 0: print(f"No data for class {c}")
            hessian_per_class = hessian_norm[labels == c, :, :]
            hessian_mean[c, :, :] = torch.mean(
                hessian_per_class if hessian_per_class.numel() != 0 else torch.zeros((n, n)),
                dim=0)

    return hessian_norm.to(cfg.DEVICE), hessian_mean.to(cfg.DEVICE)


def get_mask_matrix(cfg):  # generates mask for diagonal elements
    # assert (cfg.MODEL.EMBED_DIM - cfg.MODEL.MISC) % cfg.MODEL.C == 0, 'Incompatible latent dimension'

    n = int((cfg.MODEL.EMBED_DIM - cfg.MODEL.MISC) / cfg.MODEL.C)
    d = cfg.MODEL.EMBED_DIM
    od = d * d - d

    m = cfg.REG.ALPHA_off_diag * torch.ones((cfg.MODEL.C, d, d)) / od  # off-diagonal

    diagonal_mask = torch.eye(d, d).unsqueeze(0).expand(cfg.MODEL.C, d, d).bool()
    m.masked_fill_(diagonal_mask, cfg.REG.ALPHA_diag / d)

    for c in range(cfg.MODEL.C):
        for idx in range(c * n, (c + 1) * n):
            m[c, idx, idx] *= 0

    if cfg.MODEL.MISC:
        misc_mask = torch.zeros((cfg.MODEL.C, d, d))
        for i in range(cfg.MODEL.MISC):
            misc_mask[:, -(i + 1), -(i + 1)] = 1  # set last few
        m.masked_fill_(misc_mask.bool(), 0)

    return m.to(cfg.DEVICE)


def get_off_diag_matrix(cfg):
    d = cfg.MODEL.EMBED_DIM
    od = d * d - d

    m = cfg.REG.ALPHA_off_diag * torch.ones((cfg.MODEL.C, d, d))  # off-diagonal

    diagonal_mask = torch.eye(d, d).unsqueeze(0).expand(cfg.MODEL.C, d, d).bool()
    # m.masked_fill_(diagonal_mask, -cfg.REG.ALPHA_diag/d)
    m.masked_fill_(diagonal_mask, 0)
    return m.to(cfg.DEVICE)


def compute_hessian_loss(cfg, crit_cl, crit_ae, outputs, reconstr_im, inputs, labels,
                         latent, model, mask_class_matrix, epoch=None,
                         norm=False, test=False, hessian_wrt='autoencoder', method='autograd'):
    if hessian_wrt == 'classifier':
        hessian_norm, hessian_mean = get_hessian(cfg, crit=crit_cl, decoder=model.decoder,
                                                 z=latent, labels=labels, norm=norm, test=test,
                                                 hessian_wrt=hessian_wrt, method=method)
    elif hessian_wrt == 'autoencoder':
        hessian_norm, hessian_mean = get_hessian(cfg, crit=crit_ae, decoder=model.reconstr_decoder,
                                                 z=latent, labels=labels, norm=norm, test=test,
                                                 inputs=inputs,
                                                 hessian_wrt=hessian_wrt,
                                                 method=method)  # pass inputs for ae hessian, else None
    elif hessian_wrt == 'both':
        hessian_norm_ae, hessian_sum_ae = get_hessian(cfg, crit=crit_ae, decoder=model.reconstr_decoder,
                                                      z=latent, labels=labels, norm=norm, test=test,
                                                      inputs=inputs,
                                                      hessian_wrt='autoencoder',
                                                      method=method)  # pass inputs for ae hessian, else None
        hessian_norm_cl, hessian_sum_cl = get_hessian(cfg, crit=crit_cl, decoder=model.decoder,
                                                      z=latent, labels=labels, norm=norm, test=test,
                                                      hessian_wrt='classifier', method=method)
        hessian_mean = hessian_sum_ae + hessian_sum_cl

    else:
        print('Inavlid Input')
        exit(0)

    if cfg.MODEL.C == 1:
        reg_hessian_per_class = torch.sum(mask_class_matrix * hessian_mean, dim=(1, 2))  # element wise mul
        reg_hessian = torch.sum(reg_hessian_per_class) / torch.sum(hessian_mean)
        trace = torch.norm(torch.diagonal(hessian_mean, dim1=1, dim2=2), dim=1)
        l2_diag = torch.norm(trace) / cfg.MODEL.EMBED_DIM
        reg = reg_hessian + l2_diag

    else:
        masked_hessian = mask_class_matrix * hessian_mean
        reg_hessian_per_class = torch.sum(masked_hessian, dim=(1, 2))  # element wise mul
        hessian_sum = torch.mean(hessian_mean, dim=(1, 2))
        non_zero_mask = torch.nonzero(hessian_sum)
        reg_hessian = torch.sum(torch.div(reg_hessian_per_class[non_zero_mask], hessian_sum[non_zero_mask]))
        reg = reg_hessian

    if torch.isnan(reg):
        raise ValueError(f'Nan, {hessian_mean}')

    loss_autoencoder = torch.mean(crit_ae(reconstr_im, inputs)) if cfg.REG.ALPHA_ae else 0.0
    loss_classifier = torch.mean(crit_cl(outputs, labels)) if cfg.REG.ALPHA_cl else 0.0
    loss_combined = cfg.REG.ALPHA_ae * loss_autoencoder + cfg.REG.ALPHA_cl * loss_classifier
    loss = loss_combined + cfg.REG.ALPHA_reg * reg
    return loss, loss_classifier, loss_autoencoder, reg, hessian_mean
