import os
import torch
from torch import optim
import time

from utils.runningvars import RunningVars, GeneratePlots
from compute_hessian import compute_hessian_loss
# from compare_hessian.hessian_utils import rss_hess_per_itr
from utils.utils import create_heatmap, calculate_accuracy

# structure of code inspired by https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b/mit_semseg/utils.py#L33


def train_model(cfg, filename, trainloader, valloader, testloader, model, criterion_cl, criterion_ae,
                mask_class_matrix, lr, compare=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    running_val_losses = RunningVars(['Combined loss', 'CL loss', 'AE loss', 'Accuracy'])
    if compare:
        running_train_losses = RunningVars(['Combined loss', 'CL loss', 'AE loss', 'Accuracy', 'Regulariser',
                                            'Hessian RSS'])
    else:
        running_train_losses = RunningVars(['Combined loss', 'CL loss', 'AE loss', 'Accuracy', 'Regulariser'])
    running_test_losses = RunningVars(['Combined loss', 'CL loss', 'AE loss', 'Accuracy'])

    for epoch in range(cfg.TRAIN.N_EPOCHS):
        print('----------------')
        running_train_losses.reset_all()
        running_val_losses.reset_all()
        hessian_per_class_running = {}

        for i in range(cfg.MODEL.C):
            hessian_per_class_running[int(i)] = [0, torch.zeros(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM)]
        tic = time.time()
        train_per_epoch(cfg, trainloader, optimizer, model, criterion_cl, criterion_ae, mask_class_matrix,
                        epoch, hessian_per_class_running, running_train_losses, compare=compare)
        if not compare:
            validate_per_epoch(cfg, valloader=valloader, model=model, criterion_ae=criterion_ae, criterion_cl=criterion_cl,
                               running_val_losses=running_val_losses)
        print(f"time taken : {time.time() - tic}")
        os.makedirs(f'./weights/{filename}', exist_ok=True)
        torch.save(model.state_dict(), os.path.join(f'./weights/{filename}/epoch_{epoch}.pkl'))

    print('Finished Training')
    print('Models saved at {}'.format(os.path.join(f'./weights/{filename}/')))

    if not compare:
        print('Testing model')
        test_model(cfg, testloader, model, criterion_cl, criterion_ae, running_test_losses)
        running_test_losses.print_all('Test')

    plots = GeneratePlots(running_train_losses, running_val_losses, running_test_losses)
    plots.plot_all(filename)


def train_per_epoch(cfg, trainloader, optimizer, model, criterion_cl, criterion_ae, mask_class_matrix, epoch,
                    hessian_per_class_running, running_train_losses, compare=False, running_rss=None):
    for i, (inputs, labels) in enumerate(trainloader, 0):
        if cfg.MODEL.C == 1:
            labels = torch.zeros((cfg.TRAIN.BATCH_SIZE, 1), dtype=torch.float)
        optimizer.zero_grad()

        if model.is_linear:
            inputs = inputs.view(inputs.size(0), -1)

        inputs = inputs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)

        classifier_only = False

        latent, outputs, reconstr_im = model(x=inputs, classifier_only=classifier_only)

        if epoch >= cfg.TRAIN.REG_EPOCH:

            loss, loss_cl, loss_ae, reg, hess_mean = compute_hessian_loss(cfg,
                crit_cl=criterion_cl, crit_ae=criterion_ae, outputs=outputs, reconstr_im=reconstr_im, inputs=inputs,
                labels=labels, latent=latent, model=model, mask_class_matrix=mask_class_matrix,
                norm=False, hessian_wrt=cfg.REG.WRT, method='autograd')

            if compare:
                # rss = rss_hess_per_itr(model, criterion_cl, criterion_ae, mask_class_matrix, inputs,
                #                                  outputs, reconstr_im, latent, labels)
                rss = 0.0

            for label in labels.unique():
                hessian_per_class_running[int(label)][0] += torch.numel(labels[labels == label])  # count
                hessian_per_class_running[int(label)][1] += hess_mean[int(label)].squeeze(
                    0).detach().cpu().numpy()  # hessian

        else:

            loss_cl = torch.mean(criterion_cl(outputs, labels)) if cfg.REG.ALPHA_cl else 0.0
            loss_ae = torch.mean(criterion_ae(reconstr_im, inputs)) if cfg.REG.ALPHA_ae else 0.0
            loss = cfg.REG.ALPHA_cl * loss_cl + cfg.REG.ALPHA_ae * loss_ae
            reg = 0.0
            if compare:
                raise NotImplementedError

        acc = calculate_accuracy(outputs, labels) if cfg.REG.ALPHA_cl else 0.0
        loss.backward()
        optimizer.step()

        if compare:
            running_train_losses.add_all([loss, loss_cl, loss_ae, acc, reg, rss])
        else:
            running_train_losses.add_all([loss, loss_cl, loss_ae, acc, reg])

    running_train_losses.average_all()

    # if epoch >= cfg.TRAIN.REG_EPOCH:
    #     create_heatmap(hessian_per_class_running, epoch, file_name=cfg.FILENAME)
    running_train_losses.append_all()
    running_train_losses.print_all('Train')


def validate_per_epoch(cfg, valloader, model, criterion_cl, criterion_ae, running_val_losses):
    test_model(cfg, valloader, model, criterion_cl, criterion_ae, running_val_losses)
    running_val_losses.print_all('Validation')


def test_model(cfg, loader, model, criterion_cl, criterion_ae, running_losses):
    """Runs the model on the loader given. To be used only with val/test"""

    model.eval()

    with torch.no_grad():
        for (inputs, labels) in loader:
            if model.is_linear:
              inputs = inputs.view(inputs.size(0), -1)

            inputs = inputs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            latent, outputs, reconstr_im = model(x=inputs)

            loss_cl = torch.mean(criterion_cl(outputs, labels)) if cfg.REG.ALPHA_cl else 0.0
            loss_ae = torch.mean(criterion_ae(reconstr_im, inputs))  if cfg.REG.ALPHA_ae else 0.0
            combined_loss = cfg.REG.ALPHA_cl * loss_cl + cfg.REG.ALPHA_ae * loss_ae
            acc = calculate_accuracy(outputs, labels) if cfg.REG.ALPHA_cl else 0.0

            running_losses.add_all([combined_loss, loss_cl, loss_ae, acc])
        running_losses.average_all()
        running_losses.append_all()
