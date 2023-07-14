import os
import torch.nn as nn
import torch
import argparse
import yaml

from train import train_model
from utils.utils import reconstruct_single, sweep_all_dims, plot_hessian_without_norm, load_config, \
    set_random_seed, load_data, get_distribution_parameters
from compute_hessian import get_mask_matrix, get_off_diag_matrix, loss_jacobian
from eval.calc_mi import correl, compute_mi_edge
from eval.metrics import compute_distances_bw_labels, compute_correlation_bw_labels

# structure of code inspired by https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--reconstr", action="store_true", default=False, help="Sweep latent variables.")
    parser.add_argument("--test_hess", action="store_true", default=False, help="Sweep latent variables.")
    parser.add_argument("--sweep", action="store_true", default=False, help="Sweep latent variables.")
    parser.add_argument("--param_tune", action="store_true", default=False, help="Tune hyper-parameters")
    parser.add_argument("--cfg", default="./configs/mnist_config.yaml", help="config yaml file name")
    parser.add_argument("--mi", action="store_true", default=False, help="calculate mutual information")
    parser.add_argument("--distances", action="store_true", default=False, help="calculate distances between latent vectors")
    parser.add_argument("--jacob", action="store_true", default=False, help="calculate jacobian of loss wrt latent vector")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(cfg)

    model, train_dataset, val_dataset, test_dataset = load_data(cfg)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, drop_last=True, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True)
    sweep_testloader = torch.utils.data.DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=True)
    print(f'Train data: {len(train_dataset)}, Val data: {len(val_dataset)}, Test data: {len(test_dataset)}')

    criterion_cl = nn.CrossEntropyLoss(reduction='none')
    criterion_ae = nn.MSELoss(reduction='none')

    images, labels = next(iter(testloader))
    if cfg.MODEL.C == 1:
        mask_class_matrix = get_off_diag_matrix(cfg)
    else:
        mask_class_matrix = get_mask_matrix(cfg)

    if args.param_tune:
        for lr in cfg.LR:
            print(f'Hyperparameter Tuning LR = {lr}')
            filename = f'{cfg.FILENAME}/LR_{lr}'
            train_model(cfg, filename, trainloader, valloader, testloader, model=model, criterion_cl=criterion_cl,
                        criterion_ae=criterion_ae, mask_class_matrix=mask_class_matrix, lr=lr)

            for epoch in range(cfg.N_EPOCHS):
                print("*** Loading checkpoint: ", os.path.join(f'./weights/{filename}/epoch_{epoch}.pkl'))
                model.load_state_dict(torch.load(os.path.join(f'./weights/{filename}/epoch_{epoch}.pkl'),
                                                 map_location=cfg.DEVICE))

                reconstruct_single(cfg, filename, images, labels, model, criterion_ae, epoch=epoch)
                if epoch >= cfg.TRAIN.REG_EPOCH:
                    plot_hessian_without_norm(cfg, testloader, model, criterion_cl, criterion_ae,
                                              mask_class_matrix=mask_class_matrix, epoch=epoch, test=True, filename=filename)

    if args.reconstr:
        model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'),
                                         map_location=cfg.DEVICE))
        reconstruct_single(cfg, cfg.FILENAME, images, labels, model, criterion_ae, epoch=cfg.TEST.EPOCH)

    if args.test_hess:
        model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'),
                                         map_location=cfg.DEVICE))
        plot_hessian_without_norm(cfg, testloader, model, criterion_cl, criterion_ae,
                                  mask_class_matrix=mask_class_matrix, epoch=cfg.TEST.EPOCH, test=True, filename=cfg.FILENAME)

    if args.train:
        train_model(cfg, cfg.FILENAME, trainloader, valloader, testloader, model=model, criterion_cl=criterion_cl,
                    criterion_ae=criterion_ae, mask_class_matrix=mask_class_matrix, lr=cfg.TRAIN.LR)

        with open(f'./results/{cfg.FILENAME}/config_used.yaml', 'w') as file:
                        documents = yaml.dump(cfg, file)

        for epoch in range(cfg.TRAIN.N_EPOCHS):
            print("*** Loading checkpoint: ", os.path.join(f'./weights/{cfg.FILENAME}/epoch_{epoch}.pkl'))
            model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{epoch}.pkl'),
                                             map_location=cfg.DEVICE))

            reconstruct_single(cfg, cfg.FILENAME, images, labels, model, criterion_ae, epoch=epoch)
            if epoch >= 0: # cfg.TRAIN.REG_EPOCH:
                plot_hessian_without_norm(cfg, testloader, model, criterion_cl, criterion_ae, mask_class_matrix, epoch=epoch, test=True, filename=cfg.FILENAME)

    if args.sweep:

        for lbl in range(cfg.MODEL.C):
            if cfg.DATASET == 'dshapes':
                for clr in ['red', 'green']:
                    sweep_all_dims(cfg, model, sweep_testloader, sweep_label=lbl, sweep_clr=clr,
                                   model_name=os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'), j=0,
                                   z_mu=0, z_sig=0)
            else:
                  sweep_all_dims(cfg, model, sweep_testloader, sweep_label=lbl, sweep_clr=None,
                                   model_name=os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'), j=0,
                                    z_mu=0, z_sig=0)

    if args.mi:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'),
                                         map_location=cfg.DEVICE))
        mig = compute_mi_edge(cfg, model, testloader)

    if args.distances:
        model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'),
                                         map_location=cfg.DEVICE))
        compute_correlation_bw_labels(cfg, model, testloader)

    if args.jacob:
        model.load_state_dict(torch.load(os.path.join(f'./weights/{cfg.FILENAME}/epoch_{cfg.TEST.EPOCH}.pkl'),
                                         map_location=cfg.DEVICE))
        batch = next(iter(testloader))
        images, labels = batch
        images = images.view(images.size(0), -1) if model.is_linear else images
        z = model.encoder(images)
        jcb = loss_jacobian(criterion_ae, model.reconstr_decoder, z, labels, cfg.REG.WRT, images)
        print(jcb)

