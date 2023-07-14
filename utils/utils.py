import os
import numpy as np
import torch
import torchvision
import seaborn
import PIL
import yaml
import cv2
import matplotlib.pyplot as plt
from compute_hessian import compute_hessian_loss
from models.cifar_model import CIFARModel
from models.mnist_model import MNISTModel, MNISTConvModel
from models.dshapes_model import DShapesModel
from models.dsprites_model import DspritesConvModel, DspritesFCModel
from dataloader.dataset_helper import (get_cifar_data, get_clr_mnist_data,
                                       get_dsprites_data, get_celeba_data, get_dsprites_test_data,
                                       get_pendulum_data)


def plot_hessian_without_norm(cfg, data_loader, model, criterion_cl, criterion_ae, mask_class_matrix,
                              epoch, filename, val=False, test=False):
    """
    plots hessian without norm
    """

    hessian_per_class_running = {}
    for i in range(cfg.MODEL.C):
        hessian_per_class_running[int(i)] = [0, torch.zeros(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM)]

    model.eval()

    for i, (inputs, labels) in enumerate(data_loader, 0):
        if model.is_linear:
            inputs = inputs.view(inputs.size(0), -1)
        inputs = inputs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)

        if cfg.MODEL.C == 1:
            labels = torch.zeros((cfg.TRAIN.BATCH_SIZE), dtype=torch.float)

        latent, outputs, reconstr_im = model(x=inputs)

        _, _, _, _, hess_mean = compute_hessian_loss(cfg, crit_cl=criterion_cl, crit_ae=criterion_ae,
                                                     outputs=outputs, reconstr_im=reconstr_im,
                                                     inputs=inputs, labels=labels, latent=latent,
                                                     model=model, mask_class_matrix=mask_class_matrix,
                                                     norm=False, test=True,
                                                     hessian_wrt=cfg.REG.WRT)

        for label in labels.unique():
            hessian_per_class_running[int(label)][0] += torch.numel(labels[labels == label])  # count
            hessian_per_class_running[int(label)][1] += hess_mean[int(label)].squeeze(
                0).detach().cpu().numpy()  # hessian

    create_heatmap(hessian_per_class_running, epoch, val=val, test=test, file_name=filename)


def create_heatmap(hessian_per_class, epoch, file_name, val=False, test=False):
    """hessian heatmaps
    :param val: only affects plot names. names as train/test"""

    hessian_dir = f'./results/{file_name}'
    if val:
        dir_path = f'{hessian_dir}/val'
    elif test:
        dir_path = f'{hessian_dir}/test'
    else:
        dir_path = f'{hessian_dir}/train'

    for label, (cnt_label, hessian) in hessian_per_class.items():
        plotname = os.path.join(dir_path, f'hessian_epoch_{epoch}_class_{label}.png')

        print('Plotting hessian for Class', label)
        heatmap_hessian(np.abs(hessian) / cnt_label, label, loc=plotname)


def reconstruct_single(cfg, name, images, labels, classifier, criterion_ae, epoch):

    images_ = images.view(images.size(0), -1) if classifier.is_linear else images
    images_ = images_.to(cfg.DEVICE)
    labels = labels.to(cfg.DEVICE)
    if cfg.MODEL.C == 1:
        labels = torch.zeros((cfg.TRAIN.BATCH_SIZE), dtype=torch.float)
    _, outputs, decoded_imgs = classifier(images_)
    loss_ae = torch.mean(criterion_ae(decoded_imgs, images_))

    decoded_imgs = decoded_imgs.to(cfg.DEVICE).view(images.shape)
    acc = calculate_accuracy(outputs, labels) if cfg.REG.ALPHA_cl else 0.0

    print(f' On test mini-batch: Reconstruction loss: {loss_ae: .4f} | Accuracy: {acc}%')
    imshow(cfg, torchvision.utils.make_grid(images.data, pad_value=1),
           decoded_img=torchvision.utils.make_grid(decoded_imgs.data, pad_value=1),
           fullpath=os.path.join(f'./results/{name}/reconstructed/output_epoch_{epoch}.jpg'))


def sweep(cfg, model, name, image_, label, dim, class_n, model_name, clr, j, z_mu, z_sig):
    """Sweep a dimension of the latent variables to see how the output varies"""
    model.load_state_dict(torch.load(model_name, map_location=cfg.DEVICE))

    image = image_.view(image_.size(0), -1) if model.is_linear else image_

    image = image.to(cfg.DEVICE)
    label = label.to(cfg.DEVICE)
    if cfg.MODEL.C == 1:
        label = torch.zeros(cfg.TEST.BATCH_SIZE, dtype=torch.float)
    out_im = []
    for i in range(0, 64, 8):
        _, outputs, decoded_imgs = model(image, dim=dim, sweep=True, sweep_var=i / 10)
        if cfg.REG.ALPHA_cl:
            _, pred = torch.max(outputs.data, 1)
            if pred != label:
                print(f'Dim: {dim} | Wrong classification at delta factor: {i} | Label: {label} | Pred: {pred}')

        decoded_imgs = decoded_imgs.to(cfg.DEVICE).view(image_.shape)
        out_im.append(decoded_imgs)
    im = torch.cat(out_im, dim=0)
    os.makedirs(f'./results/{name}/sweep_{j}/', exist_ok=True)
    if clr:
        path = os.path.join(f'./results/{name}/sweep_{j}/class_{str(class_n)}_dim_{str(dim)}_{clr}.jpg')
    else:
        path = os.path.join(f'./results/{name}/sweep_{j}/class_{str(class_n)}_dim_{str(dim)}.jpg')
    imshow(cfg, torchvision.utils.make_grid(image_.data, pad_value=1),
           decoded_img=torchvision.utils.make_grid(im.data, pad_value=1), fullpath=path)


def sweep_all_dims(cfg, model, testloader, sweep_label, model_name, j,
                   sweep_clr, z_mu, z_sig):
    if cfg.MODEL.C == 1:
        for i in range(j):
            img, label = next(iter(testloader))
            # os.makedirs(f'./results/{cfg.FILENAME}/sweep_all/', exist_ok=True)
            # cv2.imwrite(f'./results/{cfg.FILENAME}/sweep_all/img_{i}.png',  np.transpose(np.array(img[0]/2 + 0.5)*256.0, (1,2,0))) # for one channel images
    else:
        img, label = get_image_class(testloader, label=sweep_label)
    for sweep_dim in range(cfg.MODEL.EMBED_DIM):
        sweep(cfg, model, cfg.FILENAME, img, dim=sweep_dim, class_n=sweep_label,
              label=label, model_name=model_name, clr=sweep_clr, j=j,
              z_mu=0, z_sig=0)


def compute_latent_labels(cfg, model, testloader, n_iter=None):
    l_latent = []
    l_labels = []
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            if n_iter is not None and idx > n_iter:
                break

            if model.is_linear:
                inputs = inputs.view(inputs.size(0), -1)

            inputs = inputs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            latent, outputs, reconstr_im = model(x=inputs)
            l_latent.append(latent)
            l_labels.append(labels)

    return torch.cat(l_latent), torch.cat(l_labels)


def get_distribution_parameters(cfg, model, testloader):
    latent, _ = compute_latent_labels(cfg, model, testloader)
    return torch.mean(latent, dim=0), torch.std(latent, dim=0)


def imshow(cfg, img, fullpath, decoded_img=None):
     # parts of code taken from https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py

    fig, ax = plt.subplots(figsize=(50, 50), dpi=100)
    # fig.set_size_inches(20, 20, forward=True)
    fig.add_subplot(2, 1, 1)
    npimg = img.cpu().numpy()
    if cfg.DATASET == ['cifar', 'celeba']:
        npimg = npimg / 2 + 0.5  # unnormalize

    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 1 0 2
    plt.title('Original Image')

    fig.add_subplot(2, 1, 2)
    deimg = decoded_img.cpu().numpy()
    if cfg.DATASET == 'cifar':
        deimg = deimg / 2 + 0.5
    plt.axis('off')
    plt.imshow(np.transpose(deimg, (1, 2, 0)))
    # plt.title('Reconstructed Image')

    os.makedirs(os.path.dirname(fullpath), exist_ok=True)
    plt.savefig(fullpath)


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    return torch.sum(predicted == labels) * 100 / len(labels)


def gen_plots(epoch_list, loss_ae_list, file_name, loss_list=None, reg_list=None, mode='train'):
    plt.figure()
    plt.plot(epoch_list, loss_ae_list, label=f'{mode} reconstr. loss')
    if loss_list is not None:
        plt.plot(epoch_list, loss_list, label=f'{mode} total train loss')
    if reg_list is not None:
        plt.plot(epoch_list, reg_list, label=f'{mode} regulariser')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Loss Plot')
    plt.legend()
    os.makedirs('./results', exist_ok=True)
    plt.savefig(os.path.join(f'./results/loss_{file_name}.png'))


def heatmap_hessian(var, label, loc, lim=True, mi=False):
    plt.figure()
    if mi:
        ax = seaborn.heatmap(var,
                             cmap=seaborn.cubehelix_palette(rot=-.2, reverse=True, as_cmap=True))  # vmin=0, vmax=0.1)
    else:
        ax = seaborn.heatmap(var)  # vmin=0, vmax=0.1)

    # plt.imshow(var)  # hot
    plt.title(f"Label: %d" % label)
    os.makedirs(os.path.dirname(loc), exist_ok=True)
    plt.savefig(loc)


def get_image(loader, colour='red', label=0):
    while True:
        img, lbl = next(iter(loader))
        clr = get_main_color(img)
        if clr == colour and lbl == label:
            return img, lbl


def get_image_class(loader, label=0):
    c = 0
    while c <= 12:
        img, lbl = next(iter(loader))
        while lbl != label:
            img, lbl = next(iter(loader))
        c += 1

    return img, lbl


def get_main_color(img):
    " taken from https://stackoverflow.com/questions/2270874/image-color-detection-using-python"
    img = tensor_to_image(img)
    colors = img.getcolors(256)  # put a higher value if there are many colors in your image
    max_occurence, most_present = 0, 0
    try:
        for c in colors:
            if c[0] > max_occurence and not all(v == 0 for v in c[1]):
                (max_occurence, most_present) = c
        return get_clr_name(most_present)
    except TypeError:
        raise Exception("Too many colors in the image")


def get_clr_name(rgb):
    argmax = rgb.index(max(rgb))
    clr_dict = {0: 'red', 1: 'green', 2: 'blue'}
    return clr_dict[argmax]


def tensor_to_image(tensor):
    tensor = tensor * 255
    array = np.array(tensor, dtype=np.uint8)
    if np.ndim(array) > 3:
        assert array.shape[0] == 1
        array = array[0]
    while array.shape.index(3) != 2:
        array = array.T
    return PIL.Image.fromarray(array)


class Dict2Class(object):
    " taken from https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object"
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Dict2Class(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Dict2Class(v) if isinstance(v, dict) else v)


def load_config(yaml_file):
    with open(yaml_file, "r") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
        return Dict2Class(cfg)


def set_random_seed(cfg):
    # take from https://github.com/chenjie/PyTorch-CIFAR-10-autoencoder/blob/master/main.py
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)


def load_data(cfg):
    if cfg.DATASET == 'cifar':
        train_dataset, val_dataset, test_dataset = get_cifar_data()
        model = CIFARModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'mnist':
        train_dataset, val_dataset, test_dataset = get_clr_mnist_data(cfg)
        model = MNISTModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'dshapes':
        train_dataset, val_dataset, test_dataset = get_dshapes_data()
        model = DShapesModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'mnist_cnn':
        train_dataset, val_dataset, test_dataset = get_clr_mnist_data(cfg)
        model = MNISTConvModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'dsprites':
        train_dataset, val_dataset, test_dataset = get_dsprites_data(cfg)
        model = DspritesConvModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'dsprites_fc':
        train_dataset, val_dataset, test_dataset = get_dsprites_data(cfg)
        model = DspritesFCModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'celeba':
        train_dataset, val_dataset, test_dataset = get_celeba_data()
        model = DspritesConvModel(cfg).to(cfg.DEVICE)
    elif cfg.DATASET == 'pendulum':
        train_dataset, val_dataset, test_dataset = get_pendulum_data()
        model = DspritesFCModel(cfg).to(cfg.DEVICE)
    else:
        raise ValueError('Incorrect dataset.')
    return model, train_dataset, val_dataset, test_dataset
