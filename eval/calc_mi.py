
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from eval.edge import EDGE
from utils.utils import heatmap_hessian


def correl(model, testloader):
    batch = next(iter(testloader))
    images, labels = batch
    z = model.encoder(images.view(images.size(0), -1)).detach().numpy() if model.is_linear else model.encoder(images).detach().numpy()
    for i in range(z.shape[1]):
        plt.scatter(z[:, i], labels)
        plt.show()


def compute_mi_edge(cfg, model, testloader):
    batch = next(iter(testloader))
    images, labels = batch
    latent_vectors = model.encoder(images.view(images.size(0), -1).to(cfg.DEVICE)).detach().cpu().numpy()
    scores = dict()
    # #between latent and labels
    labels = labels.detach().cpu().numpy()
    for i in range(cfg.MODEL.EMBED_DIM):
        # breakpoint()
        scores[str(i)] = EDGE(latent_vectors[:, i], labels)

    print('MI with labels', scores)
    scores_list = sorted(scores.items())
    scores_x, scores_y = zip(*scores_list)
    plt.bar(scores_x, scores_y, width=0.1)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Mutual Information")
    plt.savefig(f"./results/{cfg.FILENAME}/mi_label_plot.png")
    # between latent neurons
    scores_inter_latent = np.zeros((cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM))

    for i in range(cfg.MODEL.EMBED_DIM):
        for j in range(cfg.MODEL.EMBED_DIM):
            scores_inter_latent[i, j] = EDGE(latent_vectors[:, i], latent_vectors[:, j])

    plotname = os.path.join(f'./results/{cfg.FILENAME}/mi_plot.png')
    heatmap_hessian(scores_inter_latent, 0, plotname)
