from scipy.stats import pearsonr, wasserstein_distance
import numpy as np
import itertools
import dcor
from utils.utils import compute_latent_labels
import ot
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE


def latent_distances(cfg, model, testloader, dict_distance_fn):
    latent, _ = compute_latent_labels(cfg, model, testloader)
    distances = {}
    for dist_fn_name, dist_fn in dict_distance_fn.items():
      distances[dist_fn_name] = np.zeros((cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM))
      for i, j in itertools.product( list(range(cfg.MODEL.EMBED_DIM)),repeat=2):
          distances[dist_fn_name][i, j] = dist_fn(latent[:, i].cpu().detach().numpy(),
                                latent[:, j].cpu().detach().numpy())

      distances[dist_fn_name] = distances[dist_fn_name] / np.linalg.norm(distances[dist_fn_name])                           

    for dist_name, dist in distances.items():
      print(f"======={dist_name}=============")
      print(dist)


def compute_distances(cfg, model, testloader):
  dict_distance_fn = {
    # "correlation": pearsonr,
    "energy distance": dcor.energy_distance,
    "wasserstein distance": wasserstein_distance,
  }
  latent_distances(cfg, model, testloader, dict_distance_fn)


def latent_distance_bw_labels(cfg, model, testloader, dict_distance_fn):
    latent, labels = compute_latent_labels(cfg, model, testloader)
    labels = labels.cpu().detach().numpy()
    latent = latent.cpu().detach().numpy()
    
    unique_labels = np.unique(labels)
    d_latent_cls = {}
    for cls in unique_labels:
      d_latent_cls[cls] = latent[labels == cls]

    distances = {}
    for dist_fn_name, dist_fn in dict_distance_fn.items():
      distances[dist_fn_name] = np.zeros((len(unique_labels), len(unique_labels)))
      for i, j in itertools.product(unique_labels, repeat=2):
          distances[dist_fn_name][i, j] = dist_fn(d_latent_cls[i], d_latent_cls[j])

      # distances[dist_fn_name] = distances[dist_fn_name] / np.linalg.norm(distances[dist_fn_name])

    for dist_name, dist in distances.items():
      print(f"======={dist_name}=============")
      print(np.min(dist[np.nonzero(dist)])/np.max(dist[np.nonzero(dist)]))
      print(dist)
    

def compute_distances_bw_labels(cfg, model, testloader):
  dict_distance_fn = {
    "energy distance": dcor.energy_distance,
    "max-sliced wasserstein distance": ot.max_sliced_wasserstein_distance,
  }
  latent_distance_bw_labels(cfg, model, testloader, dict_distance_fn)


def compute_correlation_bw_labels(cfg, model, testloader):
    latent, labels = compute_latent_labels(cfg, model, testloader)
    labels = labels.cpu().detach().numpy()[:3000]
    latent = latent.cpu().detach().numpy()[:3000]
    tsne_latent = TSNE(n_components=2, init='pca', learning_rate='auto',
                       perplexity=50).fit_transform(latent)
    plt.scatter(tsne_latent[:, 0], tsne_latent[:, 1], c=labels, s=1)
    plt.savefig(os.path.join(f'./results/{cfg.FILENAME}/latent_bw_labels.png'))
    
    # unique_labels = np.unique(labels)
    # d_latent_cls = {}
    # for cls in unique_labels:
    #   d_latent_cls[cls] = latent[labels == cls]
