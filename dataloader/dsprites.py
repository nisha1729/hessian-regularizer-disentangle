import numpy as np
from torch.utils.data import Dataset
import torch
import random
import pickle as pl


class DspritesDataset(Dataset):
    def __init__(self, root='./data', env='train', eval_mode=False):
        super().__init__()
        self.eval_mode = eval_mode  # edited for faster data loading (works only if data already exists locally).
        # all_data = np.load(f'./{root}/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        #
        # l_idx = list(range(len(all_data['imgs'])))
        # random.seed(10)
        # random.shuffle(l_idx)
        #
        # split = int(0.7 * len(all_data['imgs']))
        if env == 'train':
            # self.images = all_data['imgs'][l_idx][:split]
            # self.vfacs = all_data['latents_values'][l_idx][:split]
            # with open('./data/dsprites_train.pkl', 'wb') as f:
            #     pl.dump([self.images, self.vfacs], f, pl.HIGHEST_PROTOCOL)
            with open(f'{root}/dsprites_train.pkl', 'rb') as f:
                self.images, self.vfacs = pl.load(f)

        elif env == 'test':
            # self.images = all_data['imgs'][l_idx][split:]
            # self.vfacs = all_data['latents_values'][l_idx][split:]
            # with open('./data/dsprites_test.pkl', 'wb') as f:
            #     pl.dump([self.images, self.vfacs], f, pl.HIGHEST_PROTOCOL)
            with open(f'{root}/dsprites_test.pkl', 'rb') as f:
                self.images, self.vfacs = pl.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.eval_mode:
            return torch.tensor(np.expand_dims(self.images[idx], 0), dtype=torch.float), torch.tensor(self.vfacs[idx], dtype=torch.float)
        else:
            return torch.tensor(np.expand_dims(self.images[idx], 0), dtype=torch.float), torch.tensor(0, dtype=torch.float)
