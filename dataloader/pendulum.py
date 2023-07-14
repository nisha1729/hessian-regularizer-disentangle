import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torchvision import transforms
from PIL import Image


class PendulumDataset(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms.Compose([transforms.Resize((64, 64)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        pil_img = Image.open(img_path)
        data = self.transforms(pil_img)
        return data, torch.tensor(0, dtype=torch.float)

    def __len__(self):
        return len(self.imgs)
