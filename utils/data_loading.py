import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from utils.utils import transform_pair

from numpy import load
import re
from pathlib import Path
from glob import glob
from typing import Literal

class DentalDataset(Dataset):
    def __init__(
            self, 
            part: Literal['train', 'val', 'test'],
            img_transform=None,
            mask_transform=None):
        self.part = part
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        names = {
            re.sub(r'\.jpg$', '', Path(path).name): part
            for path in glob(f'data/{part}/imgs/*')
        }
        if part == 'train':
            names.update({
                re.sub(r'\.jpg$', '', Path(path).name): 'aug'
                for path in glob(f'data/aug/imgs/*')
            })

        self.names = list(names.items())
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        name, dir = self.names[index]

        image = read_image(f'data/{dir}/imgs/{name}.jpg')
        mask = torch.from_numpy(load(f'data/{dir}/masks/{name}.npy')).unsqueeze(0)

        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {
            'image': image,
            'mask': mask
        }