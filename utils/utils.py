import torch
import cv2
import numpy as np

import os
import re
from pathlib import Path
from tqdm import tqdm
from glob import glob

def read_labelmap(path: str):
    mapper = {}
    labels = []
    with open(path, "r") as f:
        txt = f.read().split("\n")[1:-1]
        for i, cls in enumerate(txt):
            c = cls.split(":")
            label = c[0]
            color = tuple(map(int, c[1].split(",")))
            mapper[color] = i
            labels.append(label)
    return mapper, labels


def rgb_to_class(mask: np._typing.ArrayLike, mapper: dict[tuple, int]):
    mask_out = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
    
    for cls_color in mapper:
        idx = mask == np.array(cls_color)
        validx = idx.sum(-1) == 3
        mask_out[validx] = mapper[cls_color]
    return mask_out


def read_anno(src: str):
    mapper, classes = read_labelmap(f'{src}/labelmap.txt')
    images = []
    masks = []
    for dir in glob(f'{src}/*'):
        if not os.path.isdir(dir):
            continue

        dir_name = Path(dir).name

        names = [
            name
            for img_path in glob(f'{dir}/Images/*')
            if not (name := re.sub(r'.JPG$', '', Path(img_path).name)).endswith(' - Copy')
        ]
        i = [
            cv2.imread(f"{dir}/Images/{name}.JPG", flags=cv2.IMREAD_GRAYSCALE)
            for name in tqdm(names, desc=f'Reading {dir_name} images', leave=True, position=0)
        ]
        m = [
            rgb_to_class(cv2.cvtColor(cv2.imread(f"{dir}/Annotations/{name}.png"), cv2.COLOR_BGR2RGB), mapper)
            for name in tqdm(names, desc=f'Reading {dir_name} masks', leave=True, position=0)
        ]

        images.extend(i)
        masks.extend(m)
    
    return {
        'mapper': mapper,
        'classes': classes,
        'images': images,
        'masks': masks
    }


def read_part(src: str):
    names = [
        name
        for img_path in glob(f'{src}/imgs/*')
        if not (name := re.sub(r'.jpg$', '', Path(img_path).name)).endswith(' - Copy')
    ]
    images = [
        cv2.imread(f'{src}/imgs/{name}.jpg', flags=cv2.IMREAD_GRAYSCALE)
        for name in names
    ]
    masks = [
        np.load(f'{src}/masks/{name}.npy')
        for name in names
    ]

    return images, masks


def transform_pair(
        transform,
        image: torch.Tensor,
        mask: torch.Tensor):
    rng_state = torch.get_rng_state()
    i_trans = transform(image)
    torch.set_rng_state(rng_state)
    m_trans = transform(mask)
    return i_trans, m_trans