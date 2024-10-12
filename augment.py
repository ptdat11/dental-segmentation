import torch
import cv2
import numpy as np
from torchvision.transforms import v2
import config
from utils.utils import read_labelmap, read_part, transform_pair

import os
import argparse
from pandas import Series
from tqdm import tqdm
from typing import Sequence

def create_dist(
        masks: Sequence, 
        classes: Sequence[str], 
        T: float = 10.):
    total_px = 0
    px_cls_count = {i: 0 for i, cls in enumerate(classes)}
    for mask in masks:
        total_px += mask.size
        for i, cls in enumerate(classes):
            px_cls_count[i] += np.sum(mask == i)
    idf = np.log(Series(px_cls_count) / total_px + 1e-5)
    
    H_i_list = []
    for mask in masks:
        cls_count = Series(mask.ravel()).value_counts()
        tf = np.array([
            cls_count.get(i, 0) / mask.size
            for i, cls in enumerate(classes)
        ])
        H_i = -np.sum(tf * idf)
        
        H_i_list.append(H_i)
    
    # Normalize, sum(probs) = 1
    H_i_list = np.array(H_i_list)
    exp_list = np.exp(H_i_list / T)
    H_i_list = exp_list / exp_list.sum()
    return H_i_list


def augment_sample(
        augmenter: v2.Compose, 
        img_transform: v2.Compose,
        image: Sequence, 
        mask: Sequence):
    i_trans, m_trans = transform_pair(augmenter,
                                      torch.from_numpy(image).reshape(1, *image.shape),
                                      torch.from_numpy(mask).reshape(1, *mask.shape))
    i_trans = img_transform(i_trans)
    return i_trans.squeeze(0).numpy(), m_trans.squeeze(0).numpy()


def augment(
        augmenter: v2.Compose,
        img_transform: v2.Compose,
        images: Sequence,
        masks: Sequence,
        classes: Sequence[str],
        n_augment: int,
        T: float,
        dst: str = 'data/aug',
        add: bool = True):
    os.makedirs(f'{dst}/imgs', exist_ok=True)
    os.makedirs(f'{dst}/masks', exist_ok=True)

    sample_dist = create_dist(masks, classes, T=T)
    to_be_sampled = np.random.choice(len(masks), size=n_augment, replace=True, p=sample_dist).tolist()

    start_idx = len([name for name in os.listdir(dst) if os.path.isfile(os.path.join(dst, name))]) if add \
                    else 0
    for i, sample_idx in tqdm(enumerate(to_be_sampled, start=start_idx), 
                              total=n_augment, 
                              desc="Augmenting"):
        image, mask = images[sample_idx], masks[sample_idx]
        new_image, new_mask = augment_sample(augmenter, img_transform, image, mask)

        cv2.imwrite(f'{dst}/imgs/aug_im{i}.jpg', new_image)
        np.save(f'{dst}/masks/aug_im{i}.npy', new_mask)


def parse_args():
    parser = argparse.ArgumentParser(description='Process raw annotations')
    parser.add_argument('--train-src', '-src', default='data/train',
                        metavar='SRC', help='Source training directory', dest='src')
    parser.add_argument('--labelmap-file', '-lbl', default='data/anno/labelmap.txt',
                        metavar='LBL', help='Label map file', dest='labelmap_file')
    parser.add_argument('--dest', '-d', default='data/aug',
                        metavar='DEST', help='Destination directory')
    parser.add_argument('--temperature', '-T', default=10.,
                        type=float, metavar='T', help='Temperature for training sampling distribution', dest='T')
    parser.add_argument('--n-augment', '-n', default=5000,
                        type=int, metavar='N', help='Number of augmentations', dest='n')
    parser.add_argument('--add', '-a', default=True,
                        action='store_true', help='Add augmentations to existing data', dest='add')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    images, masks = read_part(args.src)
    mapper, classes = read_labelmap(args.labelmap_file)    
    
    augment(config.IMG_MASK_AUGMENTER, 
            config.IMG_TRANSFORM_AFTER_AUGMENT,
            images, 
            masks, 
            classes=classes, 
            n_augment=args.n,
            T=args.T,
            dst=args.dest,
            add=args.add)