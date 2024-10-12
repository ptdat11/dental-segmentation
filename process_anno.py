import cv2
import numpy as np
from utils.utils import read_anno
from sklearn.model_selection import train_test_split

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process raw annotations')
    parser.add_argument('--src', default='data/anno',
                        metavar='SRC', help='Source annotation directory')
    parser.add_argument('--train', '-tr', default=0.7,
                        type=float, help='Training proportion')
    parser.add_argument('--random-state', '-seed', default=42,
                        type=int, help='Random state', dest='random_state')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    raw_data = read_anno('data/anno')
    train_idx, val_test_idx = train_test_split(range(len(raw_data['images'])),
                                           test_size=1-args.train, 
                                           random_state=args.random_state)
    val_idx, test_idx = train_test_split(val_test_idx, 
                                         test_size=0.5,
                                         random_state=args.random_state)
    
    indices = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    for dir, idx in indices.items():
        dst = Path('data/') / dir
        (dst / 'masks').mkdir(parents=True, exist_ok=True)
        (dst / 'imgs').mkdir(parents=True, exist_ok=True)

        for i, i_im in enumerate(idx):
            image = raw_data['images'][i_im]
            cv2.imwrite(dst / 'imgs' / f'im{i}.jpg', image)

        for i, i_ma in enumerate(idx):
            mask = raw_data['masks'][i_ma]
            np.save(dst / 'masks' / f'im{i}.npy', mask)