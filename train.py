import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from utils.data_loading import DentalDataset
from utils.evaluate import evaluate

import gc
import os
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from importlib import import_module


def train(
        model: nn.Module,
        device: torch.device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        grad_accum: int = 32,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 0.,
        gradient_clipping: float = 1.):
    dir_checkpoint = Path(f'checkpoints') / model.name
    model.to(device)

    train_set = DentalDataset('train', 
                              img_transform=config.DATASET_IMG_TRANSFORM,
                              mask_transform=config.DATASET_MASK_TRANSFORM)
    val_set = DentalDataset('val', 
                            img_transform=config.DATASET_IMG_TRANSFORM,
                            mask_transform=config.DATASET_MASK_TRANSFORM)
    n_train, n_val = len(train_set), len(val_set)

    loader_args = dict(batch_size=batch_size,
                       num_workers=os.cpu_count(), 
                       pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    n_batches = len(train_loader)

    logging.info(f'''Starting training:
        Model:                  {model.name}
        Epochs:                 {epochs}
        Batch size:             {batch_size}
        Learning rate:          {learning_rate}
        Gradient accumulation:  {grad_accum}
        Weight decay:           {weight_decay}
        Gradient clipping:      {gradient_clipping}
        Training size:          {n_train}
        Validation size:        {n_val}
        Checkpoints:            {save_checkpoint}
        Device:                 {device.type}
        Mixed Precision:        {amp}
    ''')

    optimizer = config.OPTIMIZER(model.parameters(), lr=learning_rate, **config.OPTIMIZER_HYPERPARAMS)
    grad_scaler = torch.amp.GradScaler(device, enabled=amp)
    loss_fn = config.LOSS_FN(**config.LOSS_HYPERPARAMS)
    global_step = 0

    for epoch in range(1, epochs+1):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i_batch, batch in enumerate(train_loader, start=1):
                image, true_mask = batch['image'], batch['mask']
                image = image.to(device=device)
                true_mask = true_mask.to(device=device)
                n_samples = image.size(0)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    pred_mask = model(image)
                    loss = loss_fn(pred_mask, true_mask)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)

                if i_batch % grad_accum == 0 or i_batch == n_batches:
                    del image, true_mask, pred_mask
                    for param in model.parameters():
                        param.grad /= grad_accum
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                pbar.update(n_samples)
                pbar.set_postfix(**{'loss (batch)': f'{loss.item():.4f}'})
                global_step += 1

                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    evals = evaluate(model, val_loader, config.IDX2LABEL, device, amp)
                    for name, eval in evals.items():
                        print(f'\n{name}')
                        print(eval)
        
        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f'epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', metavar='M',
                        help='Model option', required=True)
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-wd', metavar='WD', default=0.,
                        type=float, help='Weight decay', dest='weight_decay')
    parser.add_argument('--grad-accum', '-ga', metavar='GA', default=32,
                        type=int, help='Accumulate gradient over batches', dest='grad_accum')
    parser.add_argument('--gradient-clipping', '-gc', metavar='GC', default=1.,
                        type=float, help='Gradient clipping', dest='gradient_clipping')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float,
                        default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true',
                        default=False, help='Use mixed precision')
    parser.add_argument('--checkpointing', '-chkp', default=True,
                        action='store_true', help='Checkpoint after every epoch')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    hyperparams = config.MODEL_HYPERPARAMS.get(args.model, dict())
    model = import_module(f'models.{args.model}').Model(**hyperparams)

    train(model=model,
          device=device,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          grad_accum=args.grad_accum,
          save_checkpoint=args.checkpointing,
          amp=args.amp,
          weight_decay=args.weight_decay,
          gradient_clipping=args.gradient_clipping)