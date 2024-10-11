import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, ConfusionMatrix

from pandas import DataFrame
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, idx2label, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    f1 = MulticlassF1Score(num_classes=15, average='none')
    iou = MulticlassJaccardIndex(num_classes=15, average='none')
    conf = ConfusionMatrix(num_classes=15)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32,
                             memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            f1.update(mask_pred, mask_true)
            iou.update(mask_pred, mask_true)
            conf.update(mask_pred.ravel(), mask_true.ravel())

    net.train()

    scores = DataFrame({
        'F1 Score': f1.compute().numpy(),
        'IoU': iou.compute().numpy()
    })
    scores.rename(columns=idx2label, inplace=True)
    scores['Macro-avg'] = scores.mean(axis=1)

    conf_mat = DataFrame(conf.compute().numpy())
    conf_mat.rename(index=idx2label, columns=idx2label, inplace=True)
    conf_mat.columns = conf_mat.columns.map(lambda s: f'Pred {s}')
    conf_mat.index = conf_mat.index.map(lambda s: f'True {s}')

    return {
        'Scores': scores,
        'Confusion Matrix': conf_mat
    }
