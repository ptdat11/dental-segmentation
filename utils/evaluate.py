import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassConfusionMatrix
import config

from pandas import DataFrame
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, idx2label, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    f1 = MulticlassF1Score(num_classes=config.N_CLASS, average='none').to(device)
    iou = MulticlassJaccardIndex(num_classes=config.N_CLASS, average='none').to(device)
    conf = MulticlassConfusionMatrix(num_classes=config.N_CLASS, normalize='true').to(device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32,
                             memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long).squeeze(-3)

            # predict the mask
            mask_pred = net(image)

            f1.update(mask_pred, mask_true)
            iou.update(mask_pred, mask_true)
            conf.update(mask_pred.argmax(dim=-3).ravel(), mask_true.ravel())

    net.train()

    scores = DataFrame({
        'F1 Score': f1.compute().cpu().numpy(),
        'IoU': iou.compute().cpu().numpy()
    }).T
    scores.rename(columns=idx2label, inplace=True)
    scores['Macro-avg'] = scores.mean(axis=1)

    conf_mat = DataFrame(conf.compute().cpu().numpy())
    conf_mat.rename(index=idx2label, columns=idx2label, inplace=True)
    conf_mat.columns = conf_mat.columns.map(lambda s: f'Pred {s}')
    conf_mat.index = conf_mat.index.map(lambda s: f'True {s}')

    return {
        'Scores': scores,
        'Confusion Matrix': conf_mat
    }
