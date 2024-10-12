import torch
from torchvision.transforms import v2
from utils.loss import MulticlassCombinationLoss

DATASET_IMG_TRANSFORM = v2.Compose([
    v2.ToDtype(torch.float),
    v2.Resize(256),
    v2.Normalize(mean=[0], std=[255])
])

DATASET_MASK_TRANSFORM = v2.Compose([
    v2.ToDtype(torch.long),
    v2.Resize(256)
])

MODEL_HYPERPARAMS = {
    'unet': dict(bn_momentum=0.999,
                 bilinear=False)
}

OPTIMIZER = torch.optim.Adam
OPTIMIZER_HYPERPARAMS = dict()

LOSS_FN = MulticlassCombinationLoss
LOSS_HYPERPARAMS = dict(w_dice=1.,
                        w_focal=1.,
                        reduction='mean',
                        dice_kwargs={'smooth': 1},
                        focal_kwargs={'alpha': 0.3, 'gamma': 3.5})

IDX2LABEL = {
    0: 'background',
    1: 'bone',
    2: 'cavity',
    3: 'crown',
    4: 'dental-implant',
    5: 'dental-implant-crown',
    6: 'dentin',
    7: 'enamel',
    8: 'filling-metal',
    9: 'filling-non-metal',
    10: 'periapical-radiolucence',
    11: 'pulp',
    12: 'root canal',
    13: 'sinus'
}
N_CLASS = len(IDX2LABEL)

IMG_MASK_AUGMENTER = v2.Compose([
    v2.RandomAffine(degrees=10, 
                    scale=(0.95, 1.05), 
                    fill=1),
    v2.RandomHorizontalFlip(),
    v2.RandomResizedCrop((660, 844), 
                         scale=(0.7, 1))
])

IMG_TRANSFORM_AFTER_AUGMENT = v2.Compose([
    v2.ToDtype(torch.float, scale=True),
    v2.ColorJitter(brightness=(0.8, 1.2),
                   contrast=(0.8, 1.2)),
    v2.RandomEqualize(),
    v2.RandomApply([v2.GaussianNoise()],
                    p=0.3),
    v2.ToDtype(torch.uint8, scale=True)
])