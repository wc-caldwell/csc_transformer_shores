import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SET_GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import SegformerForSemanticSegmentation   # TODO: Try out Mask2Former in a future project. It out performed segformer in https://doi.org/10.3390/rs18020325
from datetime import datetime

if torch.cuda.is_available():
    device = "cuda" 
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

from src import dataset_loader, training_help

from src.loss_functions import (
    FocalLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    TverskyLoss,
    FocalTverskyLoss,
    AsymmetricUnifiedFocalLoss,
    ComboLoss,
    LovaszSoftmaxLoss
)

import argparse

def parse_args():
    p = argparse.ArgumentParser(description="SegFormer Fine-Tuning Loop")
    p.add_argument("--seed", type=int, required = True, choices=["7", "42", "300", "1337", "12345"])
    p.add_argument("--backbone", type=str, required = True, default="b0", choies=["b0", "b1", "b2", "b3", "b4", "b5"]
    p.add_argument("--loss", type=str, required = True, choices=["CCE", "wCCE", "Focal", "Dice", "GenDice", "Combo", "Tversky", "FocalTversky", "AsymmetricUnifiedFocal", "LovaszSoftmax"])
    p.add_argument("--images", type=Path, required = True, help="Path to the folder where the RGB images are stored")
    p.add_argument("--labels", type=Path, required = True, help="Path to the folder where the labels corresponding to the RGB images are stored")
    p.add_argument("--out-dir", type=Path, required = True, help="Path to the location where you would like the training outputs to be stored")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

with open(Path.cwd() / 'ini_training_config.json') as config_data:
    config_dict = json.load(config_data)

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

config = Config(**config_dict)
config.SEED = args.seed        
config.LOSS = args.loss
config.IMAGES_DIR = args.images
config.LABELS_DIR = args.labels
config.OUTPUT_DIR = args.out_dir / "trained_models" / f'mit-{args.backbone}_{config.SEED}_{config.LOSS}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

LOSSES = {
    "CCE":     nn.CrossEntropyLoss(ignore_index=255),
    "wCCE":     nn.CrossEntropyLoss(weight = None, ignore_index=255),                   # NOTE: weight tensor determined using median freq
    "Focal":  FocalLoss(gamma=2.0, weight=None, ignore_index=255, reduction='mean'),
    "Dice":   DiceLoss(config.NUM_CLASSES, smooth=1.),
    'GenDice': GeneralizedDiceLoss(config.NUM_CLASSES, weight_mode='batch', smooth=1.),
    "Combo": ComboLoss(config.NUM_CLASSES, beta=0.6, alpha=0.5, smooth=1.),
    "Tversky":TverskyLoss(alpha=0.3, beta=0.7, smooth=1.),
    "FocalTversky": FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75, smooth=1.),
    "AsymmetricUnifiedFocal": AsymmetricUnifiedFocalLoss(config.NUM_CLASSES, rare_classes=[1,2], lam=0.5, delta=0.6, gamma=0.5, smooth=1.),
    "LovaszSoftmax": LovaszSoftmaxLoss(config.NUM_CLASSES, per_image=True)
}

# establish loss function based on specific CLI argument
criterion = LOSSES[config.LOSS]
np.random.seed(config.SEED)

torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

# load the data used for training, testing, and validation
if (Path.cwd() / 'data' / 'train_loader.pt').exists():
    train_loader = torch.load(Path.cwd() / 'data' / 'train_loader.pt', weights_only=False)
    test_loader = torch.load(Path.cwd() / 'data' / 'test_loader.pt', weights_only=False)
    val_loader = torch.load(Path.cwd() / 'data' / 'val_loader.pt', weights_only=False)

else:
    image_paths = sorted([
        str(config.IMAGES_DIR / f) 
        for f in os.listdir(config.IMAGES_DIR) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    label_paths = [
        Path(config.LABELS_DIR / f"{Path(im).stem}_label.jpg") 
        for im in image_paths
    ]
    
    # Verify all label files exist
    valid_pairs = [(img, lbl) for img, lbl in zip(image_paths, label_paths) if os.path.exists(lbl)]
    image_paths, label_paths = zip(*valid_pairs) if valid_pairs else ([], [])
        
    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        list(image_paths),
        list(label_paths),
        test_size=1-config.VAL_SPLIT,
        random_state=config.SEED,
        shuffle=True
    )
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
        random_state=config.SEED,
        shuffle=True
    )
    
    processor = dataset_loader.build_processor()
    
    train_ds = dataset_loader.CoastalSegDataset(X_train, y_train,
                                 num_classes=config.NUM_CLASSES, tile_size=config.TILE_SIZE,
                                 processor=processor, augment=True)
    
    test_ds = dataset_loader.CoastalSegDataset(X_test, y_test,
                                 num_classes=config.NUM_CLASSES, tile_size=config.TILE_SIZE,
                                 processor=processor, augment=False)

    val_ds = dataset_loader.CoastalSegDataset(X_val, y_val,
                                 num_classes=config.NUM_CLASSES, tile_size=config.TILE_SIZE,
                                 processor=processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=os.cpu_count() - 4, pin_memory=(device == 'cuda'))

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=os.cpu_count() - 4, pin_memory=(device == 'cuda'))

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=os.cpu_count() - 4, pin_memory=(device == 'cuda'))

    torch.save(train_loader, Path.cwd() / 'data' / 'train_loader.pt')
    torch.save(test_loader, Path.cwd() / 'data' / 'test_loader.pt')
    torch.save(val_loader, Path.cwd() / 'data' / 'val_loader.pt')

    del X_train, y_train, X_test, X_val, y_test, y_val, train_ds, test_ds, val_ds, image_paths, label_paths, valid_pairs

# If loss function is weighted CCE, calculate weights based on median frequency
if config.LOSS == 'wCCE':
    class_counts, suggested_weights = dataset_loader.analyze_class_distribution(config, train_loader.dataset.label_paths, plot=False)

    class_weights = torch.as_tensor(suggested_weights, dtype=torch.float32, device=device)
    config.LOSS_WEIGHTS = class_weights

    # replace the criterion which was established in the second code cell, since that was initialized without the determined class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
else:
    print(f"Loss function is {config.LOSS}, so heursitic class weighting will not be applied")

model = SegformerForSemanticSegmentation.from_pretrained(
    f"nvidia/mit-{args.backbone}",
    num_labels=config.NUM_CLASSES,
    id2label={i: str(i) for i in range(config.NUM_CLASSES)},
    label2id={str(i): i for i in range(config.NUM_CLASSES)},
    classifier_dropout_prob=config.DROPOUT,
    hidden_dropout_prob=config.DROPOUT,
).to(device)


# Establish learning rate and optimizer for training
def lr_lambda(epoch):
    if epoch < config.RAMPUP_EPOCHS:
        lr = (config.MAX_LR - config.START_LR) / config.RAMPUP_EPOCHS * epoch + config.START_LR
    elif epoch < config.RAMPUP_EPOCHS + config.SUSTAIN_EPOCHS:
        lr = config.MAX_LR  
    else:
        lr = (config.MAX_LR - config.MIN_LR) * config.EXP_DECAY ** (
              epoch - config.RAMPUP_EPOCHS - config.SUSTAIN_EPOCHS) + config.MIN_LR
    return lr / config.MAX_LR 

optimizer = torch.optim.AdamW(model.parameters(), lr=config.MAX_LR)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# run training loop
results, history, test_metrics = training_help.train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, config, device=device)
print(results)

# generate training history plot(s)
training_help.training_history_plot(config, history)

# save update config file, model history, and testing metrics for future use
config_dict = {k: v for k, v in vars(config.__class__).items() if k.isupper()}

with open(config.OUTPUT_DIR / 'model_config.json', 'w') as config_file:
        json.dump(config_dict, config_file, indent=4, default=str)

with open(config.OUTPUT_DIR / 'model_history.json', 'w') as history_file:
        json.dump(history, history_file, indent=4, default = float)

with open(config.OUTPUT_DIR / 'test_metrics.json', 'w') as test_metrics_file:
        json.dump(test_metrics, test_metrics_file, indent=4, default = float)