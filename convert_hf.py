import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # SET_GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
from src import dataset_loader, inference_help
import json
from transformers import SegformerForSemanticSegmentation
import pandas as pd
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

import argparse
import itertools

def convert(seed, backbone, loss, data_dir):
    with open(Path.cwd() / 'ini_training_config.json') as config_data:
        config_dict = json.load(config_data)

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(**config_dict)
    config.SEED = int(seed)        
    config.LOSS = loss
    config.IMAGES_DIR = data_dir / "images"
    config.LABELS_DIR = data_dir / "labels"
    config.OUTPUT_DIR = sorted((data_dir / "trained_models").glob(f"mit-{backbone}_{seed}_{loss}_*"))[0]
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit-{backbone}",
        num_labels=config.NUM_CLASSES,
        id2label={i: str(i) for i in range(config.NUM_CLASSES)},
        label2id={str(i): i for i in range(config.NUM_CLASSES)},
        classifier_dropout_prob=config.DROPOUT,
        hidden_dropout_prob=config.DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(config.OUTPUT_DIR / 'best.pt', map_location=device))
    model.save_pretrained(config.OUTPUT_DIR / 'best_hf', map_location=device)


# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================
# CLI Arguments to execute this file

def parse_args():
    p = argparse.ArgumentParser(description="SegFormer Inference Loop")
    p.add_argument("--seed", type=str, nargs="+", required = True, choices=["42", "43", "44", "45", "46"])
    p.add_argument("--backbone", type=str, nargs="+", required = True, default="b0", choices=["b0", "b1", "b2", "b3", "b4", "b5"])
    p.add_argument("--loss", type=str, nargs="+", required = True, choices=["CCE", "wCCE", "Focal", "Dice", "GenDice", "Combo", "Tversky", "FocalTversky", "AsymmetricUnifiedFocal", "LovaszSoftmax"])
    p.add_argument("--data-dir", type=Path, required=True, help="Folder where training data is stored. Images in images and Labels in labels subdirectories.")
    return p.parse_args()

# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================

if __name__ == "__main__":
    args = parse_args()

for seed, backbone, loss in itertools.product(args.seed, args.backbone, args.loss):
    print(f"Converting seed={seed} backbone={backbone} loss={loss}")
    try:
        convert(seed, backbone, loss, args.data_dir)  # your existing conversion logic
    except FileNotFoundError as e:
        print(f"  skipped: {e}")