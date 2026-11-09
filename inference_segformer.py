import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
from src import inference_help
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
# Function to loop through list of CLI arguments, so we can submit one job to complete all inference runs

def run_inference(seed, backbone, loss, data_dir):
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
    config.OUTPUT_DIR = sorted((data_dir / "trained_models").glob(f"mit-{backbone}_{seed}_{loss}*"))[0]
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = SegformerForSemanticSegmentation.from_pretrained(config.OUTPUT_DIR / 'best_hf').to(device).eval()

    test_loader = torch.load(data_dir / "test_loader.pt", weights_only=False)

    # NOTE: 0=mIou, 1=water IoU, 2=whitewater IoU, 3=sediment IoU, 4=other IoU
    test_ious = {}
    test_dices = {}

    for idx, path in enumerate(test_loader.dataset.image_paths):
        per_class_iou, miou, per_class_dice, mdice = inference_help.predict_and_visualize(config, model, test_loader.dataset, idx, device, show_plot=False, whole_scene=True)

        test_ious[Path(path).stem] = [miou] + [v for _,v in per_class_iou.items()]
        test_dices[Path(path).stem] = [mdice] + [v for _,v in per_class_dice.items()]

    os.makedirs(config.OUTPUT_DIR / "test_inference", exist_ok=True)

    iou_df = pd.DataFrame.from_dict(
            test_ious, 
            orient='index', 
            columns=['MeanIoU', 'WaterIoU', 'WhitewaterIoU', 'SedimentIoU', 'OtherIoU']
            ).reset_index().rename(columns={'index': 'Image_ID'})

    iou_df.to_csv(config.OUTPUT_DIR / "test_inference" / "iou_metrics_testing.csv")

    dice_df = pd.DataFrame.from_dict(
            test_dices, 
            orient='index', 
            columns=['MeanDice', 'WaterDice', 'WhitewaterDice', 'SedimentDice', 'OtherDice']
            ).reset_index().rename(columns={'index': 'Image_ID'})

    dice_df.to_csv(config.OUTPUT_DIR / "test_inference" / "dice_metrics_testing.csv")

# ==================================================================================================================================
# ==================================================================================================================================
# ==================================================================================================================================

if __name__ == "__main__":
    args = parse_args()

for seed, backbone, loss in itertools.product(args.seed, args.backbone, args.loss):
    print(f"Running inference with seed={seed} backbone={backbone} loss={loss}")
    try:
        run_inference(seed, backbone, loss, args.data_dir)  # your existing conversion logic
    except FileNotFoundError as e:
        print(f"  skipped: {e}")