import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from src import inference_help

class ConfusionMeter:
    """Accumulates a pixel-level confusion matrix across batches.

    Counting pixels across the whole epoch (rather than averaging per-batch
    metrics) is the correct way to report IoU/F1/etc -- per-batch means are
    biased, especially for rare classes that may be absent from some batches.

    Matrix convention: cm[i, j] = number of pixels with true label i and
    predicted label j (rows = ground truth, cols = predictions).
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.cm = torch.zeros((self.num_classes, self.num_classes))

    @torch.no_grad()
    def update(self, preds, target):
        # preds, target: (B, H, W) long, on any device
        preds = preds.view(-1)
        target = target.view(-1)

        # ignore any pixels outside the valid class range (e.g. ignore_index)
        valid = (target >= 0) & (target < self.num_classes)
        preds = preds[valid]
        target = target[valid]

        idx = target * self.num_classes + preds
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += binc.reshape(self.num_classes, self.num_classes).double().cpu()

    def confusion_matrix(self):
        return self.cm  # (num_classes, num_classes), rows=true, cols=pred

    def iou_per_class(self):
        inter = self.cm.diag()
        union = self.cm.sum(0) + self.cm.sum(1) - inter
        iou = inter / union.clamp(min=1e-9)
        iou[union == 0] = float("nan")
        return iou

    def mean_iou(self):
        iou = self.iou_per_class()
        valid = ~torch.isnan(iou)
        return iou[valid].mean().item() if valid.any() else float("nan")
    
    def freq_iou(self):
        """Frequency Weighted IoU"""
        # Compute per-class IoU
        iou = self.iou_per_class()

        # Calculate the total number of pixels for each class
        sum_true = self.cm.sum(1)

        # Frequency of each class
        freq = sum_true / sum_true.sum().clamp(min=1e-9)

        # Weighted IoU
        weighted_iou = iou * freq

        valid = ~torch.isnan(weighted_iou)
        return weighted_iou[valid].sum().item() if valid.any() else float("nan")

    def mcc_per_class(self):
        tp = self.cm.diag()
        fn = self.cm.sum(0) - tp
        fp = self.cm.sum(1) - tp
        tn = self.cm.sum() - (tp + fp + fn)
        
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator.clamp(min=1e-9)
        mcc[denominator == 0] = float("nan")
        return mcc
    
    def mean_mcc(self):
        mcc = self.mcc_per_class()
        valid = ~torch.isnan(mcc)
        return mcc[valid].mean().item() if valid.any() else float("nan")

    def precision_per_class(self):
        tp = self.cm.diag()
        pred_pos = self.cm.sum(0)  # column sums
        prec = tp / pred_pos.clamp(min=1e-9)
        prec[pred_pos == 0] = float("nan")
        return prec

    def recall_per_class(self):
        tp = self.cm.diag()
        actual_pos = self.cm.sum(1)  # row sums
        rec = tp / actual_pos.clamp(min=1e-9)
        rec[actual_pos == 0] = float("nan")
        return rec

    def f1_per_class(self):
        p = self.precision_per_class()
        r = self.recall_per_class()
        f1 = 2 * p * r / (p + r).clamp(min=1e-9)
        f1[(p + r) == 0] = float("nan")
        return f1

    def accuracy(self):
        # overall pixel accuracy
        total = self.cm.sum()
        correct = self.cm.diag().sum()
        return (correct / total.clamp(min=1e-9)).item() if total > 0 else float("nan")

    def mean_precision(self):
        return self._nanmean(self.precision_per_class())

    def mean_recall(self):
        return self._nanmean(self.recall_per_class())

    def mean_f1(self):
        return self._nanmean(self.f1_per_class())

    @staticmethod
    def _nanmean(x):
        valid = ~torch.isnan(x)
        return x[valid].mean().item() if valid.any() else float("nan")

@torch.no_grad()
def evaluate(model, loader, criterion, config, device):
    model.eval()
    # meter = IoUMeter(config.NUM_CLASSES)
    meter = ConfusionMeter(config.NUM_CLASSES)
    total_loss, n = 0.0, 0

    for batch in loader:
        imgs = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["labels"].to(device, non_blocking=True).long()

        logits = model(pixel_values=imgs).logits                      # (B,4,128,128)
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode="bilinear", align_corners=False).contiguous()  # (B,4,512,512)

        total_loss += criterion(logits, masks).item()
        n += 1

        preds = logits.argmax(dim=1)                                  # (B,512,512)
        meter.update(preds.cpu(), masks.cpu())

    avg_loss = total_loss / max(n, 1)
    return avg_loss, meter.iou_per_class(), meter.mean_iou(), meter.freq_iou(), meter.accuracy(), meter.f1_per_class(), meter.mean_f1(), meter.mean_mcc(), meter.mcc_per_class(), meter.recall_per_class(), meter.mean_recall(), meter.precision_per_class(), meter.mean_precision()


def _fmt_iou(iou_tensor, class_names):
    parts = []
    for name, v in zip(class_names, iou_tensor.tolist()):
        parts.append(f"{name}={v:.4f}" if not np.isnan(v) else f"{name}=  n/a")
    return "  ".join(parts)

# Training
def train(model, train_loader, val_loader, test_loader,
          criterion, optimizer, scheduler, config,
          device, ckpt_path=None):

    if ckpt_path is None:
        ckpt_path = str(config.OUTPUT_DIR / "best.pt")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    WATER, WHITEWATER, SEDIMENT, OTHER = 0, 1, 2, 3  # class indices

    # will use this to store the metrics for each epoch for training history plots
    # metrics should be loss, 

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_miou': [],
        'val_miou': [],
        'train_freq_iou': [],
        'val_freq_iou': [],
        'train_mean_mcc': [],
        'val_mean_mcc': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1_score': [],
        'val_f1_score': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
        'train_water_iou': [],
        'train_ww_iou': [],
        'train_sediment_iou': [],
        'train_other_iou': [],
        'val_water_iou': [],
        'val_ww_iou': [],
        'val_sediment_iou': [],
        'val_other_iou': [],
        'train_water_f1': [],
        'train_ww_f1': [],
        'train_sediment_f1': [],
        'train_other_f1': [],
        'val_water_f1': [],
        'val_ww_f1': [],
        'val_sediment_f1': [],
        'val_other_f1': [],
        'train_water_precision': [],
        'train_ww_precision': [],
        'train_sediment_precision': [],
        'train_other_precision': [],
        'val_water_precision': [],
        'val_ww_precision': [],
        'val_sediment_precision': [],
        'val_other_precision': [],
        'train_water_recall': [],
        'train_ww_recall': [],
        'train_sediment_recall': [],
        'train_other_recall': [],
        'val_water_recall': [],
        'val_ww_recall': [],
        'val_sediment_recall': [],
        'val_other_recall': [],
        'train_water_mcc': [],
        'train_ww_mcc': [],
        'train_sediment_mcc': [],
        'train_other_mcc': [],
        'val_water_mcc': [],
        'val_ww_mcc': [],
        'val_sediment_mcc': [],
        'val_other_mcc': []
        }
    
    test_metrics = {
        'loss': [],
        'miou': [],
        'freq_iou': [],
        'accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': [],
        'mean_mcc': [],
        'water_iou': [],
        'ww_iou': [],
        'sediment_iou': [],
        'other_iou': [],
        'water_f1': [],
        'ww_f1': [],
        'sediment_f1': [],
        'other_f1': [],
        'water_precision': [],
        'ww_precision': [],
        'sediment_precision': [],
        'other_precision': [],
        'water_recall': [],
        'ww_recall': [],
        'sediment_recall': [],
        'other_recall': [],
        'water_mcc': [],
        'ww_mcc': [],
        'sediment_mcc': [],
        'other_mcc': []
        }

    best_val_miou = -float("inf")
    wait = 0

    for epoch in range(config.MAX_EPOCHS):
        # train
        model.train()
        train_loss, steps = 0.0, 0
        for batch in train_loader:
            imgs = batch["pixel_values"].to(device, non_blocking=True)
            masks = batch["labels"].to(device, non_blocking=True).long()

            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False).contiguous()
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

        train_loss /= max(steps, 1)
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # train IoU
        _, train_iou, train_miou, train_freq_iou, train_accuracy, train_f1_per_class, train_mean_f1, train_mean_mcc, train_mcc_per_class, train_recall_per_class, train_mean_recall, train_precision_per_class, train_mean_precision = evaluate(
            model, train_loader, criterion, config, device)

        # validate
        val_loss, val_iou, val_miou, val_freq_iou, val_accuracy, val_f1_per_class, val_mean_f1, val_mean_mcc, val_mcc_per_class, val_recall_per_class, val_mean_recall, val_precision_per_class, val_mean_precision = evaluate(
            model, val_loader, criterion, config, device)

        print(f"\nEpoch {epoch+1:3d}/{config.MAX_EPOCHS}  lr={cur_lr:.2e}")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={val_miou:.4f}")
        print(f"  val IoU : {_fmt_iou(val_iou, config.CLASS_NAMES)}")
        print(f"  >> whitewater IoU={val_iou[WHITEWATER]:.4f}  "
              f"sediment IoU={val_iou[SEDIMENT]:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        history['train_freq_iou'].append(train_freq_iou)
        history['val_freq_iou'].append(val_freq_iou)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_f1_score'].append(train_mean_f1)
        history['val_f1_score'].append(val_mean_f1)
        history['train_precision'].append(train_mean_precision)
        history['val_precision'].append(val_mean_precision)
        history['train_recall'].append(train_mean_recall)
        history['val_recall'].append(val_mean_recall)
        history['train_mean_mcc'].append(train_mean_mcc)
        history['val_mean_mcc'].append(val_mean_mcc)

        # Per-class IoU 
        history['train_water_iou'].append(train_iou[WATER])
        history['train_ww_iou'].append(train_iou[WHITEWATER])
        history['train_sediment_iou'].append(train_iou[SEDIMENT])
        history['train_other_iou'].append(train_iou[OTHER])
        history['val_water_iou'].append(val_iou[WATER])
        history['val_ww_iou'].append(val_iou[WHITEWATER])
        history['val_sediment_iou'].append(val_iou[SEDIMENT])
        history['val_other_iou'].append(val_iou[OTHER])

        # Per-class F1-Score (Dice)
        history['train_water_f1'].append(train_f1_per_class[WATER])
        history['train_ww_f1'].append(train_f1_per_class[WHITEWATER])
        history['train_sediment_f1'].append(train_f1_per_class[SEDIMENT])
        history['train_other_f1'].append(train_f1_per_class[OTHER])
        history['val_water_f1'].append(val_f1_per_class[WATER])
        history['val_ww_f1'].append(val_f1_per_class[WHITEWATER])
        history['val_sediment_f1'].append(val_f1_per_class[SEDIMENT])
        history['val_other_f1'].append(val_f1_per_class[OTHER])

        # Per-class precision
        history['train_water_precision'].append(train_precision_per_class[WATER])
        history['train_ww_precision'].append(train_precision_per_class[WHITEWATER])
        history['train_sediment_precision'].append(train_precision_per_class[SEDIMENT])
        history['train_other_precision'].append(train_precision_per_class[OTHER])
        history['val_water_precision'].append(val_precision_per_class[WATER])
        history['val_ww_precision'].append(val_precision_per_class[WHITEWATER])
        history['val_sediment_precision'].append(val_precision_per_class[SEDIMENT])
        history['val_other_precision'].append(val_precision_per_class[OTHER])

        # Per-class recall
        history['train_water_recall'].append(train_recall_per_class[WATER])
        history['train_ww_recall'].append(train_recall_per_class[WHITEWATER])
        history['train_sediment_recall'].append(train_recall_per_class[SEDIMENT])
        history['train_other_recall'].append(train_recall_per_class[OTHER])
        history['val_water_recall'].append(val_recall_per_class[WATER])
        history['val_ww_recall'].append(val_recall_per_class[WHITEWATER])
        history['val_sediment_recall'].append(val_recall_per_class[SEDIMENT])
        history['val_other_recall'].append(val_recall_per_class[OTHER])

        # Per-class MCC
        history['train_water_mcc'].append(train_mcc_per_class[WATER])
        history['train_ww_mcc'].append(train_mcc_per_class[WHITEWATER])
        history['train_sediment_mcc'].append(train_mcc_per_class[SEDIMENT])
        history['train_other_mcc'].append(train_mcc_per_class[OTHER])
        history['val_water_mcc'].append(val_mcc_per_class[WATER])
        history['val_ww_mcc'].append(val_mcc_per_class[WHITEWATER])
        history['val_sediment_mcc'].append(val_mcc_per_class[SEDIMENT])
        history['val_other_mcc'].append(val_mcc_per_class[OTHER])

        # checkpoint on val mean-IoU
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved new best (val_mIoU={val_miou:.4f}) -> {ckpt_path}")
        else:
            wait += 1
            if wait >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no val_mIoU improvement for {config.PATIENCE} epochs).")
                break

    # final test eval on best checkpoint 
    print("\n" + "=" * 60)
    print("Loading best checkpoint and evaluating on TEST set...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_iou, test_miou, test_freq_iou, test_accuracy, test_f1_per_class, test_mean_f1, test_mean_mcc, test_mcc_per_class, test_recall_per_class, test_mean_recall, test_precision_per_class, test_mean_precision = evaluate(
        model, test_loader, criterion, config, device)

    print(f"TEST  loss={test_loss:.4f}  mIoU={test_miou:.4f}")
    print(f"TEST  IoU : {_fmt_iou(test_iou, config.CLASS_NAMES)}")
    print(f"TEST  >> whitewater IoU={test_iou[WHITEWATER]:.4f}  "
          f"sediment IoU={test_iou[SEDIMENT]:.4f}")
    print("=" * 60)

    test_metrics['loss'].append(test_loss)
    test_metrics['miou'].append(test_miou)
    test_metrics['freq_iou'].append(test_freq_iou)
    test_metrics['accuracy'].append(test_accuracy)
    test_metrics['f1_score'].append(test_mean_f1)
    test_metrics['precision'].append(test_mean_precision)
    test_metrics['recall'].append(test_mean_recall)
    test_metrics['mean_mcc'].append(test_mean_mcc)

    # Per-class IoU 
    test_metrics['water_iou'].append(test_iou[WATER])
    test_metrics['ww_iou'].append(test_iou[WHITEWATER])
    test_metrics['sediment_iou'].append(test_iou[SEDIMENT])
    test_metrics['other_iou'].append(test_iou[OTHER])

    # Per-class F1-Score (Dice)
    test_metrics['water_f1'].append(test_f1_per_class[WATER])
    test_metrics['ww_f1'].append(test_f1_per_class[WHITEWATER])
    test_metrics['sediment_f1'].append(test_f1_per_class[SEDIMENT])
    test_metrics['other_f1'].append(test_f1_per_class[OTHER])

    # Per-class precision
    test_metrics['water_precision'].append(test_precision_per_class[WATER])
    test_metrics['ww_precision'].append(test_precision_per_class[WHITEWATER])
    test_metrics['sediment_precision'].append(test_precision_per_class[SEDIMENT])
    test_metrics['other_precision'].append(test_precision_per_class[OTHER])

    # Per-class recall
    test_metrics['water_recall'].append(test_recall_per_class[WATER])
    test_metrics['ww_recall'].append(test_recall_per_class[WHITEWATER])
    test_metrics['sediment_recall'].append(test_recall_per_class[SEDIMENT])
    test_metrics['other_recall'].append(test_recall_per_class[OTHER])

    # Per-class mcc
    test_metrics['water_mcc'].append(test_mcc_per_class[WATER])
    test_metrics['ww_mcc'].append(test_mcc_per_class[WHITEWATER])
    test_metrics['sediment_mcc'].append(test_mcc_per_class[SEDIMENT])
    test_metrics['other_mcc'].append(test_mcc_per_class[OTHER])

    return {
        "best_val_miou": best_val_miou,
        "test_miou": test_miou,
        "test_iou_per_class": test_iou.tolist(),
        "ckpt_path": ckpt_path,
    }, history, test_metrics

def training_history_plot(config, history):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val')
    axes[0, 1].set_title('Pixel Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mean IoU
    axes[0, 2].plot(epochs, history['train_miou'], 'b-', label='Train')
    axes[0, 2].plot(epochs, history['val_miou'], 'r-', label='Val')
    axes[0, 2].set_title('Mean IoU')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)


    # F1-Scroe (Dice)
    axes[1, 0].plot(epochs, history['train_f1_score'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_f1_score'], 'r-', label='Val')
    axes[1, 0].set_title('F1-Score (Dice)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision
    axes[1, 1].plot(epochs, history['train_precision'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_precision'], 'r-', label='Val')
    axes[1, 1].set_title('Precision')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Recall
    axes[1, 2].plot(epochs, history['train_recall'], 'b-', label='Train')
    axes[1, 2].plot(epochs, history['val_recall'], 'r-', label='Val')
    axes[1, 2].set_title('Recall')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    best_idx = np.argmax(history['val_miou'])

    print(f"""
Training Summary
================

Best Epoch: {best_idx + 1} / {len(epochs)}

Best Validation Metrics:
  Loss:           {history['val_loss'][best_idx]:.4f}
  F1-Score:       {history['val_f1_score'][best_idx]:.4f}
  Precision:      {history['val_precision'][best_idx]:.4f}
  Accuracy:       {history['val_accuracy'][best_idx]:.4f}
  Recall:         {history['val_recall'][best_idx]:.4f}
  Mean IoU:       {history['val_miou'][best_idx]:.4f}

Final Validation Metrics:
  Loss:           {history['val_loss'][-1]:.4f}
  F1-Score:       {history['val_f1_score'][-1]:.4f}
  Precision:      {history['val_precision'][-1]:.4f}
  Accuracy:       {history['val_accuracy'][-1]:.4f}
  Recall:         {history['val_recall'][-1]:.4f}
  Mean IoU:       {history['val_miou'][-1]:.4f}
""")
    
    plt.suptitle(f'Training History - {config.LOSS} Loss', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_class_metric(config, history, target_metric, best_epoch):
    """Plot specific train/val metric for a target class over training epochs"""
    if target_metric not in ['Precision', 'Recall', 'IoU', 'F1-Score']:
        raise Exception("Metric must be 'Precision', 'Recall',  'IoU', or 'F1-Score'")
    
    metrics = {
        'Precision': 'precision',
        'Recall': 'recall',
        'IoU': 'iou',
        'F1-Score': 'f1'
    }
    
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Water
    axes[0, 0].plot(epochs, history[f'train_water_{metrics[target_metric]}'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history[f'val_water_{metrics[target_metric]}'], 'r-', label='Val')
    axes[0, 0].set_title(f'Water Class {target_metric}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # White Water
    axes[0, 1].plot(epochs, history[f'train_ww_{metrics[target_metric]}'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history[f'val_ww_{metrics[target_metric]}'], 'r-', label='Val')
    axes[0, 1].set_title(f'White Water Class {target_metric}')
    axes[0, 1].set_xlabel(f'Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sediment
    axes[1, 0].plot(epochs, history[f'train_sediment_{metrics[target_metric]}'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history[f'val_sediment_{metrics[target_metric]}'], 'r-', label='Val')
    axes[1, 0].set_title(f'Sediment Class {target_metric}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Other
    axes[1, 1].plot(epochs, history[f'train_other_{metrics[target_metric]}'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history[f'val_other_{metrics[target_metric]}'], 'r-', label='Val')
    axes[1, 1].set_title(f'Other Class {target_metric}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    print(f"""
Training Summary
================

Best Epoch: {best_epoch} / {len(epochs)} (based on maximum mean IoU)

Best Validation Metrics:
  Water {target_metric}:            {history[f"val_water_{metrics[target_metric]}"][best_epoch - 1]:.4f}
  White Water {target_metric}:      {history[f"val_ww_{metrics[target_metric]}"][best_epoch - 1]:.4f}
  Sediment {target_metric}:         {history[f"val_sediment_{metrics[target_metric]}"][best_epoch - 1]:.4f}
  Other {target_metric}:            {history[f"val_other_{metrics[target_metric]}"][best_epoch - 1]:.4f}

Final Validation Metrics:
  Water {target_metric}:            {history[f"val_water_{metrics[target_metric]}"][-1]:.4f}
  White Water {target_metric}:      {history[f"val_ww_{metrics[target_metric]}"][-1]:.4f}
  Sediment {target_metric}:         {history[f"val_sediment_{metrics[target_metric]}"][-1]:.4f}
  Other {target_metric}:            {history[f"val_other_{metrics[target_metric]}"][-1]:.4f}
""")
    
    plt.suptitle(f'Training History - {config.LOSS} Loss', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    # plt.show()

def aggregate_iou_metrics(config, model, ds, processor, device):
    """Use the model, config, and test dataset to retrieve per-calss and mean IoU from each image-label pair"""
    
    test_ious = {}

    for idx, path in enumerate(ds.image_paths):
        per_class_iou, miou = inference_help.predict_and_visualize(config, model, ds, idx, processor, device, show_plot=False)
    
        test_ious[Path(path).stem] = [miou] + [v for _,v in per_class_iou.items()]


    return test_ious    # NOTE: 0=mIou, 1=water IoU, 2=whitewater IoU, 3=sediment IoU, 4=other IoU