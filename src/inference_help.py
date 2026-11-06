import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor

def _no_resize_processor():
    """Pin data processor for our imagery"""
    return SegformerImageProcessor(do_resize=False, do_rescale=True, do_normalize=True)


def _starts(total, size, stride):
    """Window origins covering [0, total), last window flush to the edge."""
    if total <= size:
        return [0]
    xs = list(range(0, total - size + 1, stride))
    if xs[-1] != total - size:
        xs.append(total - size)
    return xs

@torch.no_grad()
def _sliding_window_predict(model, pixel_values, n_classes, valid, tile=512, stride=None):
    """pixel_values: (1, C, H, W) normalized. Returns (H, W) int64 class map."""
    stride = stride or int(tile * 0.75)
    device = pixel_values.device
    _, _, H, W = pixel_values.shape
    valid_t = torch.as_tensor(np.asarray(valid), device=device, dtype=torch.float32)

    acc = torch.zeros((n_classes, H, W), dtype=torch.float32, device=device)
    wsum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    for y0 in _starts(H, tile, stride):
        for x0 in _starts(W, tile, stride):
            ph = min(tile, H - y0)
            pw = min(tile, W - x0)
            patch = pixel_values[:, :, y0:y0 + ph, x0:x0 + pw]

            if (ph, pw) != (tile, tile):
                patch = F.pad(patch, (0, tile - pw, 0, tile - ph), mode="replicate")

            logits = model(pixel_values=patch).logits          # (1, K, tile/4, tile/4)
            logits = F.interpolate(logits.float(), size=(tile, tile),
                                   mode="bilinear", align_corners=False)
            logits = logits[:, :, :ph, :pw]

            acc[:, y0:y0 + ph, x0:x0 + pw] += logits[0]
            wsum[:, y0:y0 + ph, x0:x0 + pw] += 1.0

    pred = (acc / wsum.clamp_min(1.0)).argmax(0)
    pred[wsum[0] == 0] = 255
    return pred.cpu().numpy().astype(np.int64)

@torch.no_grad()
def _whole_scene_predict(model, image, n_classes, tile=512):
    """Predict on whole scene instead of moving window"""
    W, H = image.size
    small = image.resize((tile, tile), Image.BILINEAR)

    proc = _no_resize_processor()
    # px = proc(image=np.array(small), return_tensors='pt')['pixel_values'].to_device(model.device)
    px = proc(images=np.array(small), return_tensors='pt')['pixel_values'].to(model.device)
    
    logits = model(pixel_values=px).logits
    logits = F.interpolate(logits.float(), size=(H,W), mode='bilinear', align_corners=False)

    return logits[0].argmax(0).cpu().numpy().astype(np.int64)

def _colorize(arr, config, ignore_color=(0, 0, 0)):
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[...] = np.asarray(ignore_color, dtype=np.uint8)
    for c, color in enumerate(config.CLASS_COLORS):
        out[arr == c] = color
    return out

def _valid_mask(image, label=None):
    """True where there is real data. Catches both image-side padding (all-zero
    RGB) and label-side no-data."""
    valid = np.array(image).sum(axis=-1) > 0
    if label is not None:
        valid &= (label != 255)
    return valid

@torch.no_grad()
def predict_and_visualize(config, model, dataset, idx, device, show_plot=True, whole_scene=True, save_prediction=False):
    """Visualize prediction for a single sample and report per-class IoU, Dice, and their means."""
    model.eval()

    n_classes = config.NUM_CLASSES
    class_names = getattr(config, "CLASS_NAMES", [str(i) for i in range(n_classes)])
    tile = getattr(config, "TILE_SIZE", 512)

    # Load original image and label
    image = Image.open(dataset.image_paths[idx]).convert("RGB")
    label = np.array(Image.open(dataset.label_paths[idx]).convert("L")).astype(np.int64)

    # Flag anything outside the legal class range as ignore rather than clipping it
    label[(label < 0) | ((label >= n_classes) & (label != 255))] = 255

    valid = _valid_mask(image, label)

    if whole_scene:
        pred = _whole_scene_predict(model, image, n_classes, tile=tile)
        pred[~valid] = 255
    else:
        # Prepare for model, native resolution, no resize
        proc = _no_resize_processor()
        pixel_values = proc(images=np.array(image), return_tensors="pt")["pixel_values"].to(device)

        # Predict with a sliding window at native resolution
        pred = _sliding_window_predict(model, pixel_values, n_classes, valid, tile=tile)
        pred[~valid] = 255

    # Create colored visualizations
    label_colored = _colorize(label, config)
    pred_colored = _colorize(pred, config)

    # Whitewater overlay (visual diagnostic), restricted to valid pixels
    whitewater_gt = (label == 1) & valid
    whitewater_pred = (pred == 1) & valid

    diff = np.zeros((*label.shape, 3), dtype=np.uint8)
    diff[whitewater_gt & whitewater_pred] = [0, 255, 0]     # True positive  - green
    diff[whitewater_gt & ~whitewater_pred] = [255, 0, 0]    # False negative - red
    diff[~whitewater_gt & whitewater_pred] = [255, 165, 0]  # False positive - orange

    # Plot (single row, 4 columns)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(label_colored)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_colored)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(diff)
    axes[3].set_title("Whitewater: TP(green) FN(red) FP(orange)")
    axes[3].axis("off")

    # Per-class IoU and Dice for this single image, over VALID pixels only
    per_class_iou = np.full(n_classes, np.nan)
    per_class_dice = np.full(n_classes, np.nan)
    for c in range(n_classes):
        pred_c = (pred == c) & valid
        label_c = (label == c) & valid
        intersection = (pred_c & label_c).sum()
        union = (pred_c | label_c).sum()
        pred_sum = pred_c.sum()
        label_sum = label_c.sum()

        if union > 0:
            per_class_iou[c] = intersection / union
        # if union == 0, class doesn't appear in this image at all -> leave as NaN

        if (pred_sum + label_sum) > 0:
            per_class_dice[c] = (2 * intersection) / (pred_sum + label_sum)

    image_miou = np.nanmean(per_class_iou)
    image_mdice = np.nanmean(per_class_dice)

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    if save_prediction:
        plt.savefig(config.OUTPUT_DIR / f'prediction_sample_{idx}.png', dpi=150, bbox_inches='tight')

    return (
        dict(zip(class_names, per_class_iou)),
        image_miou,
        dict(zip(class_names, per_class_dice)),
        image_mdice,
    )