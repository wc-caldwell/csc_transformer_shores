import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from transformers import SegformerImageProcessor
import matplotlib.pyplot as plt

SENSOR_SPEC = {
    "S2": dict(rgb=(3,2,1), nodata = None),
    "L8": dict(rgb=(3,2,1), nodata = 0),
    "L9": dict(rgb=(3,2,1), nodata = 0),
    "L7": dict(rgb=(3,2,1), nodata = 0),
    "L5": dict(rgb=(3,2,1), nodata = 0),
    "planet": dict(rgb=(3,2,1), nodata = 0)
}

def build_processor():
    """Pin the processor"""
    return SegformerImageProcessor(
        do_resize=False,
        do_rescale=True,
        do_normalize=True,
    )

def build_train_aug():
    return A.Compose([
        A.Rotate(angle_range=(-5, 5), p=0.5, border_mode=0, fill=0, fill_mask=255),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05),
                 rotate=(-5, 5), p=0.5, border_mode=0, fill=0, fill_mask=0),
        A.HorizontalFlip(p=0.5),                                
        A.VerticalFlip(p=0.5),                                  
        A.RandomBrightnessContrast(brightness_range=(-0.1, 0.1), contrast_range=(-0.2, 0.2), p=0.5),
    ])

def visualize_sample(config, dataset, idx=0, show_augmented=True):
    """Visualize a sample from the dataset."""
    
    # Load original image for display
    image_np = np.ascontiguousarray(np.array(Image.open(dataset.image_paths[idx]).convert("RGB")))
    label_np = np.ascontiguousarray(np.array(Image.open(dataset.label_paths[idx]).convert("L")))

    rows = [("Original", image_np, label_np)]
    aug = dataset.aug

    if show_augmented:
        out = aug(image=image_np, mask=label_np)
        rows.append(("Augmented", out["image"], out["mask"]))
    
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 5 * len(rows)), squeeze=False)
    for r, (name, img, lbl) in enumerate(rows):
        # Create colored label visualization
        label_colored = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        for c, color in enumerate(config.CLASS_COLORS):
            label_colored[lbl == c] = color
            
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{name} image")
        axes[r, 1].imshow(label_colored)
        axes[r, 1].set_title(f"{name} label")
        for ax in axes[r]:
            ax.axis("off")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=np.array(c) / 255, label=n)
        for n, c in zip(config.CLASS_NAMES, config.CLASS_COLORS)
    ]
    axes[0, 1].legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    whitewater = label_np == 1
    sediment = label_np == 2
    print(f"Whitewater pixels: {whitewater.sum():,} ({whitewater.mean()*100:.2f}%)\n"
          f"Sediment pixels:   {sediment.sum():,} ({sediment.mean()*100:.2f}%)")

class CoastalSegDataset(Dataset):
    def __init__(self, image_paths, label_paths, num_classes=4,tile_size=512, processor=None, augment=False):
        assert len(image_paths) == len(label_paths), "image/label count mismatch"
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.processor = processor if processor is not None else build_processor()
        self.aug = build_train_aug() if augment else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("L")

        # resize (image bilinear, label nearest)
        image = image.resize((self.tile_size, self.tile_size), Image.BILINEAR)
        label = label.resize((self.tile_size, self.tile_size), Image.NEAREST)

        image_np = np.array(image, dtype=np.uint8, order="C")   # (H, W, 3)
        label_np = np.array(label, dtype=np.uint8, order="C")   # (H, W)
        label_np[label_np >= self.num_classes] = 255             # out-of-range -> ignore

        # augment (image + mask together)
        if self.aug is not None:
            out = self.aug(image=image_np, mask=label_np)
            image_np = np.ascontiguousarray(out["image"])
            label_np = np.ascontiguousarray(out["mask"])

        # processor: rescale + normalize -> (3, H, W)
        pixel_values = self.processor(images=image_np, return_tensors="pt")["pixel_values"].squeeze(0)

        labels = torch.from_numpy(np.ascontiguousarray(label_np)).long()  # full-res

        return {"pixel_values": pixel_values, "labels": labels}

def analyze_class_distribution(config, label_paths, sample: bool = False, 
                               num_samples=None, plot: bool = True):
    """Analyze class distribution in labels."""
    class_counts = np.zeros(config.NUM_CLASSES)

    if sample:
        rng = np.random.default_rng(2)
        label_paths = rng.choice(label_paths, min(num_samples, len(label_paths)), replace=False)

    for path in label_paths:
        label = np.array(Image.open(path).convert("L"))
        for c in range(config.NUM_CLASSES):
            class_counts[c] += np.sum(label == c)

    total = class_counts.sum()
    percentages = (class_counts / total) * 100

    print("\nClass Distribution:")
    for i, (name, pct, count) in enumerate(zip(config.CLASS_NAMES, percentages, class_counts)):
        print(f"  {i}: {name:12s} - {pct:6.2f}% ({int(count):,} pixels)")

    freqs = class_counts / class_counts.sum()
    median_freq = np.median(freqs)
    weights = median_freq / freqs

    print(f"\nSuggested class weights: {np.round(weights, 2).tolist()}")

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        bars = ax.bar(config.CLASS_NAMES, percentages, color='steelblue')

        ax.set_ylabel('Pixel %', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=14, fontweight='bold')
        ax.set_title('Class Distribution (% of pixels)', fontsize=13, fontweight='bold')

        ax.tick_params(axis='both', labelsize=13)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)

        for bar, pct in zip(bars, percentages):
            ax.annotate(f'{pct:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)

        plt.tight_layout()
        plt.show()

    return class_counts, weights