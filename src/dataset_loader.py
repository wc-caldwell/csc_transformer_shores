import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from transformers import SegformerImageProcessor
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from skimage import exposure, img_as_ubyte
from skimage.io import imsave

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
    """Augmentations matching the JSON config (small rot / zoom / shift + flips).
    No 90-degree rotations: coastlines have a natural land/sea orientation."""
    return A.Compose([
        # A.Rotate(limit=5, p=0.5, border_mode=0, value=0, mask_value=255),
        A.Rotate(limit=5, p=0.5, border_mode=0),                 # AUG_ROT 5
        # A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05),
        #          rotate=(-5, 5), p=0.5, cval=0, cval_mask=255, mode=0),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05),
                 rotate=(-5, 5), p=0.5),
        A.HorizontalFlip(p=0.5),                                 # AUG_HFLIP
        A.VerticalFlip(p=0.5),                                   # AUG_VFLIP
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.2, p=0.5),
    ])

def visualize_sample(config, dataset, idx=0):
    """Visualize a sample from the dataset."""
    sample = dataset[idx]
    
    # Load original image for display
    image = Image.open(dataset.image_paths[idx]).convert("RGB")
    label = Image.open(dataset.label_paths[idx]).convert("L")
    
    # Create colored label visualization
    label_np = np.array(label)
    label_colored = np.zeros((*label_np.shape, 3), dtype=np.uint8)
    for c, color in enumerate(config.CLASS_COLORS):
        label_colored[label_np == c] = color
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(label_colored)
    axes[1].set_title("Label (Colored)")
    axes[1].axis("off")
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=np.array(c)/255, label=n)
        for n, c in zip(config.CLASS_NAMES, config.CLASS_COLORS)
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Show whitewater pixels specifically
    whitewater_mask = label_np == 1
    sediment_mask = label_np == 2
    print(f"Whitewater pixels in this image: {whitewater_mask.sum():,} ({whitewater_mask.mean()*100:.2f}%)\nSediment pixels in this image: {sediment_mask.sum():,} ({sediment_mask.mean()*100:.2f}%)")


class CoastalSegDataset(Dataset):
    def __init__(
            self, 
            image_paths, 
            label_paths, 
            num_classes=4,
            tile_size=512, 
            processor=None, 
            augment=False
            ):
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

        image_np = np.array(image)                      # (H, W, 3) uint8
        label_np = np.array(label).astype(np.int64)     # (H, W)
        # label_np = np.clip(label_np, 0, self.num_classes - 1)
        bad = (label_np < 0) | (label_np >= self.num_classes)
        label_np[bad] = 255

        # augment (image + mask together)
        if self.aug is not None:
            out = self.aug(image=image_np, mask=label_np)
            image_np, label_np = out["image"], out["mask"]

        # processor: rescale + normalize -> (3, H, W)
        pixel_values = self.processor(images=image_np, return_tensors="pt")["pixel_values"].squeeze(0)

        labels = torch.from_numpy(np.ascontiguousarray(label_np)).long()  # full-res

        return {"pixel_values": pixel_values, "labels": labels}

def rescale_image_intensity(arr, mask, prob_hi=99.9):
    """Reimplimentation of CoastSat's scaling for TOA reflectance"""
    out = np.zeros(arr.shape, dtype=np.float64)
    for i in range(arr.shape[0]):
        v = arr[i][mask]
        if v.size == 0:
            continue
        prc_hi = np.percentile(v, prob_hi)
        if prc_hi <= 0:
            continue
        out[i][mask] = exposure.rescale_intensity(arr[i][mask], in_range = (0., float(prc_hi)))
    out[:, ~mask] = 0.
    return out

def geotiff_to_rgb(src_path, dst_path, satname, prob_hi=99.9, extra_mask=None, as_png=True):
    """Convert projected geotiff to 8-bit png or jpeg"""
    spec = SENSOR_SPEC[satname]
    with rasterio.open(src_path) as src:
        arr = src.read(list(spec['rgb'])).astype(np.float64)
        file_nodata = src.nodatavals

    valid = np.all(np.isfinite(arr), axis=0)
    for i,b in enumerate(spec['rgb']):
        nd = file_nodata[b-1] if file_nodata[b-1] is not None else spec['nodata']
        if nd is not None:
            valid &= (arr[i] != nd)

    if extra_mask is not None:
        valid &= ~extra_mask

    rgb = rescale_image_intensity(arr, valid, prob_hi)
    rgb8 = img_as_ubyte(np.clip(rgb, 0, 1).transpose(1,2,0))

    dst = Path(dst_path).with_suffix(".png" if as_png else ".jpg")
    imsave(dst, rgb8, **({} if as_png else {"quality": 100}))
    return dst, valid

def batch_convert(src_dir, dst_dir, satname, pattern="*.tif", as_png=True):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    return [geotiff_to_rgb(p, dst_dir / p.stem, satname, as_png=as_png)[0] for p in sorted(src_dir.glob(pattern))]

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

def geotiff_label_to_jpeg(
    src_path: str | Path,
    dst_path: str | Path,
    band: int = 1,
    nodata_value: int | None = None,
    ignore_index: int = 255,
    quality: int = 100,
) -> None:
    """Convert a single-band label GeoTIFF to JPEG, matching existing dataset convention."""
    with rasterio.open(src_path) as src:
        label = src.read(band)
        src_nodata = src.nodatavals[band - 1]

    nodata = nodata_value if nodata_value is not None else src_nodata
    if nodata is not None:
        label = np.where(label == nodata, ignore_index, label)

    label = label.astype(np.uint8)
    Image.fromarray(label, mode="L").save(dst_path, quality=quality)

def geotiff_label_to_png(
    src_path: str | Path,
    dst_path: str | Path,
    band: int = 1,
    nodata_value: int | None = None,
    ignore_index: int = 255,
    nclasses: int | None = None,
) -> None:
    """Convert a single-band label GeoTIFF to lossless PNG."""
    with rasterio.open(src_path) as src:
        label = src.read(band)
        src_nodata = src.nodatavals[band - 1]

    nodata = nodata_value if nodata_value is not None else src_nodata
    if nodata is not None:
        label = np.where(label == nodata, ignore_index, label)

    assert label.min() >= 0 and label.max() <= 255, f"{src_path}: {label.min()}..{label.max()}"
    label = label.astype(np.uint8)

    u = np.unique(label)
    off = u[(u >= 4) & (u != ignore_index)]
    assert off.size == 0, f"{src_path}: unexpected labels {off}"

    Image.fromarray(label, mode="L").save(dst_path, optimize=True)

def batch_convert_labels(
    src_dir: str | Path,
    dst_dir: str | Path,
    pattern: str = "*.tif",
    nodata_value: int | None = None,
    ignore_index: int = 255,
    quality: int = 100,
) -> list[Path]:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for src_path in sorted(src_dir.glob(pattern)):
        dst_path = dst_dir / (src_path.stem + ".png")
        geotiff_label_to_jpeg(src_path, dst_path, nodata_value=nodata_value, ignore_index=ignore_index, quality=quality)
        out_paths.append(dst_path)
    return out_paths