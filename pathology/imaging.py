import numpy as np
from PIL import Image, ImageDraw


def tissue_mask(rgb):
    arr = np.asarray(rgb, dtype=np.float32)
    return (arr.mean(axis=2) < 235) & ((arr.max(axis=2) - arr.min(axis=2)) > 18)


def quality(rgb):
    arr = np.asarray(rgb, dtype=np.float32)
    gray = arr.mean(axis=2)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
    lap += gray[1:-1, :-2] + gray[1:-1, 2:]
    tissue = float(tissue_mask(rgb).mean())
    focus = float(lap.var()) if lap.size else 0.0
    dark = float((gray < 45).mean())
    return {
        "tissue_fraction": round(tissue, 4),
        "focus_variance": round(focus, 2),
        "dark_fraction": round(dark, 4),
        "blur_warning": tissue > 0.1 and focus < 25,
        "dark_warning": dark > 0.08,
    }


def normalize_stain(rgb):
    """Simple tissue-only RGB moment matching; NOT a validated stain deconvolution."""
    arr = np.asarray(rgb, dtype=np.float32).copy()
    mask = tissue_mask(rgb)
    if mask.sum() < 20:
        return rgb.copy()
    pixels = arr[mask]
    target_mean = np.array([185, 132, 177], dtype=np.float32)
    target_std = np.array([35, 40, 30], dtype=np.float32)
    arr[mask] = (pixels - pixels.mean(0)) / np.maximum(pixels.std(0), 8) * target_std + target_mean
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def synthetic_slide(seed=42):
    """Generate a reproducible illustration, never patient or training data."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (2048, 1536), (251, 248, 249))
    draw = ImageDraw.Draw(im)
    draw.ellipse((180, 180, 1840, 1390), fill=(231, 179, 207))
    draw.ellipse((920, 360, 1590, 1080), fill=(209, 143, 183))
    for _ in range(6200):
        x, y = rng.integers(185, 1840), rng.integers(185, 1390)
        if ((x - 1010) / 820) ** 2 + ((y - 785) / 600) ** 2 > 1:
            continue
        r = int(rng.integers(3, 9))
        shade = int(rng.integers(75, 140))
        draw.ellipse((int(x - r), int(y - r), int(x + r), int(y + r)), fill=(shade + 25, shade, shade + 45))
    for _ in range(95):
        x, y = rng.integers(350, 1650), rng.integers(350, 1150)
        draw.ellipse((int(x), int(y), int(x + 36), int(y + 24)), fill=(247, 229, 237))
    return im
