"""Bounded WSI access. Coordinates are always expressed at level 0."""

import math
import warnings
from pathlib import Path

from PIL import Image

RASTER = {".png", ".jpg", ".jpeg"}
SUPPORTED = RASTER | {".svs", ".ndpi", ".tif", ".tiff"}


class Slide:
    def __init__(self, path: Path):
        self.reader = None
        if path.suffix.lower() in RASTER:
            # Ordinary raster images must fit the pixel budget; WSI use OpenSlide.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as source:
                    if source.width * source.height > 40_000_000:
                        raise ValueError("普通图片超过 4000 万像素，请使用 WSI 格式。")
                    self.raster = source.convert("RGB")
            self.dimensions = self.raster.size
            self.level_dimensions = [self.dimensions]
            self.downsamples = [1.0]
        else:
            try:
                import openslide
            except (ImportError, OSError) as exc:
                raise ValueError("WSI 需要 OpenSlide，请安装 pip install -e '.[wsi]'。") from exc
            self.reader = openslide.OpenSlide(str(path))
            self.dimensions = self.reader.dimensions
            self.level_dimensions = list(self.reader.level_dimensions)
            self.downsamples = list(self.reader.level_downsamples)

    def thumbnail(self, max_size=1200):
        if self.reader:
            return self.reader.get_thumbnail((max_size, max_size)).convert("RGB")
        result = self.raster.copy()
        result.thumbnail((max_size, max_size))
        return result

    def read(self, x, y, level, size):
        if self.reader:
            region = self.reader.read_region((x, y), level, size)
            background = Image.new("RGBA", size, "white")
            background.alpha_composite(region)
            return background.convert("RGB")
        return self.raster.crop((x, y, x + size[0], y + size[1]))

    def close(self):
        if self.reader:
            self.reader.close()
        else:
            self.raster.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tile_grid(width, height, tile_size, max_tiles):
    """Uniform spatial subsampling without allocating a whole-slide grid."""
    nx, ny = math.ceil(width / tile_size), math.ceil(height / tile_size)
    total = nx * ny
    count = min(total, max_tiles)
    # Includes both ends; uses O(max_tiles) memory even for a gigapixel slide.
    indices = [0] if count == 1 else [round(i * (total - 1) / (count - 1)) for i in range(count)]
    for index in indices:
        x, y = (index % nx) * tile_size, (index // nx) * tile_size
        yield x, y, min(tile_size, width - x), min(tile_size, height - y)
