import math
import time

from PIL import Image, ImageDraw

from .imaging import normalize_stain, quality
from .model import load_predictor
from .slides import Slide, tile_grid

LIMITATIONS = [
    "仅供研究与工程演示；结果不能用于诊断。",
    "默认使用颜色纹理规则，未训练或验证肿瘤识别能力。",
    "质控仅包含模糊与深色区域提示，尚未实现经标注验证的折叠、气泡检测。",
    "切块采用限量均匀采样，未采样区域保持透明；低分不能排除病变。",
]


def analyze(path, output_dir, settings, options, progress=lambda *_: None):
    started = time.monotonic()
    trace = []

    def step(name, message, percent):
        trace.append({"tool": name, "message": message})
        progress(percent, name)

    predictor = load_predictor(settings)
    with Slide(path) as slide:
        level = options["level"]
        if level >= len(slide.level_dimensions):
            raise ValueError(f"切片只有 {len(slide.level_dimensions)} 层，不能读取第 {level} 层。")
        thumb = slide.thumbnail()
        thumb.save(output_dir / "thumbnail.png")
        step("read_slide", "读取切片元数据与缩略图", 10)
        qc = quality(thumb)
        step("quality_control", "计算组织占比、清晰度和深色区域比例", 20)
        width, height = slide.level_dimensions[level]
        downsample = slide.downsamples[level]
        grid = list(tile_grid(width, height, options["tile_size"], options["max_tiles"]))
        total = math.ceil(width / options["tile_size"]) * math.ceil(height / options["tile_size"])
        tiles = []
        skipped = 0
        for index, (x, y, w, h) in enumerate(grid):
            x0, y0 = round(x * downsample), round(y * downsample)
            tile = slide.read(x0, y0, level, (w, h))
            tile_qc = quality(tile)
            if tile_qc["tissue_fraction"] < 0.1:
                skipped += 1
            else:
                processed = normalize_stain(tile) if options["normalize"] else tile
                score = predictor.predict(processed)
                tiles.append(
                    {
                        "id": len(tiles),
                        "x": x0,
                        "y": y0,
                        "width": min(round(w * downsample), slide.dimensions[0] - x0),
                        "height": min(round(h * downsample), slide.dimensions[1] - y0),
                        "score": round(score, 4),
                        "qc": tile_qc,
                    }
                )
            progress(20 + int((index + 1) / len(grid) * 60), "tile_inference")
        step("tile_inference", f"抽样 {len(grid)} 块，分析 {len(tiles)} 块，背景过滤 {skipped} 块", 82)
        overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        sx, sy = thumb.width / slide.dimensions[0], thumb.height / slide.dimensions[1]
        for tile in tiles:
            s = tile["score"]
            box = (
                tile["x"] * sx,
                tile["y"] * sy,
                (tile["x"] + tile["width"]) * sx,
                (tile["y"] + tile["height"]) * sy,
            )
            draw.rectangle(box, fill=(int(255 * s), int(180 * (1 - s)), 90, 100), outline=(255, 255, 255, 90))
        overlay.save(output_dir / "heatmap.png")
        step("render_heatmap", "在 level-0 坐标上绘制采样区域叠加层", 90)
        ranked = sorted(tiles, key=lambda t: t["score"], reverse=True)[:8]
        limitations = list(LIMITATIONS)
        if predictor.mode == "torch":
            limitations[1] = "使用本地 PyTorch 研究模型；本仓库没有提供临床验证或校准证据。"
        report = {
            "title": "病理切片研究分析报告",
            "mode": predictor.mode,
            "score_label": predictor.label,
            "slide": {
                "width": slide.dimensions[0],
                "height": slide.dimensions[1],
                "levels": len(slide.level_dimensions),
                "analysis_level": level,
            },
            "quality": qc,
            "sampling": {
                "total_grid_tiles": total,
                "sampled_tiles": len(grid),
                "analyzed_tiles": len(tiles),
                "background_tiles": skipped,
                "grid_coverage": round(len(grid) / total, 4),
                "is_full_grid": len(grid) == total,
            },
            "summary": f"完成 {len(tiles)} 个组织图块的研究分析。"
            + (
                "当前为规则演示模式，分数仅反映颜色纹理。"
                if predictor.mode == "demo"
                else "当前为本地研究模型模式，阳性类别分数需要独立验证。"
            ),
            "regions": ranked,
            "tiles": tiles,
            "limitations": limitations,
            "normalization": "tissue_rgb_moment_matching" if options["normalize"] else "none",
            "options": options,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        step("generate_report", "生成可追溯结构化报告与高分区域索引", 100)
        report["trace"] = trace
        return report
