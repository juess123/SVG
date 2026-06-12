from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "text.png"
DEFAULT_FONT_DIR = Path("C:/Windows/Fonts")
DEFAULT_EXTRA_FONT_DIR = SCRIPT_DIR / "data" / "fonts"


@dataclass
class TextItem:
    text: str
    x: float
    y: float
    width: float
    height: float
    fill: str
    score: float
    vertical: bool = False
    font_family: str = ""


@dataclass(frozen=True)
class FontCandidate:
    family: str
    path: Path


def polygon_bounds(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(round(float(channel))) for channel in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def choose_font_family(item: TextItem) -> str:
    text_len = max(1, len(item.text))
    avg_width = item.width / text_len
    is_brown = item.fill.lower() not in {"#333333", "#424141", "#424241"} and item.fill.lower().startswith("#9")
    is_large = item.height >= 90 or avg_width >= 90

    if item.vertical and text_len > 2 and avg_width < 45:
        return "SimHei, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    if is_large and (text_len == 1 or item.vertical or is_brown):
        return "KaiTi, DFKai-SB, STKaiti, Microsoft YaHei, serif"
    if item.vertical:
        return "SimHei, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    if item.height >= 32 and is_brown:
        return "SimHei, Microsoft YaHei, Noto Sans CJK SC, sans-serif"
    return "Microsoft YaHei, Noto Sans SC, Noto Sans CJK SC, SimHei, Arial, sans-serif"


def font_family_chain(primary: str, fallback: str) -> str:
    names: list[str] = []
    for value in [primary, *fallback.split(",")]:
        name = value.strip()
        if name and name not in names:
            names.append(name)
    return ", ".join(names)


def collect_font_candidates(font_dir: Path) -> list[FontCandidate]:
    if not font_dir.is_dir():
        return []

    candidates: list[FontCandidate] = []
    seen: set[tuple[str, Path]] = set()
    for path in sorted(font_dir.iterdir()):
        if path.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
            continue
        try:
            if path.stat().st_size < 500_000:
                continue
            font = ImageFont.truetype(str(path), 32)
            family, style = font.getname()
        except Exception:
            continue

        if not family:
            continue
        family_lower = family.lower()
        path_lower = path.name.lower()
        cjk_hint = (
            "yahei",
            "jhenghei",
            "simhei",
            "simsun",
            "fangsong",
            "kaiti",
            "dfkai",
            "dengxian",
            "noto",
            "source han",
            "黑",
            "宋",
            "楷",
            "仿",
            "等线",
            "雅黑",
            "ma shan zheng",
            "long cang",
            "liu jian mao cao",
            "zhi mang xing",
            "zcool",
            "xiaowei",
            "qingke",
            "kuaile",
            "yuji",
            "hina",
            "reggae",
            "rocknroll",
            "rampart",
            "stick",
            "shippori",
            "kaisei",
        )
        if not any(key in family_lower or key in path_lower for key in cjk_hint):
            continue
        key = (family, path)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(FontCandidate(family=family, path=path))
    return candidates


def collect_font_candidates_from_dirs(font_dirs: list[Path]) -> list[FontCandidate]:
    candidates: list[FontCandidate] = []
    seen: set[tuple[str, Path]] = set()
    for font_dir in font_dirs:
        for candidate in collect_font_candidates(font_dir):
            key = (candidate.family, candidate.path)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
    return candidates


def crop_foreground_mask(image: np.ndarray, item: TextItem) -> np.ndarray:
    h, w = image.shape[:2]
    left = max(0, int(np.floor(item.x)))
    top = max(0, int(np.floor(item.y)))
    right = min(w, int(np.ceil(item.x + item.width)))
    bottom = min(h, int(np.ceil(item.y + item.height)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=bool)
    return np.max(crop, axis=2) < 235


def trim_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255


def render_text_mask(item: TextItem, font_path: Path) -> np.ndarray:
    if item.vertical:
        font_size = max(8, int(round(item.width * 0.82)))
        target_w = max(1, int(round(item.width)))
        target_h = max(1, int(round(item.height)))
    else:
        font_size = max(8, int(round(item.height * 0.92)))
        target_w = max(1, int(round(item.width)))
        target_h = max(1, int(round(item.height)))

    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        return np.zeros((1, 1), dtype=np.uint8)

    canvas = Image.new("L", (max(64, target_w * 3), max(64, target_h * 3)), 0)
    draw = ImageDraw.Draw(canvas)

    if item.vertical and len(item.text) > 1:
        y = 0
        line_step = max(1, target_h / len(item.text))
        for char in item.text:
            bbox = draw.textbbox((0, 0), char, font=font)
            char_w = bbox[2] - bbox[0]
            x = max(0, (target_w - char_w) / 2)
            draw.text((x - bbox[0], y - bbox[1]), char, fill=255, font=font)
            y += line_step
    else:
        bbox = draw.textbbox((0, 0), item.text, font=font)
        draw.text((-bbox[0], -bbox[1]), item.text, fill=255, font=font)

    return trim_mask(np.array(canvas) > 0)


def mask_similarity(original: np.ndarray, rendered: np.ndarray) -> float:
    original_trimmed = trim_mask(original)
    if original_trimmed.size <= 1 or rendered.size <= 1:
        return 0.0

    target = Image.fromarray(original_trimmed).convert("L")
    candidate = Image.fromarray(rendered).convert("L").resize(target.size, Image.Resampling.LANCZOS)
    a = np.array(target) > 0
    b = np.array(candidate) > 64
    intersection = np.count_nonzero(a & b)
    total = np.count_nonzero(a) + np.count_nonzero(b)
    if total == 0:
        return 0.0
    return 2.0 * intersection / total


def likely_font_candidates(item: TextItem, candidates: list[FontCandidate]) -> list[FontCandidate]:
    text_len = max(1, len(item.text))
    avg_width = item.width / text_len
    is_large = item.height >= 90 or avg_width >= 90
    is_small_vertical = item.vertical and text_len > 2 and avg_width < 45
    preferred: list[FontCandidate] = []
    fallback: list[FontCandidate] = []

    for candidate in candidates:
        name = candidate.family.lower()
        file_name = candidate.path.name.lower()
        combined = f"{name} {file_name}"

        if is_large and not is_small_vertical:
            if any(
                key in combined
                for key in (
                    "kaiti",
                    "dfkai",
                    "楷",
                    "ma shan zheng",
                    "long cang",
                    "liu jian mao cao",
                    "zhi mang xing",
                    "zcool",
                    "xiaowei",
                    "qingke",
                    "kuaile",
                    "yuji",
                    "hina",
                    "reggae",
                    "rocknroll",
                    "rampart",
                    "stick",
                    "shippori",
                    "kaisei",
                )
            ):
                preferred.append(candidate)
            continue

        if item.height >= 30 or is_small_vertical:
            if any(key in combined for key in ("simhei", "yahei", "jhenghei", "noto", "dengxian", "黑", "雅黑", "等线")):
                preferred.append(candidate)
            continue

        if any(key in combined for key in ("yahei", "jhenghei", "noto", "dengxian", "simhei", "黑", "雅黑", "等线")):
            preferred.append(candidate)
        elif any(key in combined for key in ("simsun", "fangsong", "宋", "仿")):
            fallback.append(candidate)

    selected = preferred + [c for c in fallback if c not in preferred]
    return selected or candidates


def match_item_font(image: np.ndarray, item: TextItem, candidates: list[FontCandidate]) -> str:
    if not candidates:
        return choose_font_family(item)

    original = crop_foreground_mask(image, item)
    best_family = choose_font_family(item)
    best_score = 0.0
    for candidate in likely_font_candidates(item, candidates):
        rendered = render_text_mask(item, candidate.path)
        score = mask_similarity(original, rendered)
        if score > best_score:
            best_score = score
            best_family = font_family_chain(candidate.family, choose_font_family(item))
    return best_family


def font_primary(font_family: str) -> str:
    return font_family.split(",", 1)[0].strip()


def font_group_key(item: TextItem) -> tuple[str, str, int]:
    color = item.fill.lower()
    if color.startswith("#9") or color.startswith("#8"):
        tone = "brown"
    elif color.startswith("#5") or color.startswith("#6") or color.startswith("#7"):
        tone = "gray"
    else:
        tone = "other"

    if item.vertical:
        direction = "vertical"
        size = item.width
    else:
        direction = "horizontal"
        size = item.height

    if size >= 90:
        bucket = 100
    elif size >= 45:
        bucket = 50
    elif size >= 28:
        bucket = 32
    else:
        bucket = 20
    return direction, tone, bucket


def unify_similar_size_fonts(items: list[TextItem]) -> None:
    groups: dict[tuple[str, str, int], list[TextItem]] = {}
    for item in items:
        groups.setdefault(font_group_key(item), []).append(item)

    for group_items in groups.values():
        if len(group_items) < 2:
            continue

        counts: dict[str, int] = {}
        family_by_primary: dict[str, str] = {}
        for item in group_items:
            primary = font_primary(item.font_family)
            counts[primary] = counts.get(primary, 0) + 1
            family_by_primary.setdefault(primary, item.font_family)

        chosen_primary = max(
            counts,
            key=lambda primary: (
                counts[primary],
                sum(item.score for item in group_items if font_primary(item.font_family) == primary),
            ),
        )
        chosen_family = family_by_primary[chosen_primary]
        for item in group_items:
            item.font_family = chosen_family


def estimated_text_units(text: str) -> float:
    units = 0.0
    for char in text:
        codepoint = ord(char)
        if char.isspace():
            units += 0.35
        elif char.isascii():
            units += 0.55
        elif (
            0x3000 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0xFF00 <= codepoint <= 0xFFEF
        ):
            units += 1.0
        else:
            units += 0.8
    return max(1.0, units)


def fitted_horizontal_font_size(item: TextItem) -> float:
    height_size = item.height * 0.78
    width_size = item.width / estimated_text_units(item.text) * 0.98
    return max(6.0, min(height_size, width_size))


def dominant_text_color(image: np.ndarray, bounds: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = bounds
    h, w = image.shape[:2]
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(w, int(np.ceil(x2)))
    bottom = min(h, int(np.ceil(y2)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return "#333333"

    pixels = crop.reshape(-1, 3).astype(np.int16)
    # Ignore the near-white background and very light anti-aliased pixels.
    mask = np.max(pixels, axis=1) < 235
    foreground = pixels[mask]
    if len(foreground) == 0:
        return "#333333"

    red_mask = (foreground[:, 0] > foreground[:, 1] + 35) & (foreground[:, 0] > foreground[:, 2] + 35)
    if np.count_nonzero(red_mask) > len(foreground) * 0.25:
        color_pixels = foreground[red_mask]
    else:
        color_pixels = foreground

    return rgb_to_hex(np.median(color_pixels, axis=0))


def split_by_detected_color_regions(
    text: str,
    score: float,
    image: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> list[TextItem]:
    x1, y1, x2, y2 = bounds
    h, w = image.shape[:2]
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(w, int(np.ceil(x2)))
    bottom = min(h, int(np.ceil(y2)))
    crop = image[top:bottom, left:right].astype(np.int16)
    if crop.size == 0:
        return []

    red = (crop[:, :, 0] > crop[:, :, 1] + 35) & (crop[:, :, 0] > crop[:, :, 2] + 35) & (crop[:, :, 0] < 245)
    dark = (np.max(crop, axis=2) < 190) & ~red

    red_cols = np.where(np.sum(red, axis=0) > 1)[0]
    dark_cols = np.where(np.sum(dark, axis=0) > 1)[0]
    if len(red_cols) == 0 or len(dark_cols) == 0:
        return []

    dark_min, dark_max = int(dark_cols.min()), int(dark_cols.max())
    red_min, red_max = int(red_cols.min()), int(red_cols.max())
    if red_min <= dark_max + 8:
        return []

    midpoint = (dark_max + red_min) / 2
    split_index = max(1, min(len(text) - 1, round(len(text) * midpoint / max(1, right - left))))
    first_text = text[:split_index]
    second_text = text[split_index:]
    box_h = y2 - y1

    return [
        TextItem(
            text=first_text,
            x=left + dark_min,
            y=y1,
            width=max(1.0, dark_max - dark_min + 1),
            height=box_h,
            fill=dominant_text_color(image, (left + dark_min, y1, left + dark_max + 1, y2)),
            score=score,
            vertical=False,
        ),
        TextItem(
            text=second_text,
            x=left + red_min,
            y=y1,
            width=max(1.0, red_max - red_min + 1),
            height=box_h,
            fill=dominant_text_color(image, (left + red_min, y1, left + red_max + 1, y2)),
            score=score,
            vertical=False,
        ),
    ]


def split_wide_spaced_text(
    text: str,
    score: float,
    image: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> list[TextItem]:
    if len(text) < 2:
        return []

    x1, y1, x2, y2 = bounds
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w / len(text) < box_h * 1.25:
        return []

    h, w = image.shape[:2]
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(w, int(np.ceil(x2)))
    bottom = min(h, int(np.ceil(y2)))
    crop = image[top:bottom, left:right].astype(np.int16)
    if crop.size == 0:
        return []

    fg = np.max(crop, axis=2) < 235
    col_has_text = np.sum(fg, axis=0) > 2
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, has_text in enumerate(col_has_text):
        if has_text and start is None:
            start = idx
        elif not has_text and start is not None:
            if idx - start > 3:
                runs.append((start, idx - 1))
            start = None
    if start is not None and len(col_has_text) - start > 3:
        runs.append((start, len(col_has_text) - 1))

    items: list[TextItem] = []
    if len(runs) != len(text):
        cell_w = box_w / len(text)
        fallback_w = min(box_h * 0.95, cell_w * 0.72)
        runs = []
        for index in range(len(text)):
            center = (index + 0.5) * cell_w
            run_start = max(0, int(round(center - fallback_w / 2)))
            run_end = min(right - left - 1, int(round(center + fallback_w / 2)))
            runs.append((run_start, run_end))

    for char, (run_start, run_end) in zip(text, runs):
        char_bounds = (left + run_start, y1, left + run_end + 1, y2)
        items.append(
            TextItem(
                text=char,
                x=float(left + run_start),
                y=y1,
                width=float(run_end - run_start + 1),
                height=box_h,
                fill=dominant_text_color(image, char_bounds),
                score=score,
                vertical=False,
            )
        )
    return items


def ocr_to_items(
    input_path: Path,
    min_score: float,
    *,
    match_fonts: bool = True,
    font_dirs: list[Path] | None = None,
) -> tuple[int, int, list[TextItem]]:
    image = np.array(Image.open(input_path).convert("RGB"))
    height, width = image.shape[:2]
    result, _ = RapidOCR()(str(input_path))
    items: list[TextItem] = []

    for points, text, score in result or []:
        if score < min_score or not text.strip():
            continue
        bounds = polygon_bounds(points)
        split_items = split_by_detected_color_regions(text.strip(), score, image, bounds)
        if split_items:
            items.extend(split_items)
            continue
        split_items = split_wide_spaced_text(text.strip(), score, image, bounds)
        if split_items:
            items.extend(split_items)
            continue

        x1, y1, x2, y2 = bounds
        text_value = text.strip()
        items.append(
            TextItem(
                text=text_value,
                x=x1,
                y=y1,
                width=max(1.0, x2 - x1),
                height=max(1.0, y2 - y1),
                fill=dominant_text_color(image, bounds),
                score=score,
                vertical=(y2 - y1) > (x2 - x1) * 1.8 and len(text_value) > 1,
            )
        )

    items.sort(key=lambda item: (item.y, item.x))
    font_dirs = font_dirs or [DEFAULT_FONT_DIR, DEFAULT_EXTRA_FONT_DIR]
    font_candidates = collect_font_candidates_from_dirs(font_dirs) if match_fonts else []
    if match_fonts:
        dirs = ", ".join(str(path) for path in font_dirs)
        print(f"Loaded {len(font_candidates)} font candidate(s) from {dirs}")
    for item in items:
        item.font_family = match_item_font(image, item, font_candidates) if match_fonts else choose_font_family(item)
    unify_similar_size_fonts(items)
    return width, height, items


def svg_text(item: TextItem) -> str:
    escaped = html.escape(item.text)
    if item.vertical:
        font_size = max(8.0, item.width * 0.82)
        x = item.x + item.width * 0.5
        y = item.y
        return (
            f'  <text x="{x:.2f}" y="{y:.2f}" '
            f'font-family="{item.font_family}" '
            f'font-size="{font_size:.2f}" font-weight="800" fill="{item.fill}" '
            f'writing-mode="vertical-rl" textLength="{item.height:.2f}" '
            f'lengthAdjust="spacing">{escaped}</text>'
        )

    font_size = fitted_horizontal_font_size(item)
    baseline = item.y + item.height * 0.82
    return (
        f'  <text x="{item.x:.2f}" y="{baseline:.2f}" '
        f'font-family="{item.font_family}" '
        f'font-size="{font_size:.2f}" font-weight="800" fill="{item.fill}" '
        f'textLength="{item.width:.2f}" lengthAdjust="spacing">{escaped}</text>'
    )


def write_svg(
    input_path: Path,
    output_path: Path,
    min_score: float,
    include_background: bool,
    *,
    match_fonts: bool = True,
    font_dirs: list[Path] | None = None,
    print_items: bool = False,
) -> Path:
    width, height, items = ocr_to_items(input_path, min_score, match_fonts=match_fonts, font_dirs=font_dirs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    ]
    if include_background:
        lines.append('  <rect width="100%" height="100%" fill="#ffffff"/>')
    lines.extend(svg_text(item) for item in items)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Recognized {len(items)} text component(s)")
    if print_items:
        for item in items:
            print(f"{item.text}\t{item.fill}\t{item.score:.3f}\t{item.font_family}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize text in an image and write it as SVG <text> components.")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Input image path.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output SVG path. Default: <input>.svg")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum OCR confidence to keep.")
    parser.add_argument("--background", action="store_true", help="Add a white SVG background rectangle.")
    parser.add_argument(
        "--font-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan for font matching. Can be used multiple times. Default: Windows fonts + data/fonts.",
    )
    parser.add_argument("--no-font-match", action="store_true", help="Use heuristic font families without image matching.")
    parser.add_argument("--print-items", action="store_true", help="Print every recognized OCR text item.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".svg")
    try:
        result = write_svg(
            input_path,
            output_path,
            args.min_score,
            args.background,
            match_fonts=not args.no_font_match,
            font_dirs=args.font_dir,
            print_items=args.print_items,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Done: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
