from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from flatten_lab_kmeans import flatten_lab_kmeans
from merge_svgs import merge_svgs
from ocr_text_to_svg import write_svg as write_text_svg
from trace_to_svg import TRACE_PRESETS, trace_image_to_svg


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH = SCRIPT_DIR / "data" / "graph" / "1.png"
DEFAULT_TEXT = SCRIPT_DIR / "data" / "text" / "1.png"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.1f}s"


def run_timed(label: str, func, *args, **kwargs):
    started = time.perf_counter()
    print(f"[time] {label} started")
    try:
        return func(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        print(f"[time] {label} finished in {format_duration(elapsed)}")


def default_flat_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_flat.png")


def default_trace_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_flat.svg")


def flatten_graph(input_path: Path, output_path: Path, args: argparse.Namespace) -> Path:
    img = np.array(Image.open(input_path).convert("RGB"))
    result, stats = flatten_lab_kmeans(
        img,
        bilateral_diameter=args.bilateral_diameter,
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        k=args.clusters,
        attempts=args.attempts,
        sample_limit=args.sample_limit,
        merge_delta=args.merge_delta,
        bg_delta=args.bg_delta,
        white_l_threshold=args.white_l_threshold,
        min_component_area=args.min_component_area,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path)
    print(f"Flattened graph with Lab K-means: {output_path}")
    print(f"Palette colors: {stats['unique_output_colors']}")
    return output_path


def run_pipeline(args: argparse.Namespace) -> Path:
    graph_input = args.graph.resolve()
    text_input = args.text.resolve()
    flat_png = args.flat_output.resolve() if args.flat_output else default_flat_path(graph_input).resolve()
    graph_svg = args.graph_svg.resolve() if args.graph_svg else default_trace_path(graph_input).resolve()
    text_svg = args.text_svg.resolve() if args.text_svg else text_input.with_suffix(".svg").resolve()
    output_svg = args.output.resolve()

    pipeline_started = time.perf_counter()
    run_timed("SVG step 1/4 flatten graph", flatten_graph, graph_input, flat_png, args)
    run_timed(
        "SVG step 2/4 trace graph",
        trace_image_to_svg,
        flat_png,
        graph_svg,
        trace_preset=args.trace,
        visible=args.visible,
        keep_document_open=args.keep_open,
    )
    run_timed(
        "SVG step 3/4 OCR text",
        write_text_svg,
        text_input,
        text_svg,
        args.min_ocr_score,
        include_background=True,
        print_items=getattr(args, "print_ocr_items", False),
    )
    run_timed("SVG step 4/4 merge SVGs", merge_svgs, graph_svg, text_svg, output_svg, args.background)
    print(f"[time] SVG pipeline total: {format_duration(time.perf_counter() - pipeline_started)}")
    return output_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a merged SVG from separate graph and text images.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Graph image path.")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT, help="Text image path.")
    parser.add_argument("-o", "--output", type=Path, default=SCRIPT_DIR / "data" / "merged_1.svg")
    parser.add_argument("--flat-output", type=Path, default=None, help="Intermediate flattened graph PNG.")
    parser.add_argument("--graph-svg", type=Path, default=None, help="Intermediate traced graph SVG.")
    parser.add_argument("--text-svg", type=Path, default=None, help="Intermediate OCR text SVG.")
    parser.add_argument("--background", default="#ffffff")
    parser.add_argument("-t", "--trace", choices=sorted(TRACE_PRESETS), default="high_quality")
    parser.add_argument("--visible", action="store_true", help="Show CorelDRAW while tracing.")
    parser.add_argument("--keep-open", action="store_true", help="Keep CorelDRAW document open after tracing.")
    parser.add_argument("--min-ocr-score", type=float, default=0.5)
    parser.add_argument("--print-ocr-items", action="store_true", help="Print every recognized OCR text item.")

    parser.add_argument("--bilateral-diameter", type=int, default=9)
    parser.add_argument("--sigma-color", type=float, default=55.0)
    parser.add_argument("--sigma-space", type=float, default=55.0)
    parser.add_argument("-k", "--clusters", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=250000)
    parser.add_argument("--merge-delta", type=float, default=8.5)
    parser.add_argument("--bg-delta", type=float, default=5.0)
    parser.add_argument("--white-l-threshold", type=float, default=94.0)
    parser.add_argument("--min-component-area", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_pipeline(args)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Pipeline done: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
