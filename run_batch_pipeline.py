from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from api_split_image_layers import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    DEFAULT_SIZE,
    GRAPH_LAYER_PROMPT,
    TEXT_LAYER_PROMPT,
    write_layer,
)
from graph_text_pipeline import run_pipeline as run_graph_text_pipeline
from trace_to_svg import TRACE_PRESETS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    return sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_prompt(path: Path | None, default_prompt: str) -> str:
    return path.read_text(encoding="utf-8") if path else default_prompt


def output_paths(source_image: Path, output_dir: Path) -> dict[str, Path]:
    root = output_dir / source_image.stem
    return {
        "root": root,
        "text_dir": root / "text",
        "graph_dir": root / "graph",
        "out_dir": root / "out",
        "text_png": root / "text" / "1.png",
        "graph_png": root / "graph" / "1.png",
        "text_svg": root / "text" / "1.svg",
        "graph_flat_png": root / "graph" / "1_flat.png",
        "graph_flat_svg": root / "graph" / "1_flat.svg",
        "out_svg": root / "out" / "out.svg",
    }


def ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ("text_dir", "graph_dir", "out_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def build_layer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        size=args.size,
        quality=args.quality,
        output_format="png",
        proxy=args.proxy,
        timeout=args.timeout,
    )


def split_with_gpt(source_image: Path, paths: dict[str, Path], args: argparse.Namespace) -> None:
    text_ready = paths["text_png"].is_file()
    graph_ready = paths["graph_png"].is_file()
    if text_ready and graph_ready and not args.force_api:
        print(f"Reusing GPT layer images: {paths['root']}")
        return

    if not args.api_key:
        raise RuntimeError("API key missing. Set IMAGE_MODEL_gpt2_API_KEY, OPENAI_API_KEY, APIYI_API_KEY, or pass --api-key.")

    layer_args = build_layer_args(args)
    text_prompt = read_prompt(args.text_prompt_file, TEXT_LAYER_PROMPT)
    graph_prompt = read_prompt(args.graph_prompt_file, GRAPH_LAYER_PROMPT)

    if args.only_layer in {"both", "text"} and (args.force_api or not text_ready):
        write_layer(
            name="text layer",
            image_path=source_image,
            output_path=paths["text_png"],
            prompt=text_prompt,
            args=layer_args,
        )

    if args.only_layer in {"both", "graph"} and (args.force_api or not graph_ready):
        write_layer(
            name="graphics layer",
            image_path=source_image,
            output_path=paths["graph_png"],
            prompt=graph_prompt,
            args=layer_args,
        )


def build_svg_args(paths: dict[str, Path], args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        graph=paths["graph_png"],
        text=paths["text_png"],
        output=paths["out_svg"],
        flat_output=paths["graph_flat_png"],
        graph_svg=paths["graph_flat_svg"],
        text_svg=paths["text_svg"],
        background=args.background,
        trace=args.trace,
        visible=args.visible,
        keep_open=args.keep_open,
        min_ocr_score=args.min_ocr_score,
        bilateral_diameter=args.bilateral_diameter,
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        clusters=args.clusters,
        attempts=args.attempts,
        sample_limit=args.sample_limit,
        merge_delta=args.merge_delta,
        bg_delta=args.bg_delta,
        white_l_threshold=args.white_l_threshold,
        min_component_area=args.min_component_area,
    )


def build_svg(paths: dict[str, Path], args: argparse.Namespace) -> Path:
    if not paths["text_png"].is_file():
        raise FileNotFoundError(f"Missing text layer image: {paths['text_png']}")
    if not paths["graph_png"].is_file():
        raise FileNotFoundError(f"Missing graph layer image: {paths['graph_png']}")
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    return run_graph_text_pipeline(build_svg_args(paths, args))


def process_image(source_image: Path, args: argparse.Namespace) -> Path | None:
    paths = output_paths(source_image, args.output_dir.resolve())
    ensure_dirs(paths)

    print(f"\n=== {source_image.name} ===")
    print(f"Project folder: {paths['root']}")

    if args.stage in {"all", "split"}:
        split_with_gpt(source_image, paths, args)

    if args.stage in {"all", "svg"}:
        return build_svg(paths, args)

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: input image -> GPT text/graph layer images -> merged SVG."
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Process one input image.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory of input images.")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory.")
    parser.add_argument("--stage", choices=["all", "split", "svg"], default="all")
    parser.add_argument("--force-api", action="store_true", help="Regenerate text/graph layer PNGs even if they already exist.")
    parser.add_argument("--only-layer", choices=["both", "text", "graph"], default="both")

    parser.add_argument(
        "--base-url",
        default=(os.environ.get("IMAGE_MODEL_gpt2_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=(os.environ.get("IMAGE_MODEL_gpt2_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("APIYI_API_KEY")),
    )
    parser.add_argument("--model", default=os.environ.get("IMAGE_MODEL_gpt2_MODEL") or os.environ.get("IMAGE_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--text-prompt-file", type=Path, default=None)
    parser.add_argument("--graph-prompt-file", type=Path, default=None)

    parser.add_argument("--background", default="#ffffff")
    parser.add_argument("-t", "--trace", choices=sorted(TRACE_PRESETS), default="high_quality")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--min-ocr-score", type=float, default=0.5)

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
    args.output_dir = args.output_dir.resolve()

    try:
        images = [args.input.resolve()] if args.input else collect_images(args.input_dir.resolve())
        if not images:
            raise FileNotFoundError(f"No input images found in: {args.input_dir}")

        succeeded: list[Path] = []
        failed: list[tuple[Path, str]] = []
        for image in images:
            try:
                result = process_image(image, args)
                if result:
                    succeeded.append(result)
            except Exception as exc:
                failed.append((image, str(exc)))
                print(f"Failed {image.name}: {exc}", file=sys.stderr)

        print(f"\nBatch done: {len(images) - len(failed)} succeeded, {len(failed)} failed")
        if succeeded:
            print("SVG outputs:")
            for path in succeeded:
                print(f"  {path}")
        if failed:
            print("Failures:", file=sys.stderr)
            for image, reason in failed:
                print(f"  {image.name}: {reason}", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
