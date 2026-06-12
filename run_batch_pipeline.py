from __future__ import annotations

import argparse
import os
import sys
import time
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
from gpt_relayout_svg import (
    DEFAULT_BASE_URL as DEFAULT_RELAYOUT_BASE_URL,
    DEFAULT_MODEL as DEFAULT_RELAYOUT_MODEL,
    DEFAULT_OUTPUT_NAME as DEFAULT_RELAYOUT_OUTPUT_NAME,
    DEFAULT_RAW_RESPONSE_NAME as DEFAULT_RELAYOUT_RAW_RESPONSE_NAME,
    DEFAULT_REASONING_EFFORT as DEFAULT_RELAYOUT_REASONING_EFFORT,
    INSTRUCTIONS as RELAYOUT_INSTRUCTIONS,
    REASONING_EFFORTS,
    USER_PROMPT as RELAYOUT_PROMPT,
    first_env,
    normalize_base_url,
    relayout_svg,
)
from trace_to_svg import TRACE_PRESETS


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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
        print_ocr_items=args.print_ocr_items,
    )


def build_svg(paths: dict[str, Path], args: argparse.Namespace) -> Path:
    if not paths["text_png"].is_file():
        raise FileNotFoundError(f"Missing text layer image: {paths['text_png']}")
    if not paths["graph_png"].is_file():
        raise FileNotFoundError(f"Missing graph layer image: {paths['graph_png']}")
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    return run_graph_text_pipeline(build_svg_args(paths, args))


def build_relayout_args(source_image: Path, paths: dict[str, Path], args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        png=source_image,
        svg=paths["out_svg"],
        output_dir=paths["out_dir"],
        output_name=args.relayout_output_name,
        raw_response_name=args.relayout_raw_response_name,
        prompt=read_prompt(args.relayout_prompt_file, RELAYOUT_PROMPT),
        instructions=read_prompt(args.relayout_instructions_file, RELAYOUT_INSTRUCTIONS),
        api_key=args.relayout_api_key,
        base_url=normalize_base_url(args.relayout_base_url),
        model=args.relayout_model,
        reasoning_effort=args.relayout_reasoning_effort,
        max_output_tokens=args.relayout_max_output_tokens,
        timeout=args.relayout_timeout,
        retries=args.relayout_retries,
        retry_delay=args.relayout_retry_delay,
        debug=args.debug_relayout,
        dry_run=args.relayout_dry_run,
    )


def relayout_with_gpt(source_image: Path, paths: dict[str, Path], args: argparse.Namespace) -> Path:
    if not paths["out_svg"].is_file():
        raise FileNotFoundError(f"Missing SVG for relayout: {paths['out_svg']}")

    output_path = paths["out_dir"] / args.relayout_output_name
    if output_path.is_file() and not args.force_relayout and not args.relayout_dry_run:
        print(f"Reusing GPT relayout SVG: {output_path}")
        return output_path

    if not args.relayout_api_key and not args.relayout_dry_run:
        raise RuntimeError("Relayout API key missing. Set GPT_SVG_API_KEY, OPENAI_API_KEY, APIYI_API_KEY, or pass --relayout-api-key.")

    print("Running GPT SVG relayout...")
    return relayout_svg(build_relayout_args(source_image, paths, args))


def process_image(source_image: Path, args: argparse.Namespace) -> Path | None:
    image_started = time.perf_counter()
    paths = output_paths(source_image, args.output_dir.resolve())
    ensure_dirs(paths)

    print(f"\n=== {source_image.name} ===")
    print(f"Project folder: {paths['root']}")

    if args.stage in {"all", "split"}:
        run_timed(f"{source_image.name} step 1 split layers", split_with_gpt, source_image, paths, args)

    result: Path | None = None
    if args.stage in {"all", "svg"}:
        result = run_timed(f"{source_image.name} step 2 build SVG", build_svg, paths, args)

    if args.stage == "relayout" or (args.stage == "all" and not args.skip_relayout):
        result = run_timed(f"{source_image.name} step 3 GPT relayout", relayout_with_gpt, source_image, paths, args)

    print(f"[time] {source_image.name} total: {format_duration(time.perf_counter() - image_started)}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch pipeline: input image -> GPT text/graph layer images -> merged SVG -> GPT relayout SVG."
    )
    parser.add_argument("-i", "--input", type=Path, default=None, help="Process one input image.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory of input images.")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root output directory.")
    parser.add_argument("--stage", choices=["all", "split", "svg", "relayout"], default="all")
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

    default_relayout_base_url = normalize_base_url(
        first_env("GPT_SVG_BASE_URL", "OPENAI_BASE_URL", "APIYI_BASE_URL", "IMAGE_MODEL_gpt2_BASE_URL")
        or DEFAULT_RELAYOUT_BASE_URL
    )
    parser.add_argument("--skip-relayout", action="store_true", help="Do not run the final GPT SVG relayout step when --stage all is used.")
    parser.add_argument("--force-relayout", action="store_true", help="Regenerate layout.svg even if it already exists.")
    parser.add_argument("--relayout-output-name", default=DEFAULT_RELAYOUT_OUTPUT_NAME)
    parser.add_argument("--relayout-raw-response-name", default=DEFAULT_RELAYOUT_RAW_RESPONSE_NAME)
    parser.add_argument("--relayout-prompt-file", type=Path, default=None)
    parser.add_argument("--relayout-instructions-file", type=Path, default=None)
    parser.add_argument(
        "--relayout-api-key",
        default=first_env("GPT_SVG_API_KEY", "OPENAI_API_KEY", "APIYI_API_KEY", "IMAGE_MODEL_gpt2_API_KEY"),
    )
    parser.add_argument("--relayout-base-url", default=default_relayout_base_url)
    parser.add_argument("--relayout-model", default=first_env("GPT_SVG_MODEL", "OPENAI_MODEL") or DEFAULT_RELAYOUT_MODEL)
    parser.add_argument(
        "--relayout-reasoning-effort",
        choices=REASONING_EFFORTS,
        default=first_env("GPT_SVG_REASONING_EFFORT") or DEFAULT_RELAYOUT_REASONING_EFFORT,
        help="Reasoning effort for final GPT SVG relayout. Default is xhigh.",
    )
    parser.add_argument("--relayout-max-output-tokens", type=int, default=32768)
    parser.add_argument("--relayout-timeout", type=float, default=600)
    parser.add_argument("--relayout-retries", type=int, default=5)
    parser.add_argument("--relayout-retry-delay", type=float, default=5.0)
    parser.add_argument("--relayout-dry-run", action="store_true", help="Validate relayout inputs without calling the Responses API.")
    parser.add_argument("--debug", "--debug-relayout", dest="debug_relayout", action="store_true", help="Print detailed relayout API errors.")
    return parser


def main() -> int:
    program_started = time.perf_counter()
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
            print(f"[time] Program total: {format_duration(time.perf_counter() - program_started)}")
            return 1
        print(f"[time] Program total: {format_duration(time.perf_counter() - program_started)}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(f"[time] Program total: {format_duration(time.perf_counter() - program_started)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
