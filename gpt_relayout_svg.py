from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import traceback
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PNG = SCRIPT_DIR / "input" / "image13.png"
DEFAULT_SVG = SCRIPT_DIR / "output" / "image13" / "out" / "out.svg"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "image13" / "out"
DEFAULT_OUTPUT_NAME = "gpt_layout.svg"
DEFAULT_RAW_RESPONSE_NAME = "gpt_layout_response.txt"
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "xhigh"
REASONING_EFFORTS = ("default", "none", "minimal", "low", "medium", "high", "xhigh")

USER_PROMPT = """The first input is a PNG reference image that shows the desired final result.
The second input is an SVG draft that was generated from that PNG, but its visual layout is not accurate enough.

Please improve the SVG draft so that the final SVG matches the PNG reference as closely as possible.
Use the existing SVG as the editable source file: reposition, scale, align, group, reorder, and adjust its components to match the PNG.
Pay close attention to the overall composition, canvas size, spacing, object positions, proportions, colors, text placement, and visual hierarchy.

Do not create a new unrelated design. Do not embed the PNG as an image inside the SVG.
Return a complete optimized SVG file whose layout and appearance are consistent with the PNG reference."""

INSTRUCTIONS = """You are a precise SVG layout editor.
Return only the complete SVG code. Do not include explanations, Markdown fences, comments outside the SVG, or any extra text.
Use the provided SVG as the source to edit, and prefer preserving its existing components and text content.
Adjust positions, transforms, grouping, scaling, ordering, colors, and spacing only as needed to match the PNG reference.
Do not embed the PNG as a base64 image. Do not return a screenshot. Do not omit important SVG components."""


def load_env_file(path: Path = SCRIPT_DIR / ".env") -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    lowered = url.lower()
    for suffix in ("/images/edits", "/responses"):
        if lowered.endswith(suffix):
            return url[: -len(suffix)]
    return url


def proxy_env_summary() -> str:
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    values = [f"{name}={os.environ[name]}" for name in names if os.environ.get(name)]
    return ", ".join(values) if values else "not set"


def image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    with Image.open(path) as image:
        return image.size


def extract_svg_text(response_text: str) -> str:
    fenced = re.search(
        r"```(?:svg|xml)?\s*(<\?xml[\s\S]*?</svg>|<svg[\s\S]*?</svg>)\s*```",
        response_text,
        flags=re.IGNORECASE,
    )
    if fenced:
        return fenced.group(1).strip()

    lowered = response_text.lower()
    svg_start = lowered.find("<svg")
    svg_end = lowered.rfind("</svg>")
    if svg_start < 0 or svg_end < 0:
        raise ValueError("API response did not contain a complete <svg>...</svg> document.")

    xml_start = lowered.rfind("<?xml", 0, svg_start)
    start = xml_start if xml_start >= 0 else svg_start
    return response_text[start : svg_end + len("</svg>")].strip()


def validate_svg(svg_text: str) -> None:
    try:
        root = ET.fromstring(svg_text.encode("utf-8"))
    except ET.ParseError as exc:
        raise ValueError(f"Returned SVG is not valid XML: {exc}") from exc

    tag = root.tag.rsplit("}", 1)[-1].lower()
    if tag != "svg":
        raise ValueError(f"Returned XML root is <{tag}>, not <svg>.")


def build_input_content(png_path: Path, svg_path: Path, prompt: str) -> list[dict[str, str]]:
    svg_text = svg_path.read_text(encoding="utf-8")
    dimensions = png_dimensions(png_path)
    size_hint = ""
    if dimensions:
        width, height = dimensions
        size_hint = (
            f"\nPNG 最终效果图尺寸是 {width}x{height}。"
            f"输出 SVG 必须使用 width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\"。"
        )

    return [
        {
            "type": "input_image",
            "image_url": image_to_data_url(png_path),
        },
        {
            "type": "input_text",
            "text": prompt
            + "\n\n第一张 PNG 图片已作为 image13.png 提供。"
            + size_hint
            + "\n必须尽量保留第二张 SVG 里的原组件和原文字内容，不要把文字改成乱码。"
            + "\n第二张 SVG 文件内容如下，请直接修改这份 SVG：",
        },
        {
            "type": "input_text",
            "text": f"文件名: {svg_path.name}\n\n```svg\n{svg_text}\n```",
        },
    ]


def response_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "to_dict"):
        data = response.to_dict()
    else:
        raise ValueError("Could not read text from API response object.")

    texts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("API response did not include text output.")
    return "\n".join(texts)


def call_responses_api(args: argparse.Namespace, content: list[dict[str, str]]) -> str:
    try:
        from openai import APIConnectionError, APITimeoutError, OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: openai. Install it with `venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from exc

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    create_args: dict[str, object] = {
        "model": args.model,
        "instructions": args.instructions,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }
    if args.reasoning_effort != "default":
        create_args["reasoning"] = {"effort": args.reasoning_effort}
    if args.max_output_tokens:
        create_args["max_output_tokens"] = args.max_output_tokens

    for attempt in range(args.retries + 1):
        try:
            response = client.responses.create(**create_args)
            return response_output_text(response)
        except (APIConnectionError, APITimeoutError) as exc:
            if attempt >= args.retries:
                raise
            wait_seconds = args.retry_delay * (attempt + 1)
            print(
                f"Connection failed ({exc.__class__.__name__}). "
                f"Retrying in {wait_seconds:.1f}s... ({attempt + 1}/{args.retries})",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Responses API call failed without returning a response.")


def relayout_svg(args: argparse.Namespace) -> Path:
    png_path = args.png.resolve()
    svg_path = args.svg.resolve()
    args.base_url = normalize_base_url(args.base_url)
    output_dir = args.output_dir.resolve()
    output_path = output_dir / args.output_name
    raw_response_path = output_dir / args.raw_response_name

    if not png_path.is_file():
        raise FileNotFoundError(f"PNG not found: {png_path}")
    if not svg_path.is_file():
        raise FileNotFoundError(f"SVG not found: {svg_path}")
    if not args.api_key and not getattr(args, "dry_run", False):
        raise RuntimeError("API key missing. Set GPT_SVG_API_KEY, OPENAI_API_KEY, APIYI_API_KEY, or IMAGE_MODEL_gpt2_API_KEY.")

    content = build_input_content(png_path, svg_path, args.prompt)
    svg_size = svg_path.stat().st_size
    png_size = png_path.stat().st_size

    print(f"PNG: {png_path} ({png_size} bytes)")
    print(f"SVG: {svg_path} ({svg_size} bytes)")
    print(f"Model: {args.model}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Base URL: {args.base_url}")
    print(f"Proxy env: {proxy_env_summary()}")
    print(f"Output SVG: {output_path}")

    if getattr(args, "dry_run", False):
        print("Dry run OK. No API request was sent.")
        return output_path

    print("Calling Responses API...")
    response_text = call_responses_api(args, content)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path.write_text(response_text, encoding="utf-8")

    returned_svg = extract_svg_text(response_text)
    validate_svg(returned_svg)
    output_path.write_text(returned_svg + "\n", encoding="utf-8")

    print(f"Saved raw response: {raw_response_path}")
    print(f"Saved SVG: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Send image13.png and out.svg to a Responses-compatible model, then save the returned SVG."
    )
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG, help="Final-effect PNG path.")
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG, help="SVG file that the model should edit.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for returned files.")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="Returned SVG filename.")
    parser.add_argument("--raw-response-name", default=DEFAULT_RAW_RESPONSE_NAME, help="Raw model text filename.")
    parser.add_argument("--prompt", default=USER_PROMPT, help="User prompt sent with the PNG and SVG.")
    parser.add_argument("--instructions", default=INSTRUCTIONS, help="System/developer-style instructions.")
    parser.add_argument(
        "--api-key",
        default=first_env("GPT_SVG_API_KEY", "OPENAI_API_KEY", "APIYI_API_KEY", "IMAGE_MODEL_gpt2_API_KEY"),
    )
    default_base_url = normalize_base_url(
        first_env("GPT_SVG_BASE_URL", "OPENAI_BASE_URL", "APIYI_BASE_URL", "IMAGE_MODEL_gpt2_BASE_URL")
        or DEFAULT_BASE_URL
    )

    parser.add_argument(
        "--base-url",
        default=default_base_url,
    )
    parser.add_argument("--model", default=first_env("GPT_SVG_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=first_env("GPT_SVG_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT,
        help="Reasoning effort for GPT SVG relayout. Use xhigh for maximum quality.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=3, help="Retry count for connection errors.")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Base seconds between retries.")
    parser.add_argument("--debug", action="store_true", help="Print detailed exception information on failure.")
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without calling the API.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        relayout_svg(args)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            print("\nDebug details:", file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            cause = getattr(exc, "__cause__", None)
            if cause:
                print("\nCause:", file=sys.stderr)
                traceback.print_exception(type(cause), cause, cause.__traceback__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
