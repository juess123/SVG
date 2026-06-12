from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "auto"
DEFAULT_QUALITY = "high"
DEFAULT_OUTPUT_FORMAT = "png"


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


load_env_file()


TEXT_LAYER_PROMPT = """You are separating a design image into clean editable-looking layer images.

Task: create a TEXT-ONLY layer image from the input.

Rules:
1. Extract every visible text element from the original image and place it in the exact same position, scale, color, rotation, and visual style as the original.
2. Remove all non-text graphics, icons, shapes, decorations, backgrounds, lines, and image components.
3. Preserve the original canvas size and composition. Do not crop, resize, rotate, or re-layout.
4. The output must contain only text components. No extra components, no invented marks, no background graphics.
5. If transparent output is supported, use transparent background; otherwise use a plain white background.
6. Carefully check for extra generated artifacts and remove them.

Output: one image containing only the text layer."""


GRAPH_LAYER_PROMPT = """Create a graphics-only exploded component asset sheet from the input image.

The final image must visually look like a separated design asset sheet, not a completed infographic and not the original overlapping composition.

Main task:
Remove all text completely. Extract every non-text graphical component, complete any hidden or cut-off parts, and lay all components out separately on the same canvas.

Component rules:
1. Identify every non-text graphical component, including large hexagons, small hexagons, outlines, fills, gradients, shadows, dashed arrows, solid arrows, vertical decorative lines, bottom long bars, small rounded bars, chevrons, wave/arc decorations, and any other non-text design element.
2. Each repeated item is its own independent component. For example, each large hexagon, each small hexagon badge, each arrow, each chevron group, and each bar should appear as a separate complete object.
3. If a component is partially hidden behind another component in the source image, reconstruct the missing hidden part before placing it separately.
4. If two components overlap in the source image, separate them into different positions. Do not leave any stacked, touching, intersecting, or overlapping components in the final result.

Layout rules:
5. Do not preserve the original chain layout. Do not keep the original connected/overlapping arrangement.
6. Arrange the components in a clean exploded layout with generous white space between components, similar to a design asset sheet or component inventory board.
7. Keep the original canvas size and use a clean white background.
8. Preserve each component's original visual style as much as possible: size, proportions, colors, outlines, stroke thickness, gradients, shadows, transparency, and antialiasing. Do not stretch components.

Quality rules:
9. Do not add labels, numbers, boxes, separators, guide lines, captions, text, or extra decorations.
10. Do not invent new components. Do not duplicate components beyond the repeated instances already present in the source.
11. Before finalizing, inspect the image for artifacts, stray fragments, accidental text remnants, duplicate pieces, missing reconstructed parts, and any remaining overlaps. Remove or fix them.

Output:
One graphics-only image where all non-text components are fully reconstructed, separated, and laid out as independent non-overlapping assets."""


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def build_multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----svg-layer-split-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    for name, path in files.items():
        data = path.read_bytes()
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
                f"Content-Type: {guess_mime(path)}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def decode_image_response(payload: dict) -> bytes:
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            b64_json = first.get("b64_json") or first.get("image_base64")
            if b64_json:
                return base64.b64decode(b64_json)
            url = first.get("url")
            if url:
                with urllib.request.urlopen(url, timeout=120) as response:
                    return response.read()

    # Some compatible providers return a direct top-level base64 field.
    for key in ("b64_json", "image_base64", "base64"):
        if payload.get(key):
            return base64.b64decode(payload[key])

    raise RuntimeError(f"Could not find image data in API response: {json.dumps(payload, ensure_ascii=False)[:1000]}")


def resolve_edits_endpoint(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/images/edits"):
        return url
    return url + "/images/edits"


def call_image_edit(
    *,
    base_url: str,
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    size: str,
    quality: str,
    output_format: str,
    proxy: str | None,
    timeout: int,
) -> bytes:
    endpoint = resolve_edits_endpoint(base_url)
    fields = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": "1",
        "output_format": output_format,
    }

    body, content_type = build_multipart(fields, {"image": image_path})
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
            "Accept": "application/json",
        },
    )

    opener = urllib.request.build_opener()
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )

    try:
        with opener.open(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {exc.code}: {error_body}") from exc

    payload = json.loads(response_body)
    return decode_image_response(payload)


def write_layer(
    *,
    name: str,
    image_path: Path,
    output_path: Path,
    prompt: str,
    args: argparse.Namespace,
) -> Path:
    print(f"Generating {name}: {output_path}")
    started = time.time()
    image_bytes = call_image_edit(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        image_path=image_path,
        prompt=prompt,
        size=args.size,
        quality=args.quality,
        output_format=args.output_format,
        proxy=args.proxy,
        timeout=args.timeout,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    print(f"Saved {name}: {output_path} ({time.time() - started:.1f}s)")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an OpenAI-compatible image edit API to split one image into text-only and graphics-only images."
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input image path.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--text-output", type=Path, default=None, help="Exact output path for the text-only image.")
    parser.add_argument("--graph-output", type=Path, default=None, help="Exact output path for the graphics-only image.")
    parser.add_argument(
        "--base-url",
        default=(
            os.environ.get("IMAGE_MODEL_gpt2_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("IMAGE_MODEL_gpt2_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("APIYI_API_KEY")
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("IMAGE_MODEL_gpt2_MODEL") or os.environ.get("IMAGE_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT, choices=["png", "webp", "jpeg"])
    parser.add_argument("--proxy", default=None, help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7890.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--text-prompt-file", type=Path, default=None)
    parser.add_argument("--graph-prompt-file", type=Path, default=None)
    parser.add_argument("--only", choices=["both", "text", "graph"], default="both")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: API key missing. Set OPENAI_API_KEY or APIYI_API_KEY, or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    image_path = args.input.resolve()
    if not image_path.is_file():
        print(f"Error: input image not found: {image_path}", file=sys.stderr)
        return 1

    output_dir = (args.output_dir or image_path.with_name(f"{image_path.stem}_layers")).resolve()
    suffix = "." + args.output_format
    text_output = args.text_output.resolve() if args.text_output else output_dir / f"{image_path.stem}_text_only{suffix}"
    graph_output = args.graph_output.resolve() if args.graph_output else output_dir / f"{image_path.stem}_graphics_only{suffix}"

    text_prompt = args.text_prompt_file.read_text(encoding="utf-8") if args.text_prompt_file else TEXT_LAYER_PROMPT
    graph_prompt = args.graph_prompt_file.read_text(encoding="utf-8") if args.graph_prompt_file else GRAPH_LAYER_PROMPT

    try:
        if args.only in {"both", "text"}:
            write_layer(
                name="text layer",
                image_path=image_path,
                output_path=text_output,
                prompt=text_prompt,
                args=args,
            )
        if args.only in {"both", "graph"}:
            write_layer(
                name="graphics layer",
                image_path=image_path,
                output_path=graph_output,
                prompt=graph_prompt,
                args=args,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
