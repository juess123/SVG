from __future__ import annotations

import argparse
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
PATH_TOKEN_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def qname(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        values = [float(v) for v in re.split(r"[\s,]+", viewbox.strip()) if v]
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]
    width = float(re.sub(r"[^0-9.].*$", "", root.attrib.get("width", "0")) or 0)
    height = float(re.sub(r"[^0-9.].*$", "", root.attrib.get("height", "0")) or 0)
    return 0.0, 0.0, width, height


def numbers(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(match.group(0)) for match in NUMBER_RE.finditer(value)]


def add_point(
    bounds: tuple[float, float, float, float] | None,
    x: float,
    y: float,
) -> tuple[float, float, float, float]:
    point = (x, y, x, y)
    return merge_bounds(bounds, point)


def path_bounds(path_data: str | None) -> tuple[float, float, float, float] | None:
    if not path_data:
        return None

    tokens = PATH_TOKEN_RE.findall(path_data)
    index = 0
    command = ""
    x = y = 0.0
    subpath_x = subpath_y = 0.0
    bounds: tuple[float, float, float, float] | None = None

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    def has_number() -> bool:
        return index < len(tokens) and not is_command(tokens[index])

    def read_number() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if is_command(tokens[index]):
            command = tokens[index]
            index += 1
        if not command:
            break

        relative = command.islower()
        cmd = command.upper()

        if cmd == "Z":
            x, y = subpath_x, subpath_y
            bounds = add_point(bounds, x, y)
            command = ""
            continue

        if cmd == "M":
            first = True
            while has_number() and index + 1 < len(tokens):
                nx = read_number()
                ny = read_number()
                if relative:
                    nx += x
                    ny += y
                x, y = nx, ny
                bounds = add_point(bounds, x, y)
                if first:
                    subpath_x, subpath_y = x, y
                    first = False
            command = "l" if relative else "L"
            continue

        if cmd == "L":
            while has_number() and index + 1 < len(tokens):
                nx = read_number()
                ny = read_number()
                if relative:
                    nx += x
                    ny += y
                x, y = nx, ny
                bounds = add_point(bounds, x, y)
            continue

        if cmd == "H":
            while has_number():
                nx = read_number()
                if relative:
                    nx += x
                x = nx
                bounds = add_point(bounds, x, y)
            continue

        if cmd == "V":
            while has_number():
                ny = read_number()
                if relative:
                    ny += y
                y = ny
                bounds = add_point(bounds, x, y)
            continue

        if cmd == "C":
            while has_number() and index + 5 < len(tokens):
                points = [(read_number(), read_number()), (read_number(), read_number()), (read_number(), read_number())]
                abs_points = []
                for px, py in points:
                    abs_points.append((px + x, py + y) if relative else (px, py))
                for px, py in abs_points:
                    bounds = add_point(bounds, px, py)
                x, y = abs_points[-1]
            continue

        if cmd == "S" or cmd == "Q":
            while has_number() and index + 3 < len(tokens):
                points = [(read_number(), read_number()), (read_number(), read_number())]
                abs_points = []
                for px, py in points:
                    abs_points.append((px + x, py + y) if relative else (px, py))
                for px, py in abs_points:
                    bounds = add_point(bounds, px, py)
                x, y = abs_points[-1]
            continue

        if cmd == "T":
            while has_number() and index + 1 < len(tokens):
                nx = read_number()
                ny = read_number()
                if relative:
                    nx += x
                    ny += y
                x, y = nx, ny
                bounds = add_point(bounds, x, y)
            continue

        if cmd == "A":
            while has_number() and index + 6 < len(tokens):
                rx = read_number()
                ry = read_number()
                _rotation = read_number()
                _large_arc = read_number()
                _sweep = read_number()
                nx = read_number()
                ny = read_number()
                if relative:
                    nx += x
                    ny += y
                bounds = merge_bounds(bounds, (min(x, nx) - rx, min(y, ny) - ry, max(x, nx) + rx, max(y, ny) + ry))
                x, y = nx, ny
            continue

        break

    return bounds


def merge_bounds(
    bounds: tuple[float, float, float, float] | None,
    item: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if item is None:
        return bounds
    if bounds is None:
        return item
    return min(bounds[0], item[0]), min(bounds[1], item[1]), max(bounds[2], item[2]), max(bounds[3], item[3])


def element_bounds(element: ET.Element) -> tuple[float, float, float, float] | None:
    tag = element.tag.split("}", 1)[-1]
    bounds: tuple[float, float, float, float] | None = None

    if tag == "path":
        bounds = path_bounds(element.attrib.get("d"))
    elif tag in {"polygon", "polyline"}:
        vals = numbers(element.attrib.get("points"))
        if len(vals) >= 2:
            xs = vals[0::2]
            ys = vals[1::2]
            bounds = (min(xs), min(ys), max(xs), max(ys))
    elif tag in {"rect", "image"}:
        x = float(element.attrib.get("x", "0") or 0)
        y = float(element.attrib.get("y", "0") or 0)
        width = float(element.attrib.get("width", "0") or 0)
        height = float(element.attrib.get("height", "0") or 0)
        if width and height:
            bounds = (x, y, x + width, y + height)
    elif tag in {"circle", "ellipse"}:
        cx = float(element.attrib.get("cx", "0") or 0)
        cy = float(element.attrib.get("cy", "0") or 0)
        rx = float(element.attrib.get("r", element.attrib.get("rx", "0")) or 0)
        ry = float(element.attrib.get("r", element.attrib.get("ry", "0")) or 0)
        if rx and ry:
            bounds = (cx - rx, cy - ry, cx + rx, cy + ry)

    for child in element:
        bounds = merge_bounds(bounds, element_bounds(child))
    return bounds


def is_defs_or_metadata(element: ET.Element) -> bool:
    tag = element.tag.split("}", 1)[-1]
    return tag in {"defs", "metadata", "title", "desc"}


def is_background_rect(element: ET.Element) -> bool:
    tag = element.tag.split("}", 1)[-1]
    width = element.attrib.get("width")
    height = element.attrib.get("height")
    return tag == "rect" and width == "100%" and height == "100%"


def merge_svgs(graph_svg: Path, text_svg: Path, output_svg: Path, background: str) -> Path:
    graph_root = ET.parse(graph_svg).getroot()
    text_root = ET.parse(text_svg).getroot()
    _, _, target_width, target_height = parse_viewbox(text_root)

    graph_bounds = None
    for child in graph_root:
        if not is_defs_or_metadata(child):
            graph_bounds = merge_bounds(graph_bounds, element_bounds(child))
    if graph_bounds is None:
        raise ValueError(f"No drawable graph content found in {graph_svg}")

    min_x, min_y, max_x, max_y = graph_bounds
    content_width = max_x - min_x
    content_height = max_y - min_y
    sx = target_width / content_width
    sy = target_height / content_height
    scale = min(sx, sy)
    tx = -min_x * scale + (target_width - content_width * scale) / 2
    ty = -min_y * scale + (target_height - content_height * scale) / 2

    root = ET.Element(
        qname("svg"),
        {
            "width": f"{target_width:g}",
            "height": f"{target_height:g}",
            "viewBox": f"0 0 {target_width:g} {target_height:g}",
        },
    )
    ET.SubElement(root, qname("rect"), {"width": "100%", "height": "100%", "fill": background})

    for child in graph_root:
        if child.tag.split("}", 1)[-1] == "defs":
            root.append(copy.deepcopy(child))

    graph_group = ET.SubElement(
        root,
        qname("g"),
        {
            "id": "graph",
            "transform": f"matrix({scale:.8f} 0 0 {scale:.8f} {tx:.8f} {ty:.8f})",
        },
    )
    for child in graph_root:
        if not is_defs_or_metadata(child):
            graph_group.append(copy.deepcopy(child))

    text_group = ET.SubElement(root, qname("g"), {"id": "text"})
    for child in text_root:
        if not is_defs_or_metadata(child) and not is_background_rect(child):
            text_group.append(copy.deepcopy(child))

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_svg, encoding="utf-8", xml_declaration=True)

    print(f"Graph bounds: {graph_bounds}")
    print(f"Target viewBox: 0 0 {target_width:g} {target_height:g}")
    print(f"Raw scale: sx={sx:.8f}, sy={sy:.8f}")
    print(f"Uniform scale: {scale:.8f}")
    print(f"Done: {output_svg}")
    return output_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a traced graph SVG and an OCR text SVG into one SVG.")
    parser.add_argument("--graph", type=Path, required=True, help="Traced graph SVG.")
    parser.add_argument("--text", type=Path, required=True, help="OCR text SVG.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Merged SVG output path.")
    parser.add_argument("--background", default="#ffffff", help="Output background color.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        merge_svgs(args.graph.resolve(), args.text.resolve(), args.output.resolve(), args.background)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
