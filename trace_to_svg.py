"""
使用 CorelDRAW PowerTRACE（轮廓描摹）将位图转换为 SVG 矢量图。

依赖：本机已安装 CorelDRAW，且 COM 组件可用（CorelDRAW.Application）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import win32com.client

# CorelDRAW COM 常量（cdrFilter / cdrTraceType / cdrShapeType）
CDR_JPEG = 774
CDR_SVG = 1345
CDR_AUTOSENSE = 0
CDR_BITMAP_SHAPE = 5
CDR_CURRENT_PAGE = 1

# 轮廓描摹（Outline Trace）预设 —— 对应菜单「位图 → 轮廓描摹」
TRACE_PRESETS = {
    "line_art": 1,           # cdrTraceLineArt — 线稿/黑白插画
    "logo": 2,               # cdrTraceLogo
    "detailed_logo": 3,      # cdrTraceDetailedLogo
    "clipart": 4,            # cdrTraceClipart
    "low_quality": 5,        # cdrTraceLowQualityImage
    "high_quality": 6,       # cdrTraceHighQualityImage — 照片推荐
    "technical": 7,          # cdrTraceTechnical — 中心线描摹
    "line_drawing": 8,       # cdrTraceLineDrawing — 中心线描摹
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_INPUT = DATA_DIR / "image2.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def connect_coreldraw(visible: bool = False):
    """连接或启动 CorelDRAW COM 实例。"""
    try:
        app = win32com.client.Dispatch("CorelDRAW.Application")
    except Exception as exc:
        raise RuntimeError(
            "无法连接 CorelDRAW。请确认已安装 CorelDRAW 且 COM 组件已注册。"
        ) from exc

    app.Visible = visible
    app.Optimization = True
    return app


def import_bitmap(layer, image_path: Path):
    """导入 JPG/PNG 等到当前图层，返回导入的位图 Shape。"""
    # ImportEx 的第三个参数必须显式传 None，否则 pywin32 会报 COM 类型错误
    imp = layer.ImportEx(str(image_path.resolve()), CDR_AUTOSENSE, None)
    if imp.HasDialog:
        imp.Reset()
    imp.Finish()

    shape = layer.Shapes(layer.Shapes.Count)
    if shape.Type != CDR_BITMAP_SHAPE:
        raise RuntimeError(f"导入失败：最后一个对象不是位图（Type={shape.Type}）。")
    return shape


def outline_trace(bitmap_shape, trace_type: int, *, delete_original: bool = True):
    """执行 PowerTRACE 轮廓描摹，返回 TraceSettings（已完成 Finish）。"""
    trace_settings = bitmap_shape.Bitmap.Trace(trace_type)
    trace_settings.Finish()
    if delete_original:
        try:
            bitmap_shape.Delete()
        except Exception:
            pass
    return trace_settings


def export_svg(document, svg_path: Path):
    """导出当前文档为 SVG。"""
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    export_filter = document.ExportEx(
        str(svg_path.resolve()),
        CDR_SVG,
        CDR_CURRENT_PAGE,
        None,
        None,
    )
    export_filter.Finish()


def resolve_trace_type(trace_preset: str) -> int:
    if trace_preset not in TRACE_PRESETS:
        valid = ", ".join(sorted(TRACE_PRESETS))
        raise ValueError(f"未知描摹预设 '{trace_preset}'，可选：{valid}")
    return TRACE_PRESETS[trace_preset]


def collect_images(directory: Path) -> list[Path]:
    """收集目录下的所有图片文件（不含子目录）。"""
    return sorted(
        p.resolve()
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def trace_one_in_document(
    layer,
    document,
    input_path: Path,
    output_path: Path,
    trace_type: int,
) -> Path:
    print(f"导入位图：{input_path.name}")
    bitmap_shape = import_bitmap(layer, input_path)

    print(f"轮廓描摹（type={trace_type}）…")
    outline_trace(bitmap_shape, trace_type)
    time.sleep(0.5)

    print(f"导出 SVG：{output_path.name}")
    export_svg(document, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"SVG 导出失败：{output_path.name}")
    return output_path


def trace_image_to_svg(
    input_path: Path,
    output_path: Path,
    *,
    trace_preset: str = "high_quality",
    visible: bool = False,
    keep_document_open: bool = False,
    app=None,
) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入图片：{input_path}")

    trace_type = resolve_trace_type(trace_preset)
    owns_app = app is None
    if owns_app:
        app = connect_coreldraw(visible=visible)

    doc = None
    try:
        doc = app.CreateDocument()
        layer = doc.ActivePage.ActiveLayer
        result = trace_one_in_document(
            layer, doc, input_path, output_path, trace_type
        )
        print("完成。")
        return result
    finally:
        if doc is not None and not keep_document_open:
            try:
                doc.Close()
            except Exception:
                pass
        if owns_app:
            app.Optimization = False


def trace_images_batch(
    directory: Path,
    *,
    trace_preset: str = "high_quality",
    visible: bool = False,
    skip_existing: bool = False,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """批量转换目录内所有图片，复用同一个 CorelDRAW 实例。"""
    images = collect_images(directory)
    if not images:
        raise FileNotFoundError(f"目录中没有找到图片：{directory}")

    trace_type = resolve_trace_type(trace_preset)
    app = connect_coreldraw(visible=visible)
    succeeded: list[Path] = []
    failed: list[tuple[Path, str]] = []

    print(f"共 {len(images)} 张图片待处理\n")
    try:
        for index, input_path in enumerate(images, start=1):
            output_path = input_path.with_suffix(".svg")
            print(f"[{index}/{len(images)}] {input_path.name}")

            if skip_existing and output_path.is_file():
                print(f"  跳过（已存在）：{output_path.name}\n")
                succeeded.append(output_path)
                continue

            doc = None
            try:
                doc = app.CreateDocument()
                layer = doc.ActivePage.ActiveLayer
                trace_one_in_document(
                    layer, doc, input_path, output_path, trace_type
                )
                succeeded.append(output_path)
                print("  完成\n")
            except Exception as exc:
                failed.append((input_path, str(exc)))
                print(f"  失败：{exc}\n", file=sys.stderr)
            finally:
                if doc is not None:
                    try:
                        doc.Close()
                    except Exception:
                        pass
    finally:
        app.Optimization = False

    return succeeded, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用 CorelDRAW 对位图做轮廓描摹并导出 SVG。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入图片路径（默认：{DEFAULT_INPUT.name}）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 SVG 路径（默认：与输入同名 .svg）",
    )
    parser.add_argument(
        "-t",
        "--trace",
        choices=sorted(TRACE_PRESETS),
        default="high_quality",
        help="轮廓描摹预设（默认：high_quality，适合照片）",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="显示 CorelDRAW 窗口（便于调试）",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="处理完成后保留 CorelDRAW 文档不关闭",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换指定目录下的全部图片（默认目录为脚本所在文件夹）",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DATA_DIR,
        help="批量模式下的图片目录（默认：data 目录）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="批量模式下跳过已存在的 SVG 文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.all:
            succeeded, failed = trace_images_batch(
                args.dir.resolve(),
                trace_preset=args.trace,
                visible=args.visible,
                skip_existing=args.skip_existing,
            )
            print(f"批量完成：成功 {len(succeeded)}，失败 {len(failed)}")
            if failed:
                for path, reason in failed:
                    print(f"  失败 {path.name}：{reason}", file=sys.stderr)
                return 1
            return 0

        input_path = args.input.resolve()
        output_path = (
            args.output.resolve()
            if args.output
            else input_path.with_suffix(".svg")
        )
        trace_image_to_svg(
            input_path,
            output_path,
            trace_preset=args.trace,
            visible=args.visible,
            keep_document_open=args.keep_open,
        )
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
