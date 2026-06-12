from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "output" / "image14" / "graph" / "1.png"


@dataclass
class Cluster:
    label: int
    count: int
    lab: np.ndarray
    rgb: np.ndarray


def delta_e76(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_lab_flat{input_path.suffix}")


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def bilateral_denoise(rgb: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if diameter <= 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr, diameter, sigma_color, sigma_space)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def lab_float_to_rgb(lab: np.ndarray) -> np.ndarray:
    cv_lab = lab.copy().astype(np.float32)
    cv_lab[:, :, 0] *= 255.0 / 100.0
    cv_lab[:, :, 1] += 128.0
    cv_lab[:, :, 2] += 128.0
    cv_lab = np.clip(cv_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv_lab, cv2.COLOR_LAB2RGB)


def run_kmeans(lab: np.ndarray, k: int, attempts: int, sample_limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)

    if sample_limit and len(pixels) > sample_limit:
        rng = np.random.default_rng(1234)
        sample_idx = rng.choice(len(pixels), size=sample_limit, replace=False)
        train_pixels = pixels[sample_idx]
    else:
        train_pixels = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
    _compactness, _labels, centers = cv2.kmeans(
        train_pixels,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    diff = pixels[:, None, :] - centers[None, :, :]
    labels = np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int32)
    return labels.reshape(h, w), centers.astype(np.float32)


def clusters_from_labels(labels: np.ndarray, centers_lab: np.ndarray) -> list[Cluster]:
    counts = np.bincount(labels.reshape(-1), minlength=len(centers_lab))
    centers_rgb = lab_float_to_rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    clusters: list[Cluster] = []
    for idx, count in enumerate(counts):
        clusters.append(
            Cluster(
                label=idx,
                count=int(count),
                lab=centers_lab[idx],
                rgb=centers_rgb[idx].astype(np.float32),
            )
        )
    return clusters


def find_background_label(labels: np.ndarray, clusters: list[Cluster], white_l_threshold: float) -> int:
    h, w = labels.shape
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_counts = np.bincount(border, minlength=len(clusters))
    whiteish = [cluster.label for cluster in clusters if cluster.lab[0] >= white_l_threshold]
    if whiteish:
        return max(whiteish, key=lambda label: border_counts[label] * 3 + clusters[label].count)
    return int(np.argmax(border_counts))


def merge_close_clusters(
    labels: np.ndarray,
    clusters: list[Cluster],
    background_label: int,
    merge_delta: float,
    bg_delta: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    parent = list(range(len(clusters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bg_lab = clusters[background_label].lab
    for cluster in clusters:
        if cluster.label != background_label and delta_e76(cluster.lab, bg_lab) <= bg_delta:
            union(background_label, cluster.label)

    for i, a in enumerate(clusters):
        for b in clusters[i + 1 :]:
            if delta_e76(a.lab, b.lab) <= merge_delta:
                union(a.label, b.label)

    groups: dict[int, list[int]] = {}
    for cluster in clusters:
        groups.setdefault(find(cluster.label), []).append(cluster.label)

    root_to_new = {root: idx for idx, root in enumerate(groups)}
    remap = np.zeros(len(clusters), dtype=np.int32)
    palette_lab: list[np.ndarray] = []
    for root, members in groups.items():
        weights = np.array([clusters[label].count for label in members], dtype=np.float32)
        labs = np.array([clusters[label].lab for label in members], dtype=np.float32)
        palette_lab.append(np.average(labs, axis=0, weights=weights))
        for label in members:
            remap[label] = root_to_new[root]

    return remap[labels], palette_lab


def remove_small_components(labels: np.ndarray, min_area: int, background_label: int = 0) -> np.ndarray:
    if min_area <= 1:
        return labels

    out = labels.copy()
    for label in sorted(set(int(v) for v in np.unique(labels))):
        if label == background_label:
            continue
        mask = (out == label).astype(np.uint8)
        count, components, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_idx in range(1, count):
            area = int(stats[component_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                out[components == component_idx] = background_label
    return out


def reorder_background_first(labels: np.ndarray, palette_lab: list[np.ndarray], background_label: int) -> tuple[np.ndarray, list[np.ndarray]]:
    if background_label == 0:
        return labels, palette_lab

    mapping = {background_label: 0, 0: background_label}
    for idx in range(len(palette_lab)):
        mapping.setdefault(idx, idx)

    out = labels.copy()
    for old, new in mapping.items():
        out[labels == old] = new

    new_palette = list(palette_lab)
    new_palette[0], new_palette[background_label] = new_palette[background_label], new_palette[0]
    return out, new_palette


def labels_to_rgb(labels: np.ndarray, palette_lab: list[np.ndarray]) -> np.ndarray:
    palette_rgb = lab_float_to_rgb(np.array(palette_lab, dtype=np.float32).reshape(1, -1, 3)).reshape(-1, 3)
    return palette_rgb[labels].astype(np.uint8)


def flatten_lab_kmeans(
    rgb: np.ndarray,
    *,
    bilateral_diameter: int,
    sigma_color: float,
    sigma_space: float,
    k: int,
    attempts: int,
    sample_limit: int | None,
    merge_delta: float,
    bg_delta: float,
    white_l_threshold: float,
    min_component_area: int,
) -> tuple[np.ndarray, dict]:
    denoised = bilateral_denoise(rgb, bilateral_diameter, sigma_color, sigma_space)
    lab = rgb_to_lab_float(denoised)
    labels, centers_lab = run_kmeans(lab, k, attempts, sample_limit)
    clusters = clusters_from_labels(labels, centers_lab)
    original_bg = find_background_label(labels, clusters, white_l_threshold)
    merged_labels, palette_lab = merge_close_clusters(labels, clusters, original_bg, merge_delta, bg_delta)

    merged_bg = int(merged_labels[labels == original_bg][0])
    merged_labels, palette_lab = reorder_background_first(merged_labels, palette_lab, merged_bg)
    cleaned = remove_small_components(merged_labels, min_component_area, background_label=0)
    output = labels_to_rgb(cleaned, palette_lab)

    stats = {
        "input_k": k,
        "clusters_after_merge": len(palette_lab),
        "background_rgb": tuple(int(v) for v in output[cleaned == 0][0]) if np.any(cleaned == 0) else None,
        "unique_output_colors": len(np.unique(output.reshape(-1, 3), axis=0)),
    }
    return output, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten image colors with Bilateral Filter + Lab K-means + Lab color merging.")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--bilateral-diameter", type=int, default=9)
    parser.add_argument("--sigma-color", type=float, default=55.0)
    parser.add_argument("--sigma-space", type=float, default=55.0)
    parser.add_argument("-k", "--clusters", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--sample-limit", type=int, default=250000)
    parser.add_argument("--merge-delta", type=float, default=8.5)
    parser.add_argument("--bg-delta", type=float, default=5.0)
    parser.add_argument("--white-l-threshold", type=float, default=94.0)
    parser.add_argument("--min-component-area", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(input_path).resolve()
    try:
        rgb = load_rgb(input_path)
        output, stats = flatten_lab_kmeans(
            rgb,
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
        Image.fromarray(output).save(output_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
