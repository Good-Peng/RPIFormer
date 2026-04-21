import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from basicsr.metrics import calculate_lpips, calculate_psnr, calculate_ssim

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".npy"}


@dataclass(frozen=True)
class Pair:
    pred: Path
    gt: Path


def _iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*")
    else:
        yield from root.glob("*")


def _is_valid_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _imread_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def _build_gt_index(
    gt_root: Path, recursive: bool
) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    rel_index: Dict[str, Path] = {}
    stem_index: Dict[str, List[Path]] = {}
    for p in _iter_files(gt_root, recursive=recursive):
        if not _is_valid_image(p):
            continue
        rel = str(p.relative_to(gt_root)).replace("\\", "/")
        rel_index[rel] = p
        stem_index.setdefault(p.stem, []).append(p)
    return (rel_index, stem_index)


def _pair_files(
    pred_root: Path, gt_root: Path, recursive: bool, match: str
) -> List[Pair]:
    rel_index, stem_index = _build_gt_index(gt_root, recursive=recursive)
    pairs: List[Pair] = []
    missing: List[Path] = []
    for pred in _iter_files(pred_root, recursive=recursive):
        if not _is_valid_image(pred):
            continue
        rel = str(pred.relative_to(pred_root)).replace("\\", "/")
        gt: Optional[Path] = None
        if match in ("relative", "auto"):
            gt = rel_index.get(rel)
        if gt is None and match in ("stem", "auto"):
            pred_key = pred.stem
            candidates = stem_index.get(pred_key, [])
            if len(candidates) == 1:
                gt = candidates[0]
        if gt is None:
            missing.append(pred)
            continue
        pairs.append(Pair(pred=pred, gt=gt))
    if missing:
        sample = "\n".join((str(p) for p in missing[:10]))
        raise ValueError(
            f"Failed to find GT for {len(missing)} predicted files. Match mode={match}. Sample:\n{sample}"
        )
    if not pairs:
        raise ValueError(
            f"No image pairs found under pred_root={pred_root} and gt_root={gt_root}."
        )
    return sorted(pairs, key=lambda x: str(x.pred))


def _maybe_resize_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    gt_h, gt_w = (gt.shape[0], gt.shape[1])
    pred_resized = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
    if pred_resized.ndim == 2:
        pred_resized = pred_resized[..., None]
    if gt.ndim == 3 and pred_resized.ndim == 2:
        pred_resized = np.repeat(pred_resized[..., None], gt.shape[2], axis=2)
    return pred_resized


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()
    return img


def _normalize_stem(stem: str, suffix: str) -> str:
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS between a folder of predictions and GT using BasicSR metrics."
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Folder containing predicted images."
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Folder containing GT images."
    )
    parser.add_argument(
        "--match",
        type=str,
        default="auto",
        choices=["auto", "relative", "stem"],
        help="How to match pred/gt files: by relative path, by filename stem, or auto (relative then stem).",
    )
    parser.add_argument(
        "--gt-stem-suffix",
        type=str,
        default="",
        help="If GT files have an extra stem suffix (e.g. '_gt'), strip it when matching by stem.",
    )
    parser.add_argument(
        "--gt-file-suffix",
        type=str,
        default="",
        help="Only index GT files whose stem ends with this suffix (e.g. '_gt').",
    )
    parser.add_argument(
        "--pred-exclude-stem-suffix",
        type=str,
        default="",
        help="Exclude predicted files whose stem ends with this suffix (e.g. '_gt' in BasicSR visualization).",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search pred/gt folders."
    )
    parser.add_argument(
        "--crop-border", type=int, default=0, help="Crop border pixels on each side."
    )
    parser.add_argument(
        "--test-y-channel", action="store_true", help="Compute on Y channel (YCbCr)."
    )
    parser.add_argument(
        "--input-order",
        type=str,
        default="HWC",
        choices=["HWC", "CHW"],
        help="Input order for metric functions (usually HWC for images loaded by cv2).",
    )
    parser.add_argument(
        "--resize-to-gt",
        action="store_true",
        help="Resize pred to GT size if shapes mismatch (not recommended unless you know what you're doing).",
    )
    parser.add_argument(
        "--lpips-use-bgr",
        action="store_true",
        help="By default, convert BGR->RGB before LPIPS (recommended). Set this to keep BGR as-is.",
    )
    parser.add_argument(
        "--out-csv", type=str, default="", help="Optional CSV output path."
    )
    args = parser.parse_args()
    pred_root = Path(args.pred).expanduser().resolve()
    gt_root = Path(args.gt).expanduser().resolve()
    if not pred_root.exists():
        raise FileNotFoundError(pred_root)
    if not gt_root.exists():
        raise FileNotFoundError(gt_root)
    rel_index_raw, stem_index_raw_all = _build_gt_index(
        gt_root, recursive=args.recursive
    )
    rel_index: Dict[str, Path] = {}
    stem_index_raw: Dict[str, List[Path]] = {}
    for rel, p in rel_index_raw.items():
        if args.gt_file_suffix and (not p.stem.endswith(args.gt_file_suffix)):
            continue
        rel_index[rel] = p
        stem_index_raw.setdefault(p.stem, []).append(p)
    if args.gt_stem_suffix:
        stem_index: Dict[str, List[Path]] = {}
        for stem, paths in stem_index_raw.items():
            stem_norm = _normalize_stem(stem, args.gt_stem_suffix)
            stem_index.setdefault(stem_norm, []).extend(paths)

        def pair_files_with_custom_stem() -> List[Pair]:
            pairs: List[Pair] = []
            missing: List[Path] = []
            for pred in _iter_files(pred_root, recursive=args.recursive):
                if not _is_valid_image(pred):
                    continue
                if args.pred_exclude_stem_suffix and pred.stem.endswith(
                    args.pred_exclude_stem_suffix
                ):
                    continue
                rel = str(pred.relative_to(pred_root)).replace("\\", "/")
                gt: Optional[Path] = None
                if args.match in ("relative", "auto"):
                    gt = rel_index.get(rel)
                if gt is None and args.match in ("stem", "auto"):
                    candidates = stem_index.get(pred.stem, [])
                    if len(candidates) == 1:
                        gt = candidates[0]
                if gt is None:
                    missing.append(pred)
                    continue
                pairs.append(Pair(pred=pred, gt=gt))
            if missing:
                sample = "\n".join((str(p) for p in missing[:10]))
                raise ValueError(
                    f"Failed to find GT for {len(missing)} predicted files. Match mode={args.match}. Sample:\n{sample}"
                )
            if not pairs:
                raise ValueError(
                    f"No image pairs found under pred_root={pred_root} and gt_root={gt_root}."
                )
            return sorted(pairs, key=lambda x: str(x.pred))

        pairs = pair_files_with_custom_stem()
    else:

        def pair_files_simple() -> List[Pair]:
            pairs: List[Pair] = []
            missing: List[Path] = []
            for pred in _iter_files(pred_root, recursive=args.recursive):
                if not _is_valid_image(pred):
                    continue
                if args.pred_exclude_stem_suffix and pred.stem.endswith(
                    args.pred_exclude_stem_suffix
                ):
                    continue
                rel = str(pred.relative_to(pred_root)).replace("\\", "/")
                gt: Optional[Path] = None
                if args.match in ("relative", "auto"):
                    gt = rel_index.get(rel)
                if gt is None and args.match in ("stem", "auto"):
                    candidates = stem_index_raw.get(pred.stem, [])
                    if len(candidates) == 1:
                        gt = candidates[0]
                if gt is None:
                    missing.append(pred)
                    continue
                pairs.append(Pair(pred=pred, gt=gt))
            if missing:
                sample = "\n".join((str(p) for p in missing[:10]))
                raise ValueError(
                    f"Failed to find GT for {len(missing)} predicted files. Match mode={args.match}. Sample:\n{sample}"
                )
            if not pairs:
                raise ValueError(
                    f"No image pairs found under pred_root={pred_root} and gt_root={gt_root}."
                )
            return sorted(pairs, key=lambda x: str(x.pred))

        pairs = pair_files_simple()
    rows: List[Dict[str, object]] = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    for i, pair in enumerate(pairs, start=1):
        pred = _imread_any(pair.pred)
        gt = _imread_any(pair.gt)
        if args.resize_to_gt:
            pred = _maybe_resize_to_gt(pred, gt)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for:\n  pred={pair.pred} {pred.shape}\n  gt={pair.gt} {gt.shape}"
            )
        psnr = calculate_psnr(
            pred,
            gt,
            crop_border=args.crop_border,
            input_order=args.input_order,
            test_y_channel=args.test_y_channel,
        )
        ssim = calculate_ssim(
            pred,
            gt,
            crop_border=args.crop_border,
            input_order=args.input_order,
            test_y_channel=args.test_y_channel,
        )
        pred_lpips = pred if args.lpips_use_bgr else _bgr_to_rgb(pred)
        gt_lpips = gt if args.lpips_use_bgr else _bgr_to_rgb(gt)
        lpips = calculate_lpips(
            pred_lpips,
            gt_lpips,
            crop_border=args.crop_border,
            input_order=args.input_order,
            test_y_channel=False,
        )
        psnr_sum += float(psnr)
        ssim_sum += float(ssim)
        lpips_sum += float(lpips)
        rel_pred = str(pair.pred.relative_to(pred_root)).replace("\\", "/")
        rel_gt = str(pair.gt.relative_to(gt_root)).replace("\\", "/")
        rows.append(
            {
                "name": pair.pred.stem,
                "pred": rel_pred,
                "gt": rel_gt,
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips": float(lpips),
            }
        )
        if i % 50 == 0 or i == len(pairs):
            print(
                f"[{i}/{len(pairs)}] psnr={psnr_sum / i:.4f} ssim={ssim_sum / i:.6f} lpips={lpips_sum / i:.4f}"
            )
    mean_psnr = psnr_sum / len(pairs)
    mean_ssim = ssim_sum / len(pairs)
    mean_lpips = lpips_sum / len(pairs)
    print("==== Mean ====")
    print(f"PSNR : {mean_psnr:.4f}")
    print(f"SSIM : {mean_ssim:.6f}")
    print(f"LPIPS: {mean_lpips:.4f}")
    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["name", "pred", "gt", "psnr", "ssim", "lpips"]
            )
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(
                {
                    "name": "__mean__",
                    "pred": "",
                    "gt": "",
                    "psnr": mean_psnr,
                    "ssim": mean_ssim,
                    "lpips": mean_lpips,
                }
            )


if __name__ == "__main__":
    main()
