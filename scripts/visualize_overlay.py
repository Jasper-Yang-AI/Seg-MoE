from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir, load_jsonl


def _palette() -> list[tuple[int, int, int]]:
    # Stable, high-contrast palette (id->RGB). id=0 treated as background.
    return [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
    ]


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("L", "I;16", "I"):
        return img.convert("RGB")
    return img.convert("RGB")


def _overlay(image: Image.Image, mask: np.ndarray, num_classes: int, alpha: float) -> Image.Image:
    img = _to_rgb(image)
    img_np = np.array(img, dtype=np.float32)
    out = img_np.copy()

    pal = _palette()
    for cid in range(1, num_classes):
        m = mask == cid
        if not np.any(m):
            continue
        color = np.array(pal[cid % len(pal)], dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color

    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    # Draw simple boundary for each present class
    draw = ImageDraw.Draw(out_img)
    H, W = mask.shape
    for cid in range(1, num_classes):
        m = mask == cid
        if not np.any(m):
            continue
        # boundary: pixels that have a neighbor of different label
        boundary = np.zeros_like(mask, dtype=bool)
        boundary[1:, :] |= m[1:, :] & (~m[:-1, :])
        boundary[:-1, :] |= m[:-1, :] & (~m[1:, :])
        boundary[:, 1:] |= m[:, 1:] & (~m[:, :-1])
        boundary[:, :-1] |= m[:, :-1] & (~m[:, 1:])
        ys, xs = np.where(boundary)
        col = _palette()[cid % len(pal)]
        for y, x in zip(ys.tolist(), xs.tolist()):
            draw.point((x, y), fill=col)

    return out_img


def main() -> None:
    ap = argparse.ArgumentParser(description="Random overlay sanity visualization: image + mask (alpha blend + boundaries)")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--split-jsonl", default=None, help="Optional split jsonl to sample from (e.g., splits_*.jsonl)")
    ap.add_argument("--out-dir", default=None, help="Default: runs/sanity_vis/<dataset>")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.35)
    args = ap.parse_args()

    dcfg = load_config(args.dataset_config)
    num_classes = int(dcfg["task"]["num_classes"])

    splits_dir = Path(dcfg["paths"]["splits_dir"])
    index_all = splits_dir / "index_all.jsonl"
    rows = load_jsonl(Path(args.split_jsonl)) if args.split_jsonl else load_jsonl(index_all)

    rng = np.random.default_rng(args.seed)
    if len(rows) == 0:
        raise SystemExit("No rows to visualize")

    n = min(int(args.n), len(rows))
    idx = rng.choice(len(rows), size=n, replace=False)

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "sanity_vis" / dcfg["name"]
    ensure_dir(out_dir)

    # Try to load a default font (falls back silently)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for j in idx:
        r = rows[int(j)]
        img_p = Path(r["image_path"])
        msk_p = Path(r["mask_path"])
        sid = str(r.get("id", img_p.stem))

        img = Image.open(img_p)
        msk = np.array(Image.open(msk_p).convert("L"), dtype=np.int64)

        vis = _overlay(img, msk, num_classes=num_classes, alpha=float(args.alpha))

        present = sorted([int(x) for x in np.unique(msk).tolist()])
        draw = ImageDraw.Draw(vis)
        text = f"id={sid} | present={present}"
        draw.rectangle([0, 0, vis.size[0], 16], fill=(0, 0, 0))
        draw.text((2, 2), text, fill=(255, 255, 255), font=font)

        out_path = out_dir / f"{sid}.overlay.png"
        vis.save(out_path)

    print(f"Wrote overlays to: {out_dir}")


if __name__ == "__main__":
    main()
