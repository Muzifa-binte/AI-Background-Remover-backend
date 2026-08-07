"""
Smart Cropping Service.

Two-mode cropping pipeline:

Mode A — Subject crop (images with clear foreground/background separation):
  Detect the subject bounding box from the alpha mask of the bg-removed PNG,
  apply padding, aspect-ratio lock, crop the ORIGINAL image to that box.

Mode B — Center crop (full-bleed images / designs where entire frame is subject):
  When the detected bbox covers ≥ 85% of image area, skip alpha detection
  entirely and perform a centred crop of the ORIGINAL to the target aspect ratio.
  This ensures aspect ratio and padding controls always produce a visible change.

Either way, the final crop is applied to the original RGB/RGBA source so the
result looks natural rather than showing a transparent checkerboard.
"""

from __future__ import annotations

from typing import Literal
import numpy as np
import cv2
from PIL import Image


ASPECT_RATIOS: dict[str, tuple[float, float] | None] = {
    "free":   None,
    "1:1":    (1.0, 1.0),
    "4:3":    (4.0, 3.0),
    "3:4":    (3.0, 4.0),
    "16:9":   (16.0, 9.0),
    "9:16":   (9.0, 16.0),
    "3:2":    (3.0, 2.0),
    "2:3":    (2.0, 3.0),
    "5:4":    (5.0, 4.0),
    "4:5":    (4.0, 5.0),
}

# If detected bbox area / total area >= this, switch to center-crop mode
_FULL_FRAME_THRESHOLD = 0.85


# ── Internal helpers ───────────────────────────────────────────────────────

def _bbox_from_alpha(alpha: np.ndarray, threshold: int = 128) -> tuple[int, int, int, int]:
    """Return (x0,y0,x1,y1) of the largest foreground blob. Exclusive bounds."""
    binary = (alpha >= threshold).astype(np.uint8) * 255
    if binary.max() == 0:
        binary = (alpha >= 10).astype(np.uint8) * 255
        if binary.max() == 0:
            raise ValueError("Image appears fully transparent — nothing to crop.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        loose = alpha >= 10
        rows, cols = np.any(loose, axis=1), np.any(loose, axis=0)
        y0 = int(np.argmax(rows));  y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        x0 = int(np.argmax(cols));  x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        return x0, y0, x1 + 1, y1 + 1

    areas = stats[1:, cv2.CC_STAT_AREA]
    lbl   = int(np.argmax(areas)) + 1
    x0    = int(stats[lbl, cv2.CC_STAT_LEFT])
    y0    = int(stats[lbl, cv2.CC_STAT_TOP])
    bw    = int(stats[lbl, cv2.CC_STAT_WIDTH])
    bh    = int(stats[lbl, cv2.CC_STAT_HEIGHT])
    return x0, y0, x0 + bw, y0 + bh


def _center_crop_to_ratio(
    img_w: int, img_h: int,
    ratio_w: float, ratio_h: float,
) -> tuple[int, int, int, int]:
    """
    Return the largest centred crop box that fits ratio_w:ratio_h
    inside img_w × img_h. Always shrinks, never expands.
    """
    target = ratio_w / ratio_h
    current = img_w / img_h
    if current > target:
        # Image is wider — fit height, crop width
        new_w = int(img_h * target)
        x0 = (img_w - new_w) // 2
        return x0, 0, x0 + new_w, img_h
    else:
        # Image is taller — fit width, crop height
        new_h = int(img_w / target)
        y0 = (img_h - new_h) // 2
        return 0, y0, img_w, y0 + new_h


def _apply_padding(
    x0: int, y0: int, x1: int, y1: int,
    padding_pct: float, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    pad_x = int((x1 - x0) * padding_pct)
    pad_y = int((y1 - y0) * padding_pct)
    return (
        max(0,     x0 - pad_x),
        max(0,     y0 - pad_y),
        min(img_w, x1 + pad_x),
        min(img_h, y1 + pad_y),
    )


def _apply_aspect_ratio(
    x0: int, y0: int, x1: int, y1: int,
    ratio_w: float, ratio_h: float,
    img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Expand bbox (never shrink) to match ratio, clamped to image bounds."""
    bw, bh = x1 - x0, y1 - y0
    target  = ratio_w / ratio_h
    current = bw / bh if bh > 0 else 1.0
    if current < target:
        new_w = int(bh * target)
        cx    = (x0 + x1) // 2
        x0    = max(0,     cx - new_w // 2)
        x1    = min(img_w, x0 + new_w)
        x0    = max(0,     x1 - new_w)
    else:
        new_h = int(bw / target)
        cy    = (y0 + y1) // 2
        y0    = max(0,     cy - new_h // 2)
        y1    = min(img_h, y0 + new_h)
        y0    = max(0,     y1 - new_h)
    return x0, y0, x1, y1


# ── Public entry point ─────────────────────────────────────────────────────

def smart_crop(
    fg_path: str,
    output_path: str,
    original_path: str | None = None,
    padding_pct:   float = 0.05,
    aspect_ratio:  str   = "free",
    min_size:      int   = 64,
    alpha_threshold: int = 128,
) -> dict:
    """
    Smart-crop an image.

    Args:
        fg_path:        RGBA PNG (bg-removed) — used for subject detection only.
        output_path:    Destination for the cropped PNG.
        original_path:  Path to the original upload. If provided, the final crop
                        is applied to this image (better quality, no transparency).
                        Falls back to fg_path if None.
        padding_pct:    Extra space around subject as fraction of bbox (0–0.5).
        aspect_ratio:   One of ASPECT_RATIOS keys.
        min_size:       Minimum output side length in px.
        alpha_threshold: Alpha value for foreground detection (default 128).

    Returns:
        dict with crop_box, width, height, original dims, crop_mode.
    """
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(
            f"Unknown aspect_ratio '{aspect_ratio}'. "
            f"Choose from: {list(ASPECT_RATIOS.keys())}"
        )

    # Load fg for detection; load source for the actual crop
    fg_img   = Image.open(fg_path).convert("RGBA")
    img_w, img_h = fg_img.size
    alpha    = np.array(fg_img)[:, :, 3]
    total_px = img_w * img_h

    # Load the image we'll actually crop (original if available, else fg)
    src_path = original_path if original_path else fg_path
    src_img  = Image.open(src_path)

    padding_pct = max(0.0, min(0.5, padding_pct))
    ratio       = ASPECT_RATIOS[aspect_ratio]
    crop_mode   = "subject"

    # ── Try subject detection ──────────────────────────────────────────────
    try:
        x0, y0, x1, y1 = _bbox_from_alpha(alpha, threshold=alpha_threshold)
        bbox_area = (x1 - x0) * (y1 - y0)
        coverage  = bbox_area / max(total_px, 1)

        if coverage >= _FULL_FRAME_THRESHOLD:
            # Full-bleed image — switch to center-crop mode
            raise ValueError("full-frame")

        # Subject found — apply padding then aspect ratio
        x0, y0, x1, y1 = _apply_padding(x0, y0, x1, y1, padding_pct, img_w, img_h)
        if ratio is not None:
            x0, y0, x1, y1 = _apply_aspect_ratio(x0, y0, x1, y1, ratio[0], ratio[1], img_w, img_h)

    except ValueError:
        # ── Center-crop fallback ───────────────────────────────────────────
        crop_mode = "center"
        if ratio is not None:
            x0, y0, x1, y1 = _center_crop_to_ratio(img_w, img_h, ratio[0], ratio[1])
        else:
            # "free" + full-frame → apply padding inward from edges
            pad_x = int(img_w * padding_pct)
            pad_y = int(img_h * padding_pct)
            x0, y0 = pad_x, pad_y
            x1, y1 = img_w - pad_x, img_h - pad_y
            # Ensure we don't invert the box
            if x1 <= x0: x0, x1 = 0, img_w
            if y1 <= y0: y0, y1 = 0, img_h

    # ── Enforce minimum size ───────────────────────────────────────────────
    bw, bh = x1 - x0, y1 - y0
    if bw < min_size or bh < min_size:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half   = max(min_size, bw, bh) // 2
        x0 = max(0,     cx - half);  x1 = min(img_w, cx + half)
        y0 = max(0,     cy - half);  y1 = min(img_h, cy + half)

    # ── Crop source image and save ─────────────────────────────────────────
    # Ensure src_img is same logical size as fg_img
    if src_img.size != (img_w, img_h):
        src_img = src_img.resize((img_w, img_h), Image.LANCZOS)

    cropped = src_img.crop((x0, y0, x1, y1))
    cropped.save(output_path, format="PNG")

    return {
        "crop_box":  {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "width":     x1 - x0,
        "height":    y1 - y0,
        "original":  {"width": img_w, "height": img_h},
        "crop_mode": crop_mode,
    }


def get_aspect_ratio_keys() -> list[str]:
    return list(ASPECT_RATIOS.keys())
