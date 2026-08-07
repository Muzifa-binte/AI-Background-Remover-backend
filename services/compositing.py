"""
Background Compositing Service.

Takes a foreground image that already has a transparent background (RGBA PNG)
and composites it over one of three background types:

  "solid"    — a flat RGBA colour  (hex string or rgba tuple)
  "gradient" — a two-colour linear gradient (horizontal / vertical / diagonal)
  "image"    — an arbitrary image file resized to fit the foreground

All compositing is done with PIL.Image.alpha_composite so the subject's
semi-transparent edges blend correctly.
"""

from __future__ import annotations

from typing import Literal, Tuple
import re

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _parse_hex(hex_color: str) -> Tuple[int, int, int, int]:
    """
    Convert a CSS hex colour string to an RGBA tuple.

    Accepts:  #RGB  #RRGGBB  #RGBA  #RRGGBBAA  (leading # optional)
    Returns:  (R, G, B, A) each in [0, 255].
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 4:
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        h += "ff"
    if len(h) != 8:
        raise ValueError(f"Invalid hex colour: {hex_color!r}")
    r, g, b, a = (int(h[i:i+2], 16) for i in (0, 2, 4, 6))
    return r, g, b, a


# ---------------------------------------------------------------------------
# Background generators  (all return an RGBA Image the same size as fg)
# ---------------------------------------------------------------------------

def _solid_background(
    width: int,
    height: int,
    color: str = "#ffffff",
) -> Image.Image:
    """
    Create a flat-colour RGBA canvas.

    Args:
        width, height: Dimensions to match the foreground.
        color:         CSS hex string (e.g. "#e8336d", "#fff", "#ffffff80").
    """
    rgba = _parse_hex(color)
    canvas = Image.new("RGBA", (width, height), rgba)
    return canvas


def _gradient_background(
    width: int,
    height: int,
    color_start: str = "#e8336d",
    color_end: str   = "#2fbfb0",
    direction: Literal["horizontal", "vertical", "diagonal"] = "vertical",
) -> Image.Image:
    """
    Create a smooth two-colour linear gradient.

    Args:
        width, height: Canvas dimensions.
        color_start:   CSS hex for the start colour.
        color_end:     CSS hex for the end colour.
        direction:     "horizontal" | "vertical" | "diagonal"
    """
    r1, g1, b1, a1 = _parse_hex(color_start)
    r2, g2, b2, a2 = _parse_hex(color_end)

    canvas = Image.new("RGBA", (width, height))
    draw   = ImageDraw.Draw(canvas)

    if direction == "horizontal":
        steps = width
        for x in range(steps):
            t  = x / max(steps - 1, 1)
            r  = round(r1 + (r2 - r1) * t)
            g  = round(g1 + (g2 - g1) * t)
            b  = round(b1 + (b2 - b1) * t)
            a  = round(a1 + (a2 - a1) * t)
            draw.line([(x, 0), (x, height - 1)], fill=(r, g, b, a))

    elif direction == "vertical":
        steps = height
        for y in range(steps):
            t  = y / max(steps - 1, 1)
            r  = round(r1 + (r2 - r1) * t)
            g  = round(g1 + (g2 - g1) * t)
            b  = round(b1 + (b2 - b1) * t)
            a  = round(a1 + (a2 - a1) * t)
            draw.line([(0, y), (width - 1, y)], fill=(r, g, b, a))

    else:  # diagonal
        # Walk along the diagonal axis (top-left → bottom-right)
        steps  = max(width, height)
        for i in range(steps):
            t  = i / max(steps - 1, 1)
            r  = round(r1 + (r2 - r1) * t)
            g  = round(g1 + (g2 - g1) * t)
            b  = round(b1 + (b2 - b1) * t)
            a  = round(a1 + (a2 - a1) * t)
            # Draw diagonal bands
            x0 = i * width  // steps
            y0 = 0
            x1 = width - 1
            y1 = i * height // steps
            draw.line([(x0, y0), (x1, y1)], fill=(r, g, b, a))
        # Fill remaining corner gaps with a flood approach
        # Re-render with numpy for correctness on diagonal
        import numpy as np
        xs = np.arange(width,  dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        t_map  = (xx / (width - 1) + yy / (height - 1)) / 2  # [0,1]
        t_map  = np.clip(t_map, 0, 1)
        img_array = np.zeros((height, width, 4), dtype=np.uint8)
        for ch_idx, (c1, c2) in enumerate([(r1,r2),(g1,g2),(b1,b2),(a1,a2)]):
            img_array[:, :, ch_idx] = np.round(c1 + (c2 - c1) * t_map).astype(np.uint8)
        canvas = Image.fromarray(img_array, "RGBA")

    return canvas


def _image_background(
    width: int,
    height: int,
    bg_image_path: str,
    fit: Literal["cover", "contain", "stretch"] = "cover",
) -> Image.Image:
    """
    Load an image from disk and resize it to cover the canvas.

    Args:
        width, height:  Dimensions of the foreground.
        bg_image_path:  Path to the background image file.
        fit:            "cover"   — crop-fill (default, most natural)
                        "contain" — letterbox with transparency
                        "stretch" — ignore aspect ratio
    """
    bg = Image.open(bg_image_path).convert("RGBA")

    if fit == "stretch":
        bg = bg.resize((width, height), Image.LANCZOS)

    elif fit == "contain":
        bg.thumbnail((width, height), Image.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        offset_x = (width  - bg.width)  // 2
        offset_y = (height - bg.height) // 2
        canvas.paste(bg, (offset_x, offset_y))
        bg = canvas

    else:  # cover
        src_ratio = bg.width  / bg.height
        dst_ratio = width     / height
        if src_ratio > dst_ratio:
            # Source is wider → fit height, crop width
            new_h = height
            new_w = round(bg.width * height / bg.height)
        else:
            # Source is taller → fit width, crop height
            new_w = width
            new_h = round(bg.height * width / bg.width)
        bg = bg.resize((new_w, new_h), Image.LANCZOS)
        # Centre-crop
        left = (new_w - width)  // 2
        top  = (new_h - height) // 2
        bg   = bg.crop((left, top, left + width, top + height))

    return bg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def composite_background(
    foreground_path: str,
    output_path: str,
    bg_type: Literal["solid", "gradient", "image"] = "solid",
    # solid
    solid_color: str = "#ffffff",
    # gradient
    gradient_color_start: str = "#e8336d",
    gradient_color_end:   str = "#2fbfb0",
    gradient_direction: Literal["horizontal", "vertical", "diagonal"] = "vertical",
    # image
    bg_image_path: str | None = None,
    bg_fit: Literal["cover", "contain", "stretch"] = "cover",
) -> dict:
    """
    Composite a transparent-bg foreground over the chosen background and save.

    Args:
        foreground_path:      RGBA PNG with the subject (transparent bg).
        output_path:          Destination path for the composited PNG.
        bg_type:              "solid" | "gradient" | "image"
        solid_color:          Hex colour for solid mode.
        gradient_color_start: Hex start colour for gradient mode.
        gradient_color_end:   Hex end colour for gradient mode.
        gradient_direction:   Gradient direction.
        bg_image_path:        Path to bg image (required for image mode).
        bg_fit:               Resize strategy for image mode.

    Returns:
        dict with 'width', 'height' of the composited output.

    Raises:
        ValueError: If bg_type is "image" but bg_image_path is not provided,
                    or if an unsupported bg_type is given.
    """
    fg = Image.open(foreground_path).convert("RGBA")
    w, h = fg.size

    if bg_type == "solid":
        background = _solid_background(w, h, color=solid_color)

    elif bg_type == "gradient":
        background = _gradient_background(
            w, h,
            color_start=gradient_color_start,
            color_end=gradient_color_end,
            direction=gradient_direction,
        )

    elif bg_type == "image":
        if not bg_image_path:
            raise ValueError("bg_image_path is required when bg_type='image'.")
        background = _image_background(w, h, bg_image_path, fit=bg_fit)

    else:
        raise ValueError(f"Unsupported bg_type: {bg_type!r}. Use 'solid', 'gradient', or 'image'.")

    # Composite: background first, then foreground on top
    composited = Image.alpha_composite(background, fg)
    composited.save(output_path, format="PNG")

    return {"width": w, "height": h}
