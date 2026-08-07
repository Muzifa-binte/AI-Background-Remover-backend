"""
Image Enhancement Service.

Provides a composable, parameter-driven enhancement pipeline using
Pillow and OpenCV. Each enhancement step is independently controlled
by a float factor (1.0 = no change) or a boolean toggle.

Pipeline order:
  1. Auto white-balance  (optional)
  2. Brightness          (factor 0.0–3.0, 1.0 = unchanged)
  3. Contrast            (factor 0.0–3.0, 1.0 = unchanged)
  4. Saturation          (factor 0.0–3.0, 1.0 = unchanged)
  5. Noise reduction     (optional — bilateral filter)
  6. Sharpness           (factor 0.0–3.0, 1.0 = unchanged)
  7. Save enhanced PNG
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def auto_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Simple grey-world white balance.

    Scales each BGR channel so its mean equals the overall pixel mean.
    Effective for photos shot under coloured artificial light.

    Args:
        image: BGR uint8 NumPy array.

    Returns:
        White-balanced BGR uint8 array.
    """
    result = image.astype(np.float32)
    mean_bgr = result.mean(axis=(0, 1))          # per-channel mean
    overall_mean = mean_bgr.mean()               # scalar target
    scale = overall_mean / (mean_bgr + 1e-6)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def reduce_noise(image: np.ndarray, strength: int = 9) -> np.ndarray:
    """
    Edge-preserving noise reduction via bilateral filtering.

    Bilateral filtering blurs smooth regions while leaving edges sharp,
    making it ideal for portrait retouching and product photos.

    Args:
        image:    BGR uint8 NumPy array.
        strength: Diameter of the filter neighbourhood (odd integer, 5–15).

    Returns:
        Filtered BGR uint8 array.
    """
    # Clamp to odd and within [5, 15] for acceptable speed
    d = max(5, min(15, strength | 1))
    return cv2.bilateralFilter(image, d, sigmaColor=75, sigmaSpace=75)


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. factor=1.0 is unchanged, <1 darker, >1 brighter."""
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast. factor=1.0 is unchanged."""
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_saturation(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust colour saturation. factor=0 → grayscale, 1.0 → unchanged, >1 → vivid.
    Works on both RGB and RGBA images.
    """
    # ImageEnhance.Color only works on RGB/L modes — preserve alpha separately
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        rgb = Image.merge("RGB", (r, g, b))
        rgb = ImageEnhance.Color(rgb).enhance(factor)
        r2, g2, b2 = rgb.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    return ImageEnhance.Color(image).enhance(factor)


def apply_sharpness(image: Image.Image, factor: float) -> Image.Image:
    """
    Adjust sharpness. factor=0 → blurred, 1.0 → unchanged, 2.0 → sharpened.
    For stronger sharpening (factor > 2.0) we stack an UnsharpMask on top.
    """
    sharpened = ImageEnhance.Sharpness(image).enhance(factor)
    if factor > 2.0:
        sharpened = sharpened.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=2)
        )
    return sharpened


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def enhance_image(
    input_path: str,
    output_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    denoise: bool = False,
    auto_wb: bool = False,
    denoise_strength: int = 9,
) -> dict:
    """
    Full enhancement pipeline: load → optional AWB → optional denoise →
    brightness → contrast → saturation → sharpness → save.

    Args:
        input_path:       Path to the source image (any Pillow-readable format).
        output_path:      Destination path for the enhanced PNG.
        brightness:       Brightness multiplier (default 1.0).
        contrast:         Contrast multiplier (default 1.0).
        saturation:       Saturation multiplier (default 1.0).
        sharpness:        Sharpness multiplier (default 1.0).
        denoise:          Whether to apply bilateral noise reduction.
        auto_wb:          Whether to apply grey-world white balance.
        denoise_strength: Bilateral filter neighbourhood diameter (5–15).

    Returns:
        dict with 'width', 'height', and 'mode' of the output image.
    """
    # --- Load via Pillow (preserves RGBA for transparent inputs) -----------
    pil_img = Image.open(input_path)
    original_mode = pil_img.mode

    # --- OpenCV-based steps (AWB + denoise) operate on BGR uint8 ----------
    if auto_wb or denoise:
        # Convert to RGB numpy, run CV steps, convert back to Pillow
        if original_mode == "RGBA":
            r, g, b, a = pil_img.split()
            rgb_pil = Image.merge("RGB", (r, g, b))
        else:
            rgb_pil = pil_img.convert("RGB")
            a = None

        cv_img = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)

        if auto_wb:
            cv_img = auto_white_balance(cv_img)
        if denoise:
            cv_img = reduce_noise(cv_img, strength=denoise_strength)

        rgb_out = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        if a is not None:
            r2, g2, b2 = rgb_out.split()
            pil_img = Image.merge("RGBA", (r2, g2, b2, a))
        else:
            pil_img = rgb_out

    # --- Pillow-based steps -----------------------------------------------
    if brightness != 1.0:
        pil_img = apply_brightness(pil_img, brightness)
    if contrast != 1.0:
        pil_img = apply_contrast(pil_img, contrast)
    if saturation != 1.0:
        pil_img = apply_saturation(pil_img, saturation)
    if sharpness != 1.0:
        pil_img = apply_sharpness(pil_img, sharpness)

    # --- Save ---------------------------------------------------------------
    pil_img.save(output_path, format="PNG")

    return {
        "width": pil_img.width,
        "height": pil_img.height,
        "mode": pil_img.mode,
    }
