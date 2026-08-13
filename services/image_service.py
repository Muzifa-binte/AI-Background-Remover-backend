import io
from PIL import Image

class ImageService:
    """Handles image validation, processing, and preparation before AI calls."""

    def validate(self, image_bytes: bytes) -> bool:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.format in ["JPEG", "JPG", "PNG", "WEBP"]
        except Exception:
            return False

    def preprocess(self, image_bytes: bytes) -> bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Convert RGBA/LA/P to RGB (substitute transparency with white background)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if dimensions exceed 1024px to reduce transfer payload and latency
            max_size = 1024
            if max(img.width, img.height) > max_size:
                if img.width > img.height:
                    new_width = max_size
                    new_height = int(img.height * (max_size / img.width))
                else:
                    new_height = max_size
                    new_width = int(img.width * (max_size / img.height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Export back to JPEG bytes
            out_buf = io.BytesIO()
            img.save(out_buf, format="JPEG", quality=85)
            return out_buf.getvalue()
        except Exception:
            # Fallback to raw bytes if preprocessing fails
            return image_bytes
