from simple_lama_inpainting import SimpleLama
from PIL import Image

lama_model = None

def get_lama_model():
    global lama_model
    if lama_model is None:
        lama_model = SimpleLama()
    return lama_model

def inpaint_image(image_path: str, mask_path: str, output_path: str) -> dict:
    model = get_lama_model()
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    result = model(image, mask)
    result.save(output_path, format="PNG")
    
    return {"width": result.width, "height": result.height}
