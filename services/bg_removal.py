"""
Background removal service.

Bridges the FastAPI route layer with the AI inference pipeline.
"""

import sys
import asyncio
from pathlib import Path

# Add the AI module directory to sys.path so that `inference`, `preprocessing`,
# and `postprocessing` can be imported as top-level modules (the way they are
# written — no package prefix).
_AI_DIR = Path(__file__).resolve().parents[2] / "AI-Background-Remover-AI"
if str(_AI_DIR) not in sys.path:
    sys.path.insert(0, str(_AI_DIR))

from inference import run_inference  # noqa: E402


async def remove_background(input_path: str, output_path: str) -> None:
    """
    Asynchronous wrapper around the CPU/GPU-bound inference call.

    Runs `run_inference` in a thread-pool executor so it does not block
    FastAPI's event loop.

    Args:
        input_path:  Absolute or relative path to the source image.
        output_path: Destination path for the transparent PNG result.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_inference, input_path, output_path)
