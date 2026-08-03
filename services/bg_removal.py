"""
Background removal service.

Bridges the FastAPI route layer with the AI inference pipeline.
"""

import asyncio
from ai.inference import run_inference


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
