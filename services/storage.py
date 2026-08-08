"""
Storage Service — local-disk and S3/R2 adapter.

Controlled by the STORAGE_BACKEND environment variable:
  "local" (default) — write files to ./output/  (same as before auth)
  "s3"              — upload to an S3-compatible bucket (AWS S3, Cloudflare R2, MinIO)

Public API (used by every route that saves a processed image):
──────────────────────────────────────────────────────────────
  await save_file(local_path, filename)  → storage_url: str
      Persists the file and returns a URL the client can use to fetch it.

  get_download_url(filename)             → str
      Resolves a filename to its download URL without uploading.
      For local storage this is the /api/download/{filename} endpoint.
      For S3 this is the public S3/CDN URL.

Routes continue to write temp files locally (for AI processing), then call
save_file() once processing is done. For local storage this is a no-op
(file is already in output/). For S3 the file is uploaded then the local
copy can optionally be removed.
"""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower().strip()
OUTPUT_DIR      = "output"


# ── Local adapter ──────────────────────────────────────────────────────────

def _local_url(filename: str) -> str:
    """Return the API download URL for a locally-stored file."""
    return f"/api/download/{filename}"


async def _save_local(local_path: str, filename: str) -> str:
    """
    File is already on disk in output/. Just return its URL.
    If local_path is not inside output/, copy it there.
    """
    dest = os.path.join(OUTPUT_DIR, filename)
    if os.path.abspath(local_path) != os.path.abspath(dest):
        import shutil
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copy2(local_path, dest)
    return _local_url(filename)


# ── S3 / R2 adapter ────────────────────────────────────────────────────────

S3_BUCKET       = os.getenv("S3_BUCKET",       "")
S3_REGION       = os.getenv("S3_REGION",       "us-east-1")
S3_ACCESS_KEY   = os.getenv("S3_ACCESS_KEY",   "")
S3_SECRET_KEY   = os.getenv("S3_SECRET_KEY",   "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")   # blank = AWS; set for R2/MinIO
S3_PUBLIC_URL   = os.getenv("S3_PUBLIC_URL",   "")   # optional CDN prefix


def _s3_public_url(filename: str) -> str:
    """Build the public URL for a file stored in S3."""
    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL.rstrip('/')}/{filename}"
    if S3_ENDPOINT_URL:
        # R2-style: endpoint/bucket/filename
        return f"{S3_ENDPOINT_URL.rstrip('/')}/{S3_BUCKET}/{filename}"
    # Standard AWS
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"


async def _save_s3(local_path: str, filename: str) -> str:
    """
    Upload local_path to S3 under the key `filename`.
    Returns the public URL.

    boto3 is imported lazily so the app starts fine even when
    STORAGE_BACKEND=local (boto3 not required in that case).
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        raise RuntimeError(
            "boto3 is not installed. Run: pip install boto3  "
            "or switch STORAGE_BACKEND=local."
        )

    kwargs: dict = {
        "aws_access_key_id":     S3_ACCESS_KEY,
        "aws_secret_access_key": S3_SECRET_KEY,
        "region_name":           S3_REGION,
    }
    if S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = S3_ENDPOINT_URL

    import asyncio
    loop = asyncio.get_event_loop()

    def _upload() -> None:
        s3 = boto3.client("s3", **kwargs)
        with open(local_path, "rb") as fh:
            s3.upload_fileobj(
                fh,
                S3_BUCKET,
                filename,
                ExtraArgs={"ContentType": "image/png"},
            )

    await loop.run_in_executor(None, _upload)
    return _s3_public_url(filename)


# ── Public interface ───────────────────────────────────────────────────────

async def save_file(local_path: str, filename: str) -> str:
    """
    Persist a processed file to the configured storage backend.

    Args:
        local_path: Absolute or relative path to the local temp file.
        filename:   Final filename to store (e.g. "uuid_result.png").

    Returns:
        The URL clients use to access the file.
    """
    if STORAGE_BACKEND == "s3":
        return await _save_s3(local_path, filename)
    return await _save_local(local_path, filename)


def get_download_url(filename: str) -> str:
    """
    Resolve a filename to its download URL without uploading.
    Use this when the file is already stored.
    """
    if STORAGE_BACKEND == "s3":
        return _s3_public_url(filename)
    return _local_url(filename)
