"""
Tests for smart crop route:
  POST /api/smart-crop

remove_background and smart_crop services are mocked so no AI/CV runs.
"""
from __future__ import annotations

import io
import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


MOCK_CROP_META = {
    "mode": "subject_crop", "x": 10, "y": 10, "w": 80, "h": 80,
    "canvas_w": 100, "canvas_h": 100,
}


class TestSmartCrop:
    async def test_smart_crop_success(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """Valid smart-crop request returns removed_url and cropped_url."""
        mock_file_ctx = AsyncMock()
        mock_file_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_file_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_aiofiles_open = MagicMock(return_value=mock_file_ctx)
        with patch("routes.smart_crop.remove_background", new_callable=AsyncMock), \
             patch("routes.smart_crop.smart_crop", return_value=MOCK_CROP_META), \
             patch("routes.smart_crop.save_file", new_callable=AsyncMock,
                   return_value="/api/download/test.png"), \
             patch("aiofiles.open", mock_aiofiles_open):

            resp = await client.post(
                "/api/smart-crop",
                headers=auth_headers,
                files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
                data={
                    "aspect_ratio": "1:1",
                    "padding_pct": "0.05",
                    "quality": "fast",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "cropped_url" in data or "output_filename" in data or "crop_meta" in data

    async def test_smart_crop_invalid_aspect_ratio(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """Invalid aspect ratio string returns 400."""
        resp = await client.post(
            "/api/smart-crop",
            headers=auth_headers,
            files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            data={"aspect_ratio": "99:99"},
        )
        assert resp.status_code == 400

    async def test_smart_crop_unsupported_file_type(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Unsupported file type returns 400."""
        resp = await client.post(
            "/api/smart-crop",
            headers=auth_headers,
            files={"file": ("anim.gif", io.BytesIO(b"GIF89a"), "image/gif")},
            data={"aspect_ratio": "1:1"},
        )
        assert resp.status_code == 400

    async def test_smart_crop_oversized_file(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Files over 10 MB return 413."""
        big_png = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)
        resp = await client.post(
            "/api/smart-crop",
            headers=auth_headers,
            files={"file": ("huge.png", io.BytesIO(big_png), "image/png")},
            data={"aspect_ratio": "1:1"},
        )
        assert resp.status_code == 413

    async def test_smart_crop_unauthenticated(
        self, client: httpx.AsyncClient, png_file: bytes
    ):
        """Unauthenticated request returns 401."""
        resp = await client.post(
            "/api/smart-crop",
            files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            data={"aspect_ratio": "1:1"},
        )
        assert resp.status_code == 401

