"""
Tests for image enhancement route:
  POST /api/enhance

enhance_image service is mocked so no heavy image processing runs.
"""
from __future__ import annotations

import io
import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

MOCK_META = {"width": 1, "height": 1, "mode": "RGBA"}


class TestEnhance:
    async def test_enhance_success(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """Valid enhancement request returns output_filename and download_url."""
        mock_file_ctx = AsyncMock()
        mock_file_ctx.__aenter__ = AsyncMock(return_value=AsyncMock())
        mock_file_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_aiofiles_open = MagicMock(return_value=mock_file_ctx)
        with patch("routes.enhance.enhance_image", return_value=MOCK_META), \
             patch("routes.enhance.save_file", return_value="/api/download/enhanced.png"), \
             patch("aiofiles.open", mock_aiofiles_open):

            resp = await client.post(
                "/api/enhance",
                headers=auth_headers,
                files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
                data={
                    "brightness": "1.2",
                    "contrast": "1.1",
                    "saturation": "1.3",
                    "sharpness": "1.5",
                    "denoise": "false",
                    "auto_wb": "false",
                    "denoise_strength": "10",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "output_filename" in data
        assert "download_url" in data

    async def test_enhance_unsupported_file_type(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """GIF file type is rejected with 400."""
        resp = await client.post(
            "/api/enhance",
            headers=auth_headers,
            files={"file": ("test.gif", io.BytesIO(b"GIF89a"), "image/gif")},
            data={"brightness": "1.0"},
        )
        assert resp.status_code == 400

    async def test_enhance_oversized_file(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Files over 10 MB return 413."""
        big_png = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)
        resp = await client.post(
            "/api/enhance",
            headers=auth_headers,
            files={"file": ("huge.png", io.BytesIO(big_png), "image/png")},
            data={"brightness": "1.0"},
        )
        assert resp.status_code == 413

    async def test_enhance_unauthenticated(
        self, client: httpx.AsyncClient, png_file: bytes
    ):
        """Unauthenticated request returns 401."""
        resp = await client.post(
            "/api/enhance",
            files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            data={"brightness": "1.0"},
        )
        assert resp.status_code == 401

