"""
Tests for the Magic Recolor route:
  POST /api/recolor

recolor_region service is mocked so no image processing runs.
"""
from __future__ import annotations

import io
import httpx
import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_io():
    from tests.conftest import MINIMAL_PNG
    return io.BytesIO(MINIMAL_PNG)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRecolor:
    async def test_recolor_success(self, client: httpx.AsyncClient, auth_headers: dict):
        """Valid recolor request returns output_filename and download_url."""
        from tests.conftest import MINIMAL_PNG

        with patch(
            "services.recolor.recolor_region",
            return_value=MINIMAL_PNG,
        ), patch(
            "services.storage.save_file",
            return_value="/api/download/recolored.png",
        ):
            resp = await client.post(
                "/api/recolor",
                headers=auth_headers,
                files={
                    "image": ("photo.png", _png_io(), "image/png"),
                    "mask":  ("mask.png",  _png_io(), "image/png"),
                },
                data={
                    "target_color": "#e83c6d",
                    "strength":     "1.0",
                    "feather":      "15",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "output_filename" in data
        assert "download_url" in data

    async def test_recolor_invalid_hex_color(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Invalid hex color returns 400."""
        with patch("services.recolor.recolor_region", return_value=b"fake"):
            resp = await client.post(
                "/api/recolor",
                headers=auth_headers,
                files={
                    "image": ("photo.png", _png_io(), "image/png"),
                    "mask":  ("mask.png",  _png_io(), "image/png"),
                },
                data={"target_color": "notacolor"},
            )
        assert resp.status_code == 400

    async def test_recolor_missing_mask(self, client: httpx.AsyncClient, auth_headers: dict):
        """Missing mask field returns 422."""
        resp = await client.post(
            "/api/recolor",
            headers=auth_headers,
            files={"image": ("photo.png", _png_io(), "image/png")},
            data={"target_color": "#ff0000"},
        )
        assert resp.status_code == 422

    async def test_recolor_unsupported_image_type(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """GIF source image is rejected with 400."""
        resp = await client.post(
            "/api/recolor",
            headers=auth_headers,
            files={
                "image": ("anim.gif", io.BytesIO(b"GIF89a"), "image/gif"),
                "mask":  ("mask.png", _png_io(), "image/png"),
            },
            data={"target_color": "#ff0000"},
        )
        assert resp.status_code == 400

    async def test_recolor_oversized_image(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Images over 10 MB return 413."""
        big = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)
        resp = await client.post(
            "/api/recolor",
            headers=auth_headers,
            files={
                "image": ("huge.png", io.BytesIO(big), "image/png"),
                "mask":  ("mask.png", _png_io(), "image/png"),
            },
            data={"target_color": "#ff0000"},
        )
        assert resp.status_code == 413

    async def test_recolor_unauthenticated(self, client: httpx.AsyncClient):
        """Request without token returns 401."""
        resp = await client.post(
            "/api/recolor",
            files={
                "image": ("photo.png", _png_io(), "image/png"),
                "mask":  ("mask.png",  _png_io(), "image/png"),
            },
            data={"target_color": "#ff0000"},
        )
        assert resp.status_code == 401

    async def test_recolor_strength_clamped(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Strength values outside 0.0–1.0 are clamped (not rejected) per route."""
        from tests.conftest import MINIMAL_PNG
        with patch("services.recolor.recolor_region", return_value=MINIMAL_PNG), \
             patch("services.storage.save_file", return_value="/api/download/x.png"):
            # Out-of-range strength should be accepted (clamped by backend)
            resp = await client.post(
                "/api/recolor",
                headers=auth_headers,
                files={
                    "image": ("photo.png", _png_io(), "image/png"),
                    "mask":  ("mask.png",  _png_io(), "image/png"),
                },
                data={"target_color": "#aabbcc", "strength": "5.0"},
            )
        assert resp.status_code == 200
