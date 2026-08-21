"""
Tests for background removal route:
  POST /api/remove-background

AI inference (remove_background_bytes) is mocked to return
a minimal valid PNG without loading any model weights.
"""
from __future__ import annotations

import io
import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _png_file(name: str = "test.png"):
    """Return (filename, bytes, content_type) for a multipart upload."""
    from tests.conftest import MINIMAL_PNG
    return (name, io.BytesIO(MINIMAL_PNG), "image/png")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRemoveBackground:
    async def test_remove_bg_success(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """Successful removal returns output_filename and download_url."""
        from tests.conftest import MINIMAL_PNG
        mock_result = MINIMAL_PNG  # return same 1×1 PNG as mocked AI output

        with patch(
            "routes.remove_bg.remove_background_bytes",
            new_callable=AsyncMock,
            return_value=mock_result,
        ), patch("services.storage.save_file", new_callable=AsyncMock, return_value="/api/download/fake.png"):

            resp = await client.post(
                "/api/remove-background",
                headers=auth_headers,
                files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
                data={"quality": "fast"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "output_filename" in data
        assert "download_url" in data
        assert data["quality"] == "fast"

    async def test_remove_bg_all_quality_modes(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """All three quality modes are accepted."""
        from tests.conftest import MINIMAL_PNG

        for quality in ("fast", "standard", "quality"):
            with patch(
                "routes.remove_bg.remove_background_bytes",
                new_callable=AsyncMock,
                return_value=MINIMAL_PNG,
            ), patch("services.storage.save_file", new_callable=AsyncMock, return_value=f"/api/download/{quality}.png"):

                resp = await client.post(
                    "/api/remove-background",
                    headers=auth_headers,
                    files={"file": ("img.png", io.BytesIO(png_file), "image/png")},
                    data={"quality": quality},
                )
            assert resp.status_code == 200, f"Quality '{quality}' failed: {resp.text}"

    async def test_remove_bg_invalid_quality(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """Unknown quality string returns 400."""
        resp = await client.post(
            "/api/remove-background",
            headers=auth_headers,
            files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            data={"quality": "ultra"},
        )
        assert resp.status_code == 400

    async def test_remove_bg_unsupported_type(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """GIF files should be rejected with 400."""
        resp = await client.post(
            "/api/remove-background",
            headers=auth_headers,
            files={"file": ("anim.gif", io.BytesIO(b"GIF89a"), "image/gif")},
            data={"quality": "fast"},
        )
        assert resp.status_code == 400

    async def test_remove_bg_oversized_file(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Files over 10 MB return 413."""
        big_png = b"\x89PNG" + b"\x00" * (11 * 1024 * 1024)
        resp = await client.post(
            "/api/remove-background",
            headers=auth_headers,
            files={"file": ("huge.png", io.BytesIO(big_png), "image/png")},
            data={"quality": "fast"},
        )
        assert resp.status_code == 413

    async def test_remove_bg_unauthenticated(self, client: httpx.AsyncClient, png_file: bytes):
        """Requests without a token return 401."""
        resp = await client.post(
            "/api/remove-background",
            files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            data={"quality": "fast"},
        )
        assert resp.status_code == 401

    async def test_remove_bg_inference_error(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        """AI inference failures return 500."""
        with patch(
            "routes.remove_bg.remove_background_bytes",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Model crashed"),
        ):
            resp = await client.post(
                "/api/remove-background",
                headers=auth_headers,
                files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
                data={"quality": "fast"},
            )
        assert resp.status_code == 500
