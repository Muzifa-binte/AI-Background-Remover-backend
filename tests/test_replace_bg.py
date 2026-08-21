"""
Tests for background replacement route:
  POST /api/replace-background

composite_background service is mocked so no image processing runs.
"""
from __future__ import annotations

import io
import httpx
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


MOCK_IMAGE_META = {"width": 1, "height": 1, "mode": "RGBA"}
MOCK_FG_FILE = "test_fg.png"


class TestReplaceBackground:
    async def test_replace_bg_solid_success(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Solid color background replacement succeeds."""
        with patch("os.path.isfile", return_value=True), \
             patch("routes.replace_bg.composite_background", return_value=MOCK_IMAGE_META), \
             patch("routes.replace_bg.save_file", new_callable=AsyncMock,
                   return_value="/api/download/replaced.png"):

            resp = await client.post(
                "/api/replace-background",
                headers=auth_headers,
                data={
                    "fg_filename": MOCK_FG_FILE,
                    "bg_type": "solid",
                    "solid_color": "#00ff00",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "output_filename" in data
        assert "download_url" in data

    async def test_replace_bg_gradient_success(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Gradient background replacement succeeds."""
        with patch("os.path.isfile", return_value=True), \
             patch("routes.replace_bg.composite_background", return_value=MOCK_IMAGE_META), \
             patch("routes.replace_bg.save_file", new_callable=AsyncMock,
                   return_value="/api/download/replaced.png"):

            resp = await client.post(
                "/api/replace-background",
                headers=auth_headers,
                data={
                    "fg_filename": MOCK_FG_FILE,
                    "bg_type": "gradient",
                    "gradient_start": "#ff0000",
                    "gradient_end": "#0000ff",
                    "gradient_dir": "horizontal",
                },
            )

        assert resp.status_code == 200

    async def test_replace_bg_image_mode_no_file(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Image mode without bg_file returns 400."""
        with patch("os.path.isfile", return_value=True):
            resp = await client.post(
                "/api/replace-background",
                headers=auth_headers,
                data={
                    "fg_filename": MOCK_FG_FILE,
                    "bg_type": "image",
                },
            )
        assert resp.status_code == 400

    async def test_replace_bg_fg_not_found(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Missing foreground file returns 404."""
        with patch("os.path.isfile", return_value=False):
            resp = await client.post(
                "/api/replace-background",
                headers=auth_headers,
                data={
                    "fg_filename": "nonexistent.png",
                    "bg_type": "solid",
                },
            )
        assert resp.status_code == 404

    async def test_replace_bg_invalid_mode(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Invalid bg_type returns 400."""
        resp = await client.post(
            "/api/replace-background",
            headers=auth_headers,
            data={
                "fg_filename": MOCK_FG_FILE,
                "bg_type": "laser_show",
            },
        )
        assert resp.status_code == 400

    async def test_replace_bg_invalid_gradient_dir(
        self, client: httpx.AsyncClient, auth_headers: dict
    ):
        """Invalid gradient_dir returns 400."""
        resp = await client.post(
            "/api/replace-background",
            headers=auth_headers,
            data={
                "fg_filename": MOCK_FG_FILE,
                "bg_type": "gradient",
                "gradient_dir": "circular",
            },
        )
        assert resp.status_code == 400

    async def test_replace_bg_unauthenticated(self, client: httpx.AsyncClient):
        """Unauthenticated request returns 401."""
        resp = await client.post(
            "/api/replace-background",
            data={
                "fg_filename": MOCK_FG_FILE,
                "bg_type": "solid",
            },
        )
        assert resp.status_code == 401

