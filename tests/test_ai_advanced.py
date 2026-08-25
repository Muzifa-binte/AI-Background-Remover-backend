import io
import httpx
import pytest
from unittest.mock import patch, AsyncMock

# Define a mock response matching our AdvancedAnalysisResponse structure
MOCK_ADVANCED_ANALYSIS = {
    "object_detection": [
        {"label": "Person", "box_2d": [10, 20, 80, 70], "confidence": 0.95}
    ],
    "color_palette": [
        {"hex": "#3A5F43", "name": "Forest Green", "percentage": 45, "text_color": "#ffffff", "use_case": "Primary color"}
    ],
    "style_transfer": [
        {"style": "Minimalist Studio", "description": "Soft lighting", "prompts": "Studio portrait"}
    ],
    "composition": {
        "rule_of_thirds": "Aligned right",
        "leading_lines": "Collar lines",
        "balance": "Asymmetric",
        "crop_recommendation": "Crop 5%"
    },
    "suggested_backgrounds": [
        "#FFFFFF", "Soft White Studio"
    ],
    "optimal_enhancement": {
        "brightness": 1.1,
        "contrast": 1.0,
        "saturation": 1.0,
        "sharpness": 1.1,
        "denoise": False,
        "auto_wb": True,
        "denoise_strength": 9
    },
    "suggested_crop": {
        "aspect_ratio": "1:1",
        "padding_pct": 0.05
    },
    "suggested_filename": "clean_portrait"
}

class TestAdvancedAIAnalysis:
    async def test_analyze_advanced_success(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        with patch(
            "routes.image.ai_service.analyze_image_advanced",
            new_callable=AsyncMock,
            return_value=MOCK_ADVANCED_ANALYSIS,
        ):
            resp = await client.post(
                "/api/image/analyze-advanced",
                headers=auth_headers,
                files={"file": ("photo.png", io.BytesIO(png_file), "image/png")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "object_detection" in data
        assert "color_palette" in data
        assert "style_transfer" in data
        assert "composition" in data
        assert data["object_detection"][0]["label"] == "Person"
        assert data["color_palette"][0]["hex"] == "#3A5F43"

    async def test_analyze_advanced_batch_success(
        self, client: httpx.AsyncClient, auth_headers: dict, png_file: bytes
    ):
        with patch(
            "routes.image.ai_service.analyze_image_advanced",
            new_callable=AsyncMock,
            return_value=MOCK_ADVANCED_ANALYSIS,
        ):
            resp = await client.post(
                "/api/image/analyze-advanced-batch",
                headers=auth_headers,
                files=[
                    ("files", ("photo1.png", io.BytesIO(png_file), "image/png")),
                    ("files", ("photo2.png", io.BytesIO(png_file), "image/png")),
                ],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["filename"] == "photo1.png"
        assert data["results"][0]["status"] == "success"
        assert data["results"][0]["analysis"]["object_detection"][0]["label"] == "Person"
