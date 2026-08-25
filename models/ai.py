from pydantic import BaseModel
from typing import Optional


class ImageAnalysisResponse(BaseModel):
    subject: str
    image_type: str
    background_description: str
    suggested_use: str
    editing_recommendations: list[str]


class CaptionResponse(BaseModel):
    caption: str
    style: str


class CaptionsResponse(BaseModel):
    captions: list[str]
    style: str


class BackgroundSuggestionsResponse(BaseModel):
    suggestions: list[str]


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    thinking: Optional[str] = None
    action: Optional[dict] = None


class DetectedObject(BaseModel):
    label: str
    box_2d: list[int]
    confidence: float


class ColorPaletteItem(BaseModel):
    hex: str
    name: str
    percentage: int
    text_color: str
    use_case: str


class StyleTransferRecommendation(BaseModel):
    style: str
    description: str
    prompts: str


class CompositionAnalysis(BaseModel):
    rule_of_thirds: str
    leading_lines: str
    balance: str
    crop_recommendation: str


class OptimalEnhancementSettings(BaseModel):
    brightness: float
    contrast: float
    saturation: float
    sharpness: float
    denoise: bool
    auto_wb: bool
    denoise_strength: int


class SuggestedCropSettings(BaseModel):
    aspect_ratio: str
    padding_pct: float


class AdvancedAnalysisResponse(BaseModel):
    object_detection: list[DetectedObject]
    color_palette: list[ColorPaletteItem]
    style_transfer: list[StyleTransferRecommendation]
    composition: CompositionAnalysis
    suggested_backgrounds: list[str]
    optimal_enhancement: OptimalEnhancementSettings
    suggested_crop: SuggestedCropSettings
    suggested_filename: str


class BatchAdvancedAnalysisItem(BaseModel):
    filename: str
    status: str
    analysis: Optional[AdvancedAnalysisResponse] = None
    error: Optional[str] = None


class BatchAdvancedAnalysisResponse(BaseModel):
    results: list[BatchAdvancedAnalysisItem]
