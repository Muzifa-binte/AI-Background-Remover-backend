from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from typing import Optional
from models.ai import ChatResponse
from services.ai_service import AIService
from services.image_service import ImageService
from services.auth import get_current_user
from services.tracking import track_usage, track_cost
from models.user import UserOut

router = APIRouter(prefix="/chat", tags=["chat"])
ai_service = AIService()
image_service = ImageService()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token) when the provider doesn't return exact counts."""
    return max(1, len(text) // 4)


@router.post("", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    current_user: UserOut = Depends(get_current_user)
):
    try:
        image_bytes = None
        if file:
            raw_bytes = await file.read()

            if not image_service.validate(raw_bytes):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image format. Supported formats are: JPEG, PNG, WEBP."
                )

            image_bytes = image_service.preprocess(raw_bytes)

        reply, thinking, action = await ai_service.chat(message, image_bytes)

        # ── Analytics & Insights: usage + cost tracking (best-effort) ───────
        await track_usage(
            user_id=current_user.user_id,
            feature="chat",
            metadata={"has_image": bool(image_bytes)},
        )
        input_tokens = _estimate_tokens(message)
        output_tokens = _estimate_tokens(reply)
        await track_cost(
            user_id=current_user.user_id,
            feature="chat",
            provider=ai_service.provider,
            model=ai_service.vision_model if image_bytes else ai_service.chat_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ChatResponse(reply=reply, thinking=thinking, action=action)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
