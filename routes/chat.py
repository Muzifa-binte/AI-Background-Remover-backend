from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form, Depends
from typing import Optional
from models.ai import ChatResponse
from services.ai_service import AIService
from services.image_service import ImageService
from services.auth import get_current_user
from models.user import UserOut

router = APIRouter(prefix="/chat", tags=["chat"])
ai_service = AIService()
image_service = ImageService()


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

        reply, thinking = await ai_service.chat(message, image_bytes)
        return ChatResponse(reply=reply, thinking=thinking)
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
