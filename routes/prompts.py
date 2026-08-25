import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from services.database import get_collection
from services.auth import get_current_user
from models.user import UserOut
from models.collaboration import PromptTemplateCreate, PromptTemplateOut

router = APIRouter(prefix="/prompts", tags=["Prompt Templates"])


@router.post("", response_model=PromptTemplateOut)
async def create_prompt_template(
    payload: PromptTemplateCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("prompt_templates")
        doc = {
            "template_id": str(uuid.uuid4()),
            "user_id": current_user.user_id,
            "title": payload.title,
            "prompt_text": payload.prompt_text,
            "tags": payload.tags,
            "use_count": 0,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return PromptTemplateOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save prompt template: {exc}")


@router.get("", response_model=List[PromptTemplateOut])
async def list_prompt_templates(current_user: UserOut = Depends(get_current_user)):
    try:
        collection = get_collection("prompt_templates")
        cursor = (
            collection
            .find({"user_id": current_user.user_id}, {"_id": 0})
            .sort("created_at", -1)
        )
        results = await cursor.to_list(length=200)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch prompt templates: {exc}")


@router.post("/{template_id}/use", response_model=PromptTemplateOut)
async def use_prompt_template(
    template_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("prompt_templates")
        doc = await collection.find_one_and_update(
            {"template_id": template_id, "user_id": current_user.user_id},
            {"$inc": {"use_count": 1}},
            return_document=True,
        )
        if doc is None:
            raise HTTPException(status_code=404, detail="Prompt template not found.")
        doc.pop("_id", None)
        return PromptTemplateOut(**doc)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not update prompt template: {exc}")


@router.delete("/{template_id}")
async def delete_prompt_template(
    template_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("prompt_templates")
        result = await collection.delete_one(
            {"template_id": template_id, "user_id": current_user.user_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Prompt template not found.")
        return {"status": "deleted", "template_id": template_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete prompt template: {exc}")
