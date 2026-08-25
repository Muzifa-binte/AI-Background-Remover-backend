import io
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse

from services.database import get_collection
from services.auth import get_current_user
from models.user import UserOut
from models.collaboration import (
    ChatMessageIn,
    ChatMessageOut,
    CommentCreate,
    CommentOut,
)

router = APIRouter(tags=["Collaboration & Sharing"])


@router.post("/collab/{image_id}/comments", response_model=CommentOut)
async def add_comment(
    image_id: str,
    payload: CommentCreate,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("image_comments")
        doc = {
            "comment_id": str(uuid.uuid4()),
            "image_id": image_id,
            "user_id": current_user.user_id,
            "user_name": current_user.name,
            "text": payload.text,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return CommentOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not add comment: {exc}")


@router.get("/collab/{image_id}/comments", response_model=List[CommentOut])
async def list_comments(image_id: str, current_user: UserOut = Depends(get_current_user)):
    try:
        collection = get_collection("image_comments")
        cursor = collection.find({"image_id": image_id}, {"_id": 0}).sort("created_at", 1)
        results = await cursor.to_list(length=500)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch comments: {exc}")


@router.post("/chat-history", response_model=ChatMessageOut)
async def save_chat_message(
    payload: ChatMessageIn,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("chat_messages")
        doc = {
            "message_id": str(uuid.uuid4()),
            "conversation_id": payload.conversation_id,
            "user_id": current_user.user_id,
            "role": payload.role,
            "content": payload.content,
            "image_id": payload.image_id,
            "created_at": datetime.now(timezone.utc),
        }
        await collection.insert_one(doc)
        doc.pop("_id", None)
        return ChatMessageOut(**doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save chat message: {exc}")


@router.get("/chat-history/{conversation_id}", response_model=List[ChatMessageOut])
async def get_chat_history(
    conversation_id: str,
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("chat_messages")
        cursor = (
            collection
            .find(
                {"conversation_id": conversation_id, "user_id": current_user.user_id},
                {"_id": 0},
            )
            .sort("created_at", 1)
        )
        results = await cursor.to_list(length=1000)
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not fetch chat history: {exc}")


@router.get("/export/{conversation_id}")
async def export_conversation(
    conversation_id: str,
    format: str = Query("text", pattern="^(text|pdf)$"),
    current_user: UserOut = Depends(get_current_user),
):
    try:
        collection = get_collection("chat_messages")
        cursor = (
            collection
            .find(
                {"conversation_id": conversation_id, "user_id": current_user.user_id},
                {"_id": 0},
            )
            .sort("created_at", 1)
        )
        messages = await cursor.to_list(length=1000)
        if not messages:
            raise HTTPException(status_code=404, detail="Conversation not found or empty.")

        if format == "text":
            lines = []
            for m in messages:
                ts = m["created_at"].strftime("%Y-%m-%d %H:%M")
                lines.append(f"[{ts}] {m['role'].upper()}: {m['content']}")
            text_content = "\n\n".join(lines)
            return PlainTextResponse(
                content=text_content,
                headers={
                    "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.txt"'
                },
            )

        pdf_bytes = _build_pdf(messages, conversation_id)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="conversation_{conversation_id}.pdf"'
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not export conversation: {exc}")


def _build_pdf(messages: list, conversation_id: str) -> bytes:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    meta_style = ParagraphStyle("meta", parent=styles["Normal"], textColor="#666666", fontSize=9)
    role_style = ParagraphStyle("role", parent=styles["Normal"], fontName="Helvetica-Bold", spaceBefore=10)
    body_style = ParagraphStyle("body", parent=styles["Normal"], spaceAfter=4)

    story = [
        Paragraph("AI Background Remover - Conversation Export", title_style),
        Paragraph(f"Conversation ID: {escape(conversation_id)}", meta_style),
        Spacer(1, 0.25 * inch),
    ]

    for m in messages:
        ts = m["created_at"].strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(f"{escape(m['role'].upper())} - {ts}", role_style))
        story.append(Paragraph(escape(m["content"]).replace("\n", "<br/>"), body_style))

    doc.build(story)
    return buffer.getvalue()
