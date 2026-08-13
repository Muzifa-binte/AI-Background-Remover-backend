import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pathlib import Path

from services.database import connect_db, close_db
from services.quota    import setup_quota_indexes
from services.cleanup  import start_cleanup_task, stop_cleanup_task
from routes.auth        import router as auth_router
from routes.remove_bg   import router as remove_bg_router
from routes.download    import router as download_router
from routes.history     import router as history_router
from routes.images      import router as images_router
from routes.enhance     import router as enhance_router
from routes.replace_bg  import router as replace_bg_router
from routes.smart_crop  import router as smart_crop_router
from routes.batch       import router as batch_router
from routes.chat        import router as chat_router
from routes.image       import router as image_router

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    await setup_quota_indexes()
    start_cleanup_task()
    yield
    stop_cleanup_task()
    await close_db()


app = FastAPI(
    title="AI Background Remover API",
    description="REST API for AI-powered background removal with auth and cloud storage.",
    version="2.0.0",
    lifespan=lifespan,
)

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router,       prefix="/api")
app.include_router(remove_bg_router,  prefix="/api")
app.include_router(download_router,   prefix="/api")
app.include_router(history_router,    prefix="/api")
app.include_router(images_router,     prefix="/api")
app.include_router(enhance_router,    prefix="/api")
app.include_router(replace_bg_router, prefix="/api")
app.include_router(smart_crop_router, prefix="/api")
app.include_router(batch_router,      prefix="/api")
app.include_router(chat_router,       prefix="/api")
app.include_router(image_router,      prefix="/api")


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "AI Background Remover API v2 is running."}
