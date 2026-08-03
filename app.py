from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from services.database import connect_db, close_db
from routes.remove_bg import router as remove_bg_router
from routes.download   import router as download_router
from routes.history    import router as history_router
from routes.images     import router as images_router

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open DB connection on startup, close on shutdown."""
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="AI Background Remover API",
    description="REST API for AI-powered background removal using deep learning segmentation models.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the React frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Route registrations
# ---------------------------------------------------------------------------
app.include_router(remove_bg_router, prefix="/api")
app.include_router(download_router,  prefix="/api")
app.include_router(history_router,   prefix="/api")
app.include_router(images_router,    prefix="/api")


@app.get("/", tags=["Health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "AI Background Remover API is running."}
