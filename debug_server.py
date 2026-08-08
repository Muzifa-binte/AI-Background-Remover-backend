"""Run with: python debug_server.py  — captures the exact 500 traceback"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(".env")

import traceback
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from services.database import connect_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)

# Mount only the auth router
from routes.auth import router as auth_router
app.include_router(auth_router, prefix="/api")

# Global exception handler — prints full traceback
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        tb = traceback.format_exc()
        print("\n========= 500 TRACEBACK =========")
        print(tb)
        print("=================================\n")
        return JSONResponse(status_code=500, content={"detail": str(e), "traceback": tb})

if __name__ == "__main__":
    uvicorn.run("debug_server:app", host="127.0.0.1", port=8001, reload=False)
