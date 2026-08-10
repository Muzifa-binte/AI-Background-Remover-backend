"""
MongoDB connection and helper utilities powered by Motor (async driver).
"""

import os
from urllib.parse import quote_plus
from motor.motor_asyncio import AsyncIOMotorClient

client: AsyncIOMotorClient = None  # type: ignore


async def connect_db() -> None:
    """Open the MongoDB connection. Call from FastAPI startup event."""
    global client
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,   # fail fast instead of hanging 30s
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
    )
    # Eagerly verify the connection so startup fails loudly if Atlas is down
    try:
        await client.admin.command("ping")
        print("✅ MongoDB connected successfully.")
    except Exception as exc:
        print(f"⚠️  MongoDB connection failed: {exc}")
        print("   Registration and login will not work until the database is reachable.")
        print("   Check: Atlas IP whitelist, cluster is not paused, and network access.")


def get_db_name() -> str:
    return os.getenv("MONGO_DB_NAME", "ai_bg_remover")


async def close_db() -> None:
    """Close the MongoDB connection. Call from FastAPI shutdown event."""
    if client:
        client.close()


def get_collection(name: str):
    """Return a Motor collection by name."""
    return client[get_db_name()][name]
