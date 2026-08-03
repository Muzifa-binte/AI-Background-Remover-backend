"""
MongoDB connection and helper utilities powered by Motor (async driver).
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "ai_bg_remover")

client: AsyncIOMotorClient = None  # type: ignore


async def connect_db() -> None:
    """Open the MongoDB connection. Call from FastAPI startup event."""
    global client
    client = AsyncIOMotorClient(MONGO_URI)


async def close_db() -> None:
    """Close the MongoDB connection. Call from FastAPI shutdown event."""
    if client:
        client.close()


def get_collection(name: str):
    """Return a Motor collection by name."""
    return client[DB_NAME][name]
