"""Quick diagnostics — run with: python diag.py"""
import sys, os, time, asyncio
sys.path.insert(0, '.')

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path('.env'))

print("=== ENV ===")
print("SECRET_KEY present:", bool(os.getenv("SECRET_KEY")))
print("MONGO_URI present: ", bool(os.getenv("MONGO_URI")))

print("\n=== IMPORTS ===")
from models.user import UserCreate, TokenResponse
print("models.user: OK")
from services.auth import hash_password, create_access_token, verify_password
print("services.auth: OK")

print("\n=== BCRYPT TIMING ===")
start = time.time()
h = hash_password("testpass123")
elapsed = time.time() - start
print(f"hash_password() took {elapsed:.2f}s  {'<-- SLOW! bcrypt rounds too high' if elapsed > 3 else 'OK'}")
ok = verify_password("testpass123", h)
print(f"verify_password(): {ok}")

print("\n=== MONGODB ===")
from motor.motor_asyncio import AsyncIOMotorClient

async def test_mongo():
    uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=8000)
    try:
        await client.admin.command("ping")
        print("Atlas connection: OK")
        db = client[os.getenv("MONGO_DB_NAME", "ai_bg_remover")]
        cols = await db.list_collection_names()
        print("Collections:", cols or "(empty — first run)")
    except Exception as e:
        print(f"Atlas connection: FAILED\n  {e}")
    finally:
        client.close()

asyncio.run(test_mongo())

print("\n=== JWT ===")
token = create_access_token({"sub": "test-id", "email": "test@test.com"})
print("create_access_token(): OK, length =", len(token))

print("\nAll diagnostics done.")
