"""Remove test accounts created during diagnostics."""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv(".env")

KEEP_EMAIL = "binteaurangzaib2007@gmail.com"

async def main():
    from services.database import connect_db, close_db, get_collection
    await connect_db()
    users = get_collection("users")
    result = await users.delete_many({"email": {"$ne": KEEP_EMAIL}})
    print(f"Removed {result.deleted_count} test account(s)")
    remaining = await users.find({}, {"_id": 0, "email": 1, "name": 1}).to_list(10)
    print("Remaining users:", [(u["name"], u["email"]) for u in remaining])
    await close_db()

asyncio.run(main())
