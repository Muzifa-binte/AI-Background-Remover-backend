"""
Backfill script — run once to tag old history records with the owner's user_id.
Usage:  python backfill.py
"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))
from dotenv import load_dotenv; load_dotenv(".env")

OWNER_EMAIL = "binteaurangzaib2007@gmail.com"   # <-- your registered email

async def main():
    from services.database import connect_db, close_db, get_collection
    await connect_db()

    # Find your user
    users  = get_collection("users")
    user   = await users.find_one({"email": OWNER_EMAIL})
    if not user:
        print(f"No user found with email {OWNER_EMAIL!r}")
        await close_db()
        return

    user_id = user["user_id"]
    print(f"Found user: {user['name']}  user_id={user_id}")

    # Backfill all history records that have no user_id
    history = get_collection("history")
    result  = await history.update_many(
        {"user_id": {"$exists": False}},
        {"$set": {"user_id": user_id}},
    )
    print(f"Backfilled {result.modified_count} history record(s)")

    # Verify
    records = await history.find(
        {"user_id": user_id}, {"_id": 0, "upload_id": 1, "output_filename": 1}
    ).to_list(50)
    print(f"Records now visible to you ({len(records)}):")
    for r in records:
        print(f"  {r['upload_id'][:8]}  {r.get('output_filename','?')}")

    await close_db()

asyncio.run(main())
