"""One-time DB initialisation. Run once, then never again."""
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
client = create_client(url, key)

sql = (Path(__file__).parent.parent / "schema.sql").read_text()
# Note: Supabase Python client doesn't support raw SQL exec directly.
# Paste schema.sql into Supabase SQL Editor → Run.
# This script is here for documentation and future migration tooling.
print("Schema SQL ready. Paste into Supabase SQL Editor to execute.")
print(f"Connected to: {url}")
