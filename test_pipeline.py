"""
Smoke test: inject fake next-week fixtures into weekly_schedule,
then verify the inference engine reads them correctly.
Cleans up after itself.
"""
import os, sys
os.environ.setdefault("APEX_ENV", "production")

from datetime import date, timedelta
from apex10.db import get_client

db = get_client()

# 1. Build 5 fake fixtures dated 5 days from now (guaranteed "upcoming")
future_date = (date.today() + timedelta(days=5)).isoformat()

test_fixtures = [
    {"fixture_id": 990001, "league": "Premier League", "match_date": future_date, "match_time": "15:00", "home_team": "Arsenal", "away_team": "Chelsea", "round": 99, "status": "TEST"},
    {"fixture_id": 990002, "league": "La Liga", "match_date": future_date, "match_time": "18:00", "home_team": "Barcelona", "away_team": "Real Madrid", "round": 99, "status": "TEST"},
    {"fixture_id": 990003, "league": "Bundesliga", "match_date": future_date, "match_time": "15:30", "home_team": "Bayern Munich", "away_team": "Dortmund", "round": 99, "status": "TEST"},
    {"fixture_id": 990004, "league": "Serie A", "match_date": future_date, "match_time": "20:45", "home_team": "Juventus", "away_team": "Inter", "round": 99, "status": "TEST"},
    {"fixture_id": 990005, "league": "Ligue 1", "match_date": future_date, "match_time": "21:00", "home_team": "Paris Saint Germain", "away_team": "Lyon", "round": 99, "status": "TEST"},
]

print("=" * 60)
print("STEP 1: Injecting 5 test fixtures into weekly_schedule...")
print("=" * 60)

try:
    db.table("weekly_schedule").upsert(test_fixtures, on_conflict="fixture_id").execute()
    print(f"✅ Injected {len(test_fixtures)} test fixtures for {future_date}")
except Exception as e:
    print(f"❌ Failed to inject: {e}")
    sys.exit(1)

# 2. Verify fetch_upcoming_fixtures reads them
print("\n" + "=" * 60)
print("STEP 2: Testing fetch_upcoming_fixtures()...")
print("=" * 60)

from apex10.live.inference import fetch_upcoming_fixtures
fixtures = fetch_upcoming_fixtures()

test_ids = {f["fixture_id"] for f in test_fixtures}
found = [f for f in fixtures if f["id"] in test_ids]

if len(found) == 5:
    print(f"✅ fetch_upcoming_fixtures found all 5 test fixtures!")
    for f in found:
        print(f"   {f['league']}: {f['home_team']} vs {f['away_team']} ({f['match_date']})")
else:
    print(f"⚠️  Expected 5 test fixtures, found {len(found)}")
    for f in found:
        print(f"   {f['league']}: {f['home_team']} vs {f['away_team']}")

# 3. Clean up test data
print("\n" + "=" * 60)
print("STEP 3: Cleaning up test data...")
print("=" * 60)

try:
    db.table("weekly_schedule").delete().eq("status", "TEST").execute()
    print("✅ Test fixtures cleaned up successfully")
except Exception as e:
    print(f"⚠️  Cleanup failed: {e}")

print("\n" + "=" * 60)
print("PIPELINE SMOKE TEST COMPLETE")
print("If all 3 steps show ✅, your system is ready for Thursday!")
print("=" * 60)
