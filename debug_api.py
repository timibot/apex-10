"""Quick debug: view confidence votes and gate results."""
import logging
logging.basicConfig(level=logging.INFO)

from apex10.db import get_client
from apex10.filters.gates import Candidate, run_all_gates

db = get_client()
rows = db.table("upcoming_fixtures").select("*").execute().data or []

candidates = []
for r in rows:
    votes = round(float(r.get("xgb_prob", 0)) * 5)
    c = Candidate(
        fixture_id=r["api_match_id"],
        league=r["league"],
        home_team=r["home_team"],
        away_team=r["away_team"],
        bet_type=r.get("best_bet_type", "Home Win"),
        odds=float(r["best_bet_odds"]),
        lgbm_prob=float(r["lgbm_prob"]),
        xgb_prob=float(r["xgb_prob"]),
        opening_odds=float(r["opening_odds"]),
        key_player_absent_home=int(r.get("key_player_absent_home", 0)),
        key_player_absent_away=int(r.get("key_player_absent_away", 0)),
        features_complete=bool(r.get("features_complete", True)),
        confidence_votes=votes,
    )
    candidates.append(c)

qualified, rejected = run_all_gates(candidates)

print(f"\n{len(qualified)} qualified, {len(rejected)} rejected")
print()

if qualified:
    print("QUALIFIED legs:")
    for c in qualified:
        print(f"  {c.home_team:20s} vs {c.away_team:20s} | {c.bet_type:20s} @{c.odds:.2f} | {c.confidence_votes}/5 votes | Tier {c.tier.value}")

print()
if rejected:
    print("Rejected fixtures:")
    for r in rejected:
        print(f"  {r['home_team']:20s} vs {r['away_team']:20s} Gate {r['gate']} ({r['gate_name']}): {r['reason']}")
