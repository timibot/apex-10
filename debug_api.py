"""Debug: simulate gates on current fixtures to see rejection reasons."""
import logging
logging.basicConfig(level=logging.INFO)

from apex10.db import get_client
from apex10.filters.gates import Candidate, run_all_gates

db = get_client()
r = db.table("upcoming_fixtures").select("*").execute()

candidates = []
for d in r.data:
    c = Candidate(
        fixture_id=d.get("id", 0),
        league=d.get("league", ""),
        home_team=d.get("home_team", ""),
        away_team=d.get("away_team", ""),
        bet_type=d.get("best_bet_type", "Home Win"),
        odds=d.get("best_bet_odds", 1.5),
        lgbm_prob=d.get("lgbm_prob", 0),
        xgb_prob=d.get("xgb_prob", 0),
        opening_odds=d.get("opening_odds", 1.5),
        key_player_absent_home=d.get("key_player_absent_home", 0),
        key_player_absent_away=d.get("key_player_absent_away", 0),
        features_complete=d.get("features_complete", True),
    )
    candidates.append(c)

qualified, rejections = run_all_gates(candidates)
print(f"\n{len(qualified)} qualified, {len(rejections)} rejected\n")
print("Rejected fixtures:")
for rej in rejections[:10]:
    print(f"  {rej['home_team']:20} vs {rej['away_team']:20} Gate {rej['gate']} ({rej['gate_name']}): {rej['reason']}")
