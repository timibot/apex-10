"""Show upcoming fixture predictions."""
from apex10.db import get_client
db = get_client()
r = db.table("upcoming_fixtures").select(
    "home_team,away_team,lgbm_prob,xgb_prob,consensus_prob,match_date"
).order("match_date").execute()
print(f"{'Date':<12} {'Match':<45} {'LGBM':>6} {'XGB':>6} {'CONS':>6}")
print("-" * 80)
for d in r.data:
    match = f"{d['home_team']} vs {d['away_team']}"
    print(f"{d['match_date']:<12} {match:<45} {d['lgbm_prob']:>6.3f} {d['xgb_prob']:>6.3f} {d['consensus_prob']:>6.3f}")
