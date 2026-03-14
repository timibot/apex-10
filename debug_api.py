"""Show multi-market predictions."""
from apex10.db import get_client
db = get_client()
r = db.table("upcoming_fixtures").select(
    "league,home_team,away_team,best_bet_type,best_bet_odds,consensus_prob,lgbm_prob,xgb_prob,match_date"
).order("league").order("match_date").execute()
print(f"{'Date':<12} {'League':<12} {'Match':<36} {'Best Bet':<20} {'Odds':>5} {'Prob':>5}")
print("-" * 95)
for d in r.data:
    match = f"{d['home_team']} vs {d['away_team']}"
    print(f"{d['match_date']:<12} {d['league'][:11]:<12} {match:<36} {d['best_bet_type']:<20} {d['best_bet_odds']:>5.2f} {d['consensus_prob']:>5.3f}")
in_range = sum(1 for d in r.data if 1.20 <= d['best_bet_odds'] <= 1.49)
print(f"\nTotal: {len(r.data)} fixtures, {in_range} in odds range [1.20-1.49]")
