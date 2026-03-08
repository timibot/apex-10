"""Tests for historical odds loader."""
import pandas as pd

from apex10.cache.odds_loader import SEASONS, build_csv_url, parse_odds_rows


class TestBuildCsvUrl:
    def test_epl_2023_url(self):
        url = build_csv_url("EPL", 2023)
        assert "E0" in url
        assert "2324" in url
        assert url.startswith("https://www.football-data.co.uk")

    def test_all_seasons_have_url(self):
        for season in SEASONS.keys():
            url = build_csv_url("EPL", season)
            assert url.startswith("https://")


class TestParseOddsRows:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "Date": "15/01/2024", "PSH": 1.85, "PSD": 3.40, "PSA": 4.50},
            {"HomeTeam": "Liverpool", "AwayTeam": "ManCity",
             "Date": "20/01/2024", "PSH": 2.10, "PSD": 3.20, "PSA": 3.60},
            # Row with missing odds — should be skipped
            {"HomeTeam": "Tottenham", "AwayTeam": "Wolves",
             "Date": "21/01/2024", "PSH": None, "PSD": None, "PSA": None},
        ])

    def test_parses_valid_rows(self):
        records = parse_odds_rows(self._make_df(), "EPL", 2023)
        assert len(records) == 2

    def test_skips_rows_with_missing_odds(self):
        records = parse_odds_rows(self._make_df(), "EPL", 2023)
        teams = [r["home_team"] for r in records]
        assert "Tottenham" not in teams

    def test_record_structure(self):
        records = parse_odds_rows(self._make_df(), "EPL", 2023)
        r = records[0]
        assert r["bookmaker"] == "Pinnacle"
        assert r["market"] == "1X2"
        assert isinstance(r["odds_home"], float)

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=["HomeTeam", "AwayTeam", "Date", "PSH", "PSD", "PSA"])
        assert parse_odds_rows(df, "EPL", 2023) == []
