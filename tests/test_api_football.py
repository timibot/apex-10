"""Tests for API-Football fetcher — all network calls mocked."""
from unittest.mock import MagicMock

from apex10.cache.api_football import parse_fixture, upsert_matches


class TestParseFixture:
    def _raw(self, home_goals=2, away_goals=1) -> dict:
        return {
            "fixture": {"id": 12345, "date": "2024-01-15T15:00:00+00:00"},
            "teams": {
                "home": {"name": "Arsenal"},
                "away": {"name": "Chelsea"},
            },
            "goals": {"home": home_goals, "away": away_goals},
            "league": {"name": "Premier League", "season": 2023},
        }

    def test_parse_valid_fixture(self):
        result = parse_fixture(self._raw())
        assert result is not None
        assert result["api_match_id"] == 12345
        assert result["home_team"] == "Arsenal"
        assert result["away_team"] == "Chelsea"
        assert result["home_goals"] == 2
        assert result["away_goals"] == 1
        assert result["match_date"] == "2024-01-15"
        assert result["status"] == "finished"

    def test_parse_returns_none_on_missing_goals(self):
        raw = self._raw()
        raw["goals"]["home"] = None
        assert parse_fixture(raw) is None

    def test_parse_returns_none_on_missing_key(self):
        assert parse_fixture({}) is None

    def test_date_truncated_to_10_chars(self):
        result = parse_fixture(self._raw())
        assert len(result["match_date"]) == 10

    def test_goals_cast_to_int(self):
        raw = self._raw()
        raw["goals"]["home"] = "2"  # String from API
        result = parse_fixture(raw)
        assert isinstance(result["home_goals"], int)


class TestUpsertMatches:
    def test_upsert_empty_list_returns_zero(self):
        mock_db = MagicMock()
        result = upsert_matches([], mock_db)
        assert result == 0
        mock_db.table.assert_not_called()

    def test_upsert_calls_correct_table(self):
        mock_db = MagicMock()
        mock_db.table.return_value.upsert.return_value.execute.return_value.data = [{"id": 1}]
        matches = [{"api_match_id": 1, "home_team": "Arsenal"}]
        upsert_matches(matches, mock_db)
        mock_db.table.assert_called_with("matches")

    def test_upsert_uses_conflict_key(self):
        mock_db = MagicMock()
        mock_db.table.return_value.upsert.return_value.execute.return_value.data = []
        upsert_matches([{"api_match_id": 1}], mock_db)
        mock_db.table.return_value.upsert.assert_called_once_with(
            [{"api_match_id": 1}], on_conflict="api_match_id"
        )
