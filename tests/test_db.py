"""
Tests for apex10/db.py
Uses mocking — never hits the real database in unit tests.
Integration tests (marked separately) hit the TEST Supabase project.
"""
from unittest.mock import MagicMock, patch


class TestGetClient:
    def test_returns_client_instance(self):
        """get_client() must return a Supabase client object."""
        mock_client = MagicMock()
        with patch("apex10.db.create_client", return_value=mock_client), \
             patch("apex10.db.get_api_config") as mock_cfg:
            mock_cfg.return_value.SUPABASE_URL = "https://test.supabase.co"
            mock_cfg.return_value.SUPABASE_KEY = "test_key"
            # Clear lru_cache before test
            from apex10.db import get_client
            get_client.cache_clear()
            client = get_client()
            assert client is mock_client

    def test_client_is_cached(self):
        """get_client() must return the same instance on repeated calls."""
        mock_client = MagicMock()
        with patch("apex10.db.create_client", return_value=mock_client), \
             patch("apex10.db.get_api_config") as mock_cfg:
            mock_cfg.return_value.SUPABASE_URL = "https://test.supabase.co"
            mock_cfg.return_value.SUPABASE_KEY = "test_key"
            from apex10.db import get_client
            get_client.cache_clear()
            c1 = get_client()
            c2 = get_client()
            assert c1 is c2


class TestHealthCheck:
    def test_health_check_returns_true_on_success(self):
        mock_client = MagicMock()
        chain = mock_client.table.return_value.select.return_value.limit.return_value
        chain.execute.return_value = MagicMock()
        with patch("apex10.db.get_client", return_value=mock_client):
            from apex10.db import health_check
            assert health_check() is True

    def test_health_check_returns_false_on_exception(self):
        mock_client = MagicMock()
        mock_client.table.side_effect = Exception("Connection refused")
        with patch("apex10.db.get_client", return_value=mock_client):
            from apex10.db import health_check
            assert health_check() is False

    def test_health_check_does_not_raise(self):
        """health_check must NEVER raise — always returns bool."""
        with patch("apex10.db.get_client", side_effect=Exception("DB down")):
            from apex10.db import health_check
            result = health_check()
            assert isinstance(result, bool)
