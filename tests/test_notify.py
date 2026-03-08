"""
Tests for Discord notification centre.
Never hits real Discord — all HTTP calls mocked.
"""
from unittest.mock import MagicMock, patch

from apex10.live.notify import (
    AlertLevel,
    _fmt,
    _send,
    brier_breach,
    brier_recovered,
    no_ticket_week,
    result_logged,
    retrain_complete,
    system_error,
    ticket_generated,
)


class TestSend:
    def test_returns_true_on_success(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with (
            patch("apex10.live.notify.httpx.Client") as mock_client,
            patch("apex10.live.notify.get_api_config") as mock_cfg,
        ):
            mock_cfg.return_value.DISCORD_WEBHOOK = (
                "https://discord.com/webhook"
            )
            ctx = mock_client.return_value.__enter__.return_value
            ctx.post.return_value = mock_resp
            assert _send("test message") is True

    def test_returns_false_on_http_error(self):
        with (
            patch("apex10.live.notify.httpx.Client") as mock_client,
            patch("apex10.live.notify.get_api_config") as mock_cfg,
        ):
            mock_cfg.return_value.DISCORD_WEBHOOK = (
                "https://discord.com/webhook"
            )
            ctx = mock_client.return_value.__enter__.return_value
            ctx.post.side_effect = Exception("timeout")
            assert _send("test message") is False

    def test_returns_false_when_no_webhook(self):
        with patch("apex10.live.notify.get_api_config") as mock_cfg:
            mock_cfg.return_value.DISCORD_WEBHOOK = ""
            assert _send("test message") is False

    def test_never_raises(self):
        """_send must NEVER raise — always returns bool."""
        with patch(
            "apex10.live.notify.get_api_config",
            side_effect=Exception("cfg error"),
        ):
            result = _send("test")
            assert isinstance(result, bool)


class TestFmt:
    def test_contains_level_emoji(self):
        msg = _fmt(AlertLevel.SUCCESS, "Title", "Body")
        assert "✅" in msg

    def test_contains_title(self):
        msg = _fmt(AlertLevel.WARNING, "MyTitle", "Body text")
        assert "MyTitle" in msg

    def test_contains_body(self):
        msg = _fmt(AlertLevel.INFO, "Title", "My body text")
        assert "My body text" in msg


class TestAlertMethods:
    """All alert methods must: call _send, return bool, never raise."""

    def _patch_send(self, return_val=True):
        return patch("apex10.live.notify._send", return_value=return_val)

    def test_ticket_generated_returns_bool(self):
        with self._patch_send():
            result = ticket_generated(
                "2026-01-05", 6, 12.4, 500.0, 0.07
            )
            assert isinstance(result, bool)

    def test_no_ticket_week_returns_bool(self):
        with self._patch_send():
            result = no_ticket_week(
                "2026-01-05", "No qualified candidates"
            )
            assert isinstance(result, bool)

    def test_result_logged_returns_bool(self):
        with self._patch_send():
            result = result_logged("2026-01-05", "loss", 9800.0, 0.19)
            assert isinstance(result, bool)

    def test_brier_breach_returns_bool(self):
        with self._patch_send():
            result = brier_breach(0.26, 0.24)
            assert isinstance(result, bool)

    def test_brier_recovered_returns_bool(self):
        with self._patch_send():
            result = brier_recovered(0.18)
            assert isinstance(result, bool)

    def test_retrain_complete_returns_bool(self):
        with self._patch_send():
            result = retrain_complete(2, 0.18, 0.19, deployed=True)
            assert isinstance(result, bool)

    def test_system_error_returns_bool(self):
        with self._patch_send():
            result = system_error("cache.py", "ConnectionError")
            assert isinstance(result, bool)

    def test_brier_breach_message_contains_threshold(self):
        captured = {}

        def capture(msg):
            captured["msg"] = msg
            return True

        with patch("apex10.live.notify._send", side_effect=capture):
            brier_breach(0.26, 0.24)
        assert "0.24" in captured["msg"]

    def test_retrain_not_deployed_message(self):
        captured = {}

        def capture(msg):
            captured["msg"] = msg
            return True

        with patch("apex10.live.notify._send", side_effect=capture):
            retrain_complete(2, 0.21, 0.22, deployed=False)
        assert (
            "not deployed" in captured["msg"].lower()
            or "retained" in captured["msg"].lower()
        )
