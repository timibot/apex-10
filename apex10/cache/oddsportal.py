"""
Oddsportal BFF JSON endpoint fetcher for Asian Handicap odds.
Method priority:
  1. Direct JSON endpoint (XHR-discovered, fastest)
  2. Playwright-stealth (CI environments)
  3. Camoufox (local fallback for Cloudflare-blocked pages)

Fetches AH -0.5, AH -1.0, AH -1.5, BTTS odds not available on free APIs.
Falls back gracefully to Dixon-Coles derivation if all methods fail.
"""
from __future__ import annotations

import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

# Discovered XHR endpoints (re-discover if these 404)
ODDSPORTAL_API_BASE = "https://www.oddsportal.com/api/v2"

# Realistic browser headers
STEALTH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.oddsportal.com/",
    "Origin": "https://www.oddsportal.com",
}


def fetch_ah_odds_direct(
    home_team: str,
    away_team: str,
    match_date: str,
) -> dict | None:
    """
    Attempt to fetch AH odds via direct JSON endpoint.
    Returns dict of {bet_type: odds} or None on failure.
    """
    try:
        search_url = f"{ODDSPORTAL_API_BASE}/search-matches"
        params = {
            "query": f"{home_team} {away_team}",
            "date": match_date,
        }

        with httpx.Client(
            timeout=15.0, headers=STEALTH_HEADERS
        ) as client:
            resp = client.get(search_url, params=params)
            if resp.status_code != 200:
                logger.debug(
                    f"Oddsportal search returned "
                    f"{resp.status_code}"
                )
                return None

            matches = (
                resp.json().get("data", {}).get("rows", [])
            )
            if not matches:
                return None

            event_id = matches[0].get("id")
            if not event_id:
                return None

            # Fetch odds for this event
            time.sleep(1.0)  # Polite delay
            odds_url = (
                f"{ODDSPORTAL_API_BASE}/match-odds/{event_id}"
            )
            odds_resp = client.get(
                odds_url, headers=STEALTH_HEADERS
            )

            if odds_resp.status_code != 200:
                return None

            return _parse_oddsportal_response(odds_resp.json())

    except Exception as e:
        logger.debug(f"Oddsportal direct fetch failed: {e}")
        return None


def fetch_ah_odds_playwright(
    home_team: str,
    away_team: str,
    match_date: str,
) -> dict | None:
    """
    Playwright-stealth fallback for CI environments.
    Used when direct JSON endpoint returns 403.
    """
    try:
        from playwright.sync_api import sync_playwright
        from playwright_stealth import stealth_sync  # type: ignore

        search_query = (
            f"{home_team} {away_team} {match_date}"
        )
        search_url = (
            f"https://www.oddsportal.com/search/results/"
            f"?q={search_query}"
        )

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=STEALTH_HEADERS["User-Agent"]
            )
            page = context.new_page()
            stealth_sync(page)

            page.goto(
                search_url,
                wait_until="networkidle",
                timeout=30000,
            )
            content = page.content()
            browser.close()

        # Parse match link from page content
        match = re.search(
            r'href="(/football/[^"]+/)"', content
        )
        if not match:
            return None

        match_url = (
            f"https://www.oddsportal.com"
            f"{match.group(1)}asian-handicap/"
        )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            stealth_sync(page)
            page.goto(
                match_url,
                wait_until="networkidle",
                timeout=30000,
            )
            content = page.content()
            browser.close()

        return _parse_html_odds_table(content)

    except ImportError:
        logger.debug(
            "playwright-stealth not installed — skipping"
        )
        return None
    except Exception as e:
        logger.debug(f"Playwright fetch failed: {e}")
        return None


def fetch_ah_odds_camoufox(
    home_team: str,
    away_team: str,
    match_date: str,
) -> dict | None:
    """
    Camoufox fallback — Firefox-based stealth browser.
    Best resistance to Cloudflare Turnstile. Local use only.
    """
    try:
        from camoufox.sync_api import Camoufox  # type: ignore

        match_url = (
            f"https://www.oddsportal.com/search/results/"
            f"?q={home_team}+{away_team}"
        )

        with Camoufox(headless=True) as browser:
            page = browser.new_page()
            page.goto(match_url, wait_until="networkidle")
            content = page.content()

        return _parse_html_odds_table(content)

    except ImportError:
        logger.debug("camoufox not installed — skipping")
        return None
    except Exception as e:
        logger.debug(f"Camoufox fetch failed: {e}")
        return None


def fetch_ah_odds(
    home_team: str,
    away_team: str,
    match_date: str,
) -> dict | None:
    """
    Full fallback chain for AH odds:
    1. Direct JSON endpoint
    2. Playwright-stealth (CI)
    3. Camoufox (local)
    4. Returns None → caller uses Dixon-Coles derivation
    """
    result = fetch_ah_odds_direct(
        home_team, away_team, match_date
    )
    if result:
        logger.info(
            f"AH odds via direct endpoint: "
            f"{home_team} vs {away_team}"
        )
        return result

    result = fetch_ah_odds_playwright(
        home_team, away_team, match_date
    )
    if result:
        logger.info(
            f"AH odds via Playwright: "
            f"{home_team} vs {away_team}"
        )
        return result

    result = fetch_ah_odds_camoufox(
        home_team, away_team, match_date
    )
    if result:
        logger.info(
            f"AH odds via Camoufox: "
            f"{home_team} vs {away_team}"
        )
        return result

    logger.info(
        f"All AH odds methods failed for "
        f"{home_team} vs {away_team} — "
        f"Dixon-Coles derivation will be used (Layer 3)"
    )
    return None


def _parse_oddsportal_response(data: dict) -> dict | None:
    """Parse Oddsportal JSON API response into standardised odds dict."""
    try:
        odds = {}
        markets = data.get("data", {}).get("markets", {})

        ah_data = markets.get("asian-handicap", {})
        if ah_data:
            odds["ah_minus_0_5"] = float(
                ah_data.get("home", 0) or 0
            )

        btts_data = markets.get("both-teams-to-score", {})
        if btts_data:
            odds["btts_no"] = float(
                btts_data.get("no", 0) or 0
            )

        return odds if odds else None
    except Exception:
        return None


def _parse_html_odds_table(html: str) -> dict | None:
    """Parse AH odds from rendered HTML page content."""
    odds = {}

    # Look for AH -0.5 odds pattern in rendered HTML
    ah_pattern = (
        r'data-odd="([\d.]+)"[^>]*data-handicap="-0\.5"'
    )
    match = re.search(ah_pattern, html)
    if match:
        odds["ah_minus_0_5"] = float(match.group(1))

    return odds if odds else None
