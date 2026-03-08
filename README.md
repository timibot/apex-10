# APEX-10

Edge-detection engine for football accumulator betting.

## Quickstart

```bash
# 1. Clone & enter
git clone <repo-url> && cd apex10

# 2. Create virtual env
python -m venv .venv && .venv\Scripts\activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS / Linux

# 3. Install deps
pip install -e ".[dev]"

# 4. Copy env template & fill in secrets
cp .env.example .env

# 5. Run tests
pytest
```

## Project Structure

| Directory           | Purpose                            | Phase |
|---------------------|------------------------------------|-------|
| `apex10/config.py`  | Central constants & env loading    | 1     |
| `apex10/db.py`      | Supabase client singleton          | 1     |
| `apex10/cache/`     | Weekly data pull & caching         | 2     |
| `apex10/scoring/`   | Dixon-Coles, vig removal, edge     | 2–3   |
| `apex10/models/`    | LightGBM & XGBoost wrappers       | 3     |
| `apex10/filters/`   | Selection gates                    | 4     |
| `apex10/ticket/`    | Accumulator ticket builder         | 4     |
| `apex10/score/`     | Result scoring & settlement        | 5     |
| `scripts/`          | One-time DB init & utilities       | 1     |
| `tests/`            | Pytest suite per module            | 1–5   |

## CI / CD

GitHub Actions workflows live in `.github/workflows/`:

- **ci.yml** — lint + unit tests on every push
- **cache.yml** — weekly Thursday data pull (Phase 2)
- **ticket.yml** — weekly Thursday ticket generation (Phase 4)
- **score.yml** — weekly Monday result logging (Phase 5)
