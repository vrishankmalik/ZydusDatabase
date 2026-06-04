"""Shared pytest fixtures, mock HTTP transports, and fixture-file helpers."""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

# Ensure project root is importable from every test file.
sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ── Fixture helpers ───────────────────────────────────────────────────────────

def load_json(rel_path: str) -> Any:
    return json.loads((FIXTURES_DIR / rel_path).read_bytes())


def load_html(rel_path: str) -> str:
    return (FIXTURES_DIR / rel_path).read_text(encoding="utf-8")


# ── DPD router ────────────────────────────────────────────────────────────────

# Stable DIN → drug_code mapping used in golden tests.
_DIN_TO_CODE: dict[str, str] = {
    "00326925": "11111",   # SINEQUAN
    "00000019": "22222",   # PLACIDYL CAP 200MG
    "02229895": "99999",   # GLUCOPHAGE
}


def _dpd_side_effect(request: httpx.Request) -> httpx.Response:
    path = request.url.path          # e.g. "/api/drug/drugproduct/"
    params = dict(request.url.params)

    # ── drugproduct ──────────────────────────────────────────────────────────
    if "drugproduct" in path:
        if "din" in params:
            din = params["din"].zfill(8)
            code = _DIN_TO_CODE.get(din)
            if code:
                return httpx.Response(200, json=load_json(f"dpd/drugproduct_code_{code}.json"))
            return httpx.Response(200, json={})
        if "id" in params:
            fp = FIXTURES_DIR / f"dpd/drugproduct_code_{params['id']}.json"
            if fp.exists():
                return httpx.Response(200, json=json.loads(fp.read_bytes()))
            return httpx.Response(200, json={})
        if "brandname" in params or "companyname" in params:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={})

    # ── activeingredient ─────────────────────────────────────────────────────
    if "activeingredient" in path:
        if "ingredientname" in params:
            slug = re.sub(r"\s+", "_", params["ingredientname"].lower().strip())
            fp = FIXTURES_DIR / f"dpd/activeingredient_{slug}.json"
            return httpx.Response(200, json=json.loads(fp.read_bytes()) if fp.exists() else [])
        if "id" in params:
            fp = FIXTURES_DIR / f"dpd/activeingredient_code_{params['id']}.json"
            return httpx.Response(200, json=json.loads(fp.read_bytes()) if fp.exists() else [])
        return httpx.Response(200, json=[])

    # ── form / route / status / schedule ─────────────────────────────────────
    for endpoint in ("form", "route", "status", "schedule"):
        if f"/{endpoint}/" in path:
            code = params.get("id", "0")
            fp = FIXTURES_DIR / f"dpd/{endpoint}_{code}.json"
            if fp.exists():
                return httpx.Response(200, json=json.loads(fp.read_bytes()))
            # Return canonical empty value per endpoint
            return httpx.Response(200, json={} if endpoint == "status" else [])

    return httpx.Response(404, text=f"No DPD fixture for {path}")


# ── NOC API routers ───────────────────────────────────────────────────────────

def _noc_api_side_effect(request: httpx.Request) -> httpx.Response:
    """Route NOC JSON API calls to recorded fixtures."""
    path = request.url.path
    params = dict(request.url.params)

    if "medicinalingredient" in path:
        return httpx.Response(200, json=load_json("noc/api_medicinalingredient.json"))

    if "drugproduct" in path:
        noc_id = params.get("id", "0")
        fp = FIXTURES_DIR / f"noc/api_drugproduct_{noc_id}.json"
        return httpx.Response(200, json=load_json(f"noc/api_drugproduct_{noc_id}.json") if fp.exists() else [])

    if "noticeofcompliancemain" in path:
        noc_id = params.get("id", "0")
        fp = FIXTURES_DIR / f"noc/api_main_{noc_id}.json"
        return httpx.Response(200, json=load_json(f"noc/api_main_{noc_id}.json") if fp.exists() else {})

    return httpx.Response(404, text=f"No NOC API fixture for {path}")


# ── GSUR router ───────────────────────────────────────────────────────────────

def _gsur_side_effect(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=load_html("generic_submissions/page.html"))


# ── Patent Register routers ───────────────────────────────────────────────────

def _pr_get_side_effect(request: httpx.Request) -> httpx.Response:
    html = load_html("patent_register/index.html")
    return httpx.Response(
        200, text=html,
        headers={"set-cookie": "JSESSIONID=fixture-jsession; Path=/pr-rdb/; HttpOnly"},
    )


def _pr_post_side_effect(request: httpx.Request) -> httpx.Response:
    raw = request.content.decode("utf-8", errors="replace")
    parts: dict[str, str] = {}
    for pair in raw.split("&"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            parts[k] = urllib.parse.unquote_plus(v)

    ingredient = parts.get("medicinalIngredient", "").upper()
    brand = parts.get("brandName", "").upper()

    if "METFORMIN" in ingredient or "GLUMETZA" in brand:
        return httpx.Response(200, text=load_html("patent_register/results_metformin.html"))
    return httpx.Response(200, text=load_html("patent_register/results_no_results.html"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_ollama_offline(monkeypatch):
    """Offline test guard: make _is_ollama_available() return False instantly.

    Without this, every call to parse_labeling_fields() blocks 30 s waiting for
    httpx DNS resolution of localhost:11434 to time out on this machine.
    Golden tests (TestPiqrayGolden) exercise the regex fallback path, which is
    the offline behavior we actually want to test.
    """
    try:
        import asyncio as _asyncio
        import app.enrichment.labeling as _lab

        async def _offline() -> bool:
            return False

        monkeypatch.setattr(_lab, "_is_ollama_available", _offline)
    except ImportError:
        pass  # labeling module not imported yet — nothing to patch


@pytest.fixture
def no_cache(monkeypatch):
    """Disable the SQLite disk cache in every source module."""
    noop: Any = lambda *a, **k: None
    for mod in (
        "app.sources.dpd",
        "app.sources.noc",
        "app.sources.patent_register",
        "app.sources.generic_submissions",
    ):
        monkeypatch.setattr(f"{mod}.cache_get", noop)
        monkeypatch.setattr(f"{mod}.cache_set", noop)


@pytest.fixture
def mock_dpd(no_cache):
    """respx mock for all DPD REST API calls."""
    with respx.mock(assert_all_called=False) as rx:
        rx.get(re.compile(r"https://health-products\.canada\.ca/api/drug/.*")).mock(
            side_effect=_dpd_side_effect
        )
        yield rx


@pytest.fixture
def mock_noc(no_cache):
    """respx mock for the NOC JSON API endpoints."""
    with respx.mock(assert_all_called=False) as rx:
        rx.get(re.compile(r"https://health-products\.canada\.ca/api/notice-of-compliance/.*")).mock(
            side_effect=_noc_api_side_effect
        )
        yield rx


@pytest.fixture
def mock_gsur(no_cache):
    """respx mock for the Generic Submissions HTML page."""
    with respx.mock(assert_all_called=False) as rx:
        rx.get(re.compile(r"https://www\.canada\.ca/.*generic-submissions.*")).mock(
            side_effect=_gsur_side_effect
        )
        yield rx


@pytest.fixture
def mock_patent_register(no_cache):
    """respx mock for the Patent Register index + search."""
    with respx.mock(assert_all_called=False) as rx:
        rx.get(re.compile(r"https://pr-rdb\.hc-sc\.gc\.ca/.*")).mock(
            side_effect=_pr_get_side_effect
        )
        rx.post(re.compile(r"https://pr-rdb\.hc-sc\.gc\.ca/.*")).mock(
            side_effect=_pr_post_side_effect
        )
        yield rx
