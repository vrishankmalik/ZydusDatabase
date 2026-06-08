"""Test: async export path produces identical workbook to sync path.

Run with: pytest tests/test_export_async.py -m live -v
Requires live government sites (marks as 'live' to exclude from offline CI).
Uses alpelisib (PIQRAY) — small DIN set, ~2 DINs.
"""
from __future__ import annotations

import asyncio
import io
import time

import pandas as pd
import pytest

pytest_plugins = ("anyio",)


@pytest.mark.live
@pytest.mark.anyio
async def test_async_export_matches_sync_for_alpelisib() -> None:
    """Async export path must produce byte-for-byte equivalent Sheet 1 to the sync path."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # ── Async path ────────────────────────────────────────────────────
        start = await client.post(
            "/export/start",
            json={
                "q": "alpelisib",
                "field": "ingredient",
                "enable_ocr": False,
                "enable_llm": False,
            },
        )
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]

        # Poll /export/result until ready (max 5 min)
        async_bytes: bytes | None = None
        deadline = time.time() + 300
        while time.time() < deadline:
            await asyncio.sleep(3)
            res = await client.get(f"/export/result/{job_id}")
            if res.status_code == 200:
                async_bytes = res.content
                break
        assert async_bytes is not None, "Async export timed out after 5 min"

        # ── Sync path ─────────────────────────────────────────────────────
        sync = await client.get(
            "/api/export",
            params={"q": "alpelisib", "field": "ingredient"},
        )
        assert sync.status_code == 200, sync.text
        sync_bytes = sync.content

    # ── Compare Sheet 1 ───────────────────────────────────────────────────────
    sync_df = pd.read_excel(io.BytesIO(sync_bytes), sheet_name=0)
    async_df = pd.read_excel(io.BytesIO(async_bytes), sheet_name=0)

    assert set(sync_df.columns) == set(async_df.columns), (
        f"Column mismatch: sync={set(sync_df.columns) - set(async_df.columns)} "
        f"async={set(async_df.columns) - set(sync_df.columns)}"
    )

    sync_sorted = sync_df.sort_values("din").reset_index(drop=True)
    async_sorted = async_df.sort_values("din").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        sync_sorted,
        async_sorted,
        check_like=True,
        check_dtype=False,
    )

    # ── Compare Sheet 2 ───────────────────────────────────────────────────────
    sync_s2 = pd.read_excel(io.BytesIO(sync_bytes), sheet_name=1)
    async_s2 = pd.read_excel(io.BytesIO(async_bytes), sheet_name=1)

    pd.testing.assert_frame_equal(
        sync_s2.sort_values(sync_s2.columns[0]).reset_index(drop=True),
        async_s2.sort_values(async_s2.columns[0]).reset_index(drop=True),
        check_like=True,
        check_dtype=False,
    )
