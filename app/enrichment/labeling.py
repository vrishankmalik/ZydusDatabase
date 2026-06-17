"""Product Monograph extraction — three-stage pipeline.

Stage 2 (this file, top half): DPD API + info page
  - active_ingredient  ← /activeingredient/?id={drug_code}  (API, reliable)
  - pack_size          ← parsed from product_information free-text (API, reliable)
  - pack_style         ← derived from container-type keyword in product_information
                         (falls back to info-page description; never uses empty package_type)
  - pdf_url            ← DPD info page "Labelling" link       (scrape)
  DPD API values are authoritative. They are NEVER overwritten by PDF guesses.

Stage 3 (this file, bottom half): PDF extraction
  Fields extracted ONLY from the PDF:
    excipients_core, excipients_coating, preservatives,
    ph, color, shape, size_mm, weight
  Per-strength matching: use the DIN's strength to scope §6 Description block.
  Section location by keyword → only that section passed to Ollama or regex.
  Regex extraction is the active path by default (NullProvider). A configured
  LLM provider (e.g. LLM_PROVIDER=azure_openai) is preferred when available.

  Scanned PDFs: OCR'd page-by-page via pdf2image + pytesseract (ENABLE_OCR=1 default).
  OCR text is disk-cached by PDF URL (7-day TTL). The is_scanned() guard no longer
  short-circuits extraction — a scanned PM is OCR'd and then processed normally.

Three output states (never conflate them):
  real value   — string extracted from the document
  NOT_IN_PM    — section was found and searched; field is absent in this PM
  ""           — no PM found / download failed / section not present

Accuracy rules (non-negotiable):
  - Every extracted value stores the page number it came from.
  - If pytesseract/pdf2image unavailable: log actionable install message, proceed
    with whatever text layer exists (never silently skip).
  - pH: if only a pH-dependent solubility table exists, return PH_SOLUBILITY_ONLY.

CLI:
  python -m app.enrichment.labeling --drug-code 12345 --din 02498014 --strength "50 mg"
  python -m app.enrichment.labeling --validate   # pack_style/pack_size extraction demo
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import httpx
from bs4 import BeautifulSoup

from app.cache import cache_get, cache_set
from app.config import DPD_BASE, ENABLE_OCR, HTTP_TIMEOUT, USER_AGENT
from app.llm.provider import get_llm_provider
from app.enrichment.store import get_labeling_for_din, upsert_labeling

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
_DPD_INFO_BASE = "https://health-products.canada.ca/dpd-bdpp/info"

# ── Thread pool for CPU-bound PDF work ───────────────────────────────────────
# pdfplumber and pytesseract are synchronous; running them in the event loop
# blocks all concurrent labeling tasks.  A dedicated pool keeps the loop free.
_PDF_THREAD_POOL = ThreadPoolExecutor(
    max_workers=int(__import__("os").getenv("PDF_THREAD_WORKERS", "8")),
    thread_name_prefix="labeling_pdf",
)

# ── Shared httpx client (connection pooling) ──────────────────────────────────
# Creating a new AsyncClient per request means a fresh TCP+TLS handshake for
# every DPD API call (~200 ms each × 450 calls = ~90 s wasted on cold runs).
# A single shared client reuses connections: subsequent requests to the same
# host skip the handshake and return in <20 ms.
_shared_client: Optional[httpx.AsyncClient] = None
_shared_client_lock = asyncio.Lock()


async def _get_shared_client() -> httpx.AsyncClient:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        return _shared_client
    async with _shared_client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                timeout=HTTP_TIMEOUT,
                limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
                follow_redirects=True,
            )
        return _shared_client

# ── Per-URL download deduplication ───────────────────────────────────────────
# Multiple DINs sharing one Product Monograph URL would otherwise all start
# an HTTP download before any finishes and caches the bytes.  A per-URL Lock
# serialises them so only the first download is live; the rest get cache hits.
_PDF_DL_LOCKS: dict[str, asyncio.Lock] = {}
_PDF_DL_LOCKS_META = asyncio.Lock()

# Per-URL extraction deduplication — same logic for the CPU-bound OCR step.
_PDF_EXTRACT_LOCKS: dict[str, asyncio.Lock] = {}
_PDF_EXTRACT_LOCKS_META = asyncio.Lock()

NOT_IN_PM = "Not in PM"
NOT_STATED = NOT_IN_PM        # alias used by tests and external callers
NO_PM_AVAILABLE = "No PM available"  # no PM file exists for the DIN
NEEDS_OCR = "needs OCR / manual check"
PH_SOLUBILITY_ONLY = "Not stated (pH-dependent solubility only)"

# Minimum characters on a page to consider it selectable text (not a scanned image)
_MIN_TEXT_CHARS = 50

_LABELING_FIELDS = (
    "active_ingredient", "nonmedicinal_ingredients",
    "pack_size", "pack_style",
    "color", "shape", "size_mm", "weight", "ph",
)

# Fields sourced from DPD API (Stage 2) — never extracted from PDF
_STAGE2_FIELDS = frozenset({"active_ingredient", "pack_size", "pack_style"})

# ── Container vocabulary (longest-match-first) ────────────────────────────────
# Each entry: (UPPER-CASE keyword to search, Title Case display label)
_CONTAINER_VOCAB_ORDERED: list[tuple[str, str]] = [
    ("PREFILLED SYRINGE", "Prefilled Syringe"),
    ("PRE-FILLED SYRINGE", "Prefilled Syringe"),
    ("AUTO-INJECTOR", "Auto-Injector"),
    ("AUTO INJECTOR", "Auto-Injector"),
    ("AUTOINJECTOR", "Auto-Injector"),
    ("STICK PACK", "Stick Pack"),
    ("BLISTER PACK", "Blister Pack"),
    ("AMPOULE", "Ampoule"),
    ("AMPULE", "Ampoule"),
    ("AMPUL", "Ampoule"),
    ("VIAL", "Vial"),
    ("SYRINGE", "Syringe"),
    ("CARTRIDGE", "Cartridge"),
    ("BLISTER", "Blister"),
    ("BOTTLE", "Bottle"),
    ("JAR", "Jar"),
    ("TUBE", "Tube"),
    ("SACHET", "Sachet"),
    ("POUCH", "Pouch"),
    ("CARTON", "Carton"),
    ("BAG", "Bag"),
    ("CANISTER", "Canister"),
    ("INHALER", "Inhaler"),
    ("DROPPER", "Dropper"),
    ("SUPPOSITORY", "Suppository"),
    ("PEN", "Pen"),
    ("KIT", "Kit"),
]

# Pre-built regex patterns for speed (word-boundary anchored, case-insensitive).
# Optional trailing 'S' allows matching plural forms (e.g. "vials", "blisters",
# "blister packs") without requiring an exact word boundary after the keyword root.
_CONTAINER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"S?\b", re.IGNORECASE), label)
    for kw, label in _CONTAINER_VOCAB_ORDERED
]


def _extract_pack_style_from_text(text: str, source_label: str = "") -> Optional[str]:
    """Return "; "-joined Title-Case container labels for all vocabulary keywords found.

    Traverses the vocab in longest-match-first order to collect all distinct
    container types.  Shorter labels that are already covered by a compound
    label found earlier are suppressed (e.g. "Blister" is dropped when
    "Blister Pack" was already matched in the same text).
    """
    found: list[str] = []
    seen: set[str] = set()
    for pattern, label in _CONTAINER_PATTERNS:
        if label in seen:
            continue
        if pattern.search(text):
            found.append(label)
            seen.add(label)
    # Remove labels whose full text appears as a trailing word in a longer compound
    # label already matched (e.g. "Blister" when "Blister Pack" was also found).
    final = [
        lbl for lbl in found
        if not any(other != lbl and other.upper().endswith(lbl.upper()) for other in seen)
    ]
    if final:
        result = "; ".join(final)
        if source_label:
            logger.info(
                "pack_style=%r found in %s (text snippet: %r)",
                result, source_label, text[:100],
            )
        return result
    return None


def _extract_pack_size_from_product_info(prod_info: str) -> Optional[str]:
    """Parse product_information free-text into a clean pack_size string.

    When multiple container types appear with different counts, produces a
    descriptive "N-count container" entry for each so the result is unambiguous.

    Examples:
      "FOR I.V. INFUSION ONLY. 80MG/ML(RECONST.) - 5ML VIAL."   → "5 mL"
      "24/50/100/200"                                             → "24, 50, 100, 200 count"
      "100/500"                                                   → "100, 500 count"
      "100 TABLETS"                                               → "100 count"
      "5ML"                                                       → "5 mL"
      "8 COUNT BLISTERS AND BOTTLES OF 120 CAPSULES"             → "8-count blister; 120-count bottle"
      "8 COUNT BLISTERS"                                          → "8 count"
    """
    text_upper = prod_info.upper()

    # Slash-separated container sizes with mass/volume units: "20g/50g/500g", "10mL/20mL"
    # These appear in DPD product_information for weight/volume-based containers.
    # Must be handled before the single-volume check so multi-size cases win.
    if '/' in text_upper:
        size_parts = re.findall(r'\b(\d+(?:\.\d+)?)\s*(G|ML|L)\b', text_upper)
        if len(size_parts) >= 2:
            units = {u for _, u in size_parts}
            if len(units) == 1:
                display_unit = {'G': 'g', 'ML': 'mL', 'L': 'L'}[size_parts[0][1]]
                return ", ".join(f"{n} {display_unit}" for n, _ in size_parts)

    # Volume: standalone N mL / N L — NOT a concentration like 80MG/ML
    vol_m = re.search(
        r'(?:^|[\s\-\.,(])(\d+(?:\.\d+)?)\s*(ML|L)\b(?!\s*/)',
        text_upper,
    )
    if vol_m:
        num = float(vol_m.group(1))
        unit = "mL" if vol_m.group(2) == "ML" else "L"
        return f"{num:g} {unit}"

    # Slash-separated pure-integer counts: "24/50/100/200" or "100/500"
    slash_m = re.search(r'\b(\d+(?:/\d+)+)\b', text_upper)
    if slash_m:
        parts = slash_m.group(1).split("/")
        if all(p.isdigit() for p in parts):
            return ", ".join(parts) + " count"

    # Try to find (count, container) pairs so the result is self-describing.
    # Two sub-patterns tried per container keyword (longest-match-first order):
    #   Forward:  "N COUNT/CAPSULES/TABLETS CONTAINER" → e.g. "8 count blisters"
    #   Reverse:  "CONTAINER OF N [CAPSULES/TABLETS]?" → e.g. "bottles of 120 capsules"
    pairs: list[tuple[int, str]] = []
    seen_labels: set[str] = set()

    for kw, label in _CONTAINER_VOCAB_ORDERED:
        kw_pat = re.escape(kw) + r"S?"  # allow plural

        fwd = re.search(
            r'\b(\d+)\s+(?:COUNT|TABLETS?|CAPSULES?|CAPS?|UNITS?)\s+' + kw_pat + r'\b',
            text_upper,
        )
        if fwd and label not in seen_labels:
            pairs.append((int(fwd.group(1)), label))
            seen_labels.add(label)
            continue

        rev = re.search(
            kw_pat + r'\s+OF\s+(\d+)(?:\s+(?:TABLETS?|CAPSULES?|CAPS?|UNITS?))?\b',
            text_upper,
        )
        if rev and label not in seen_labels:
            pairs.append((int(rev.group(1)), label))
            seen_labels.add(label)

    if pairs:
        if len(pairs) == 1:
            # Single container — count alone is enough; pack_style carries the container name.
            return f"{pairs[0][0]} count"
        # Multiple containers — name each for clarity.
        return "; ".join(f"{n}-count {lbl.lower()}" for n, lbl in pairs)

    # Fallback: strip container words and pick up any count/unit expressions.
    text = text_upper
    for pattern, _ in _CONTAINER_PATTERNS:
        text = pattern.sub(" ", text)
    text = text.strip()

    count_matches = re.findall(
        r'\b(\d+)\s+(?:TABLETS?|CAPSULES?|CAPS?|UNITS?|COUNT)\b',
        text,
    )
    of_matches = re.findall(r'\bOF\s+(\d+)\b', text)

    all_found = list(dict.fromkeys(count_matches + of_matches))
    if all_found:
        return "; ".join(f"{m} count" for m in all_found)

    return None


# ── Stage 2: DPD API + info page ─────────────────────────────────────────────

async def _fetch_active_ingredient_api(drug_code: int) -> Optional[str]:
    """Fetch active ingredient name(s) from DPD /activeingredient/ API."""
    cache_key = f"ai:{drug_code}"
    cached = cache_get("dpd_ai", cache_key)
    if cached is not None:
        return cached or None

    try:
        client = await _get_shared_client()
        r = await client.get(
                f"{DPD_BASE}/activeingredient/",
                params={"id": drug_code, "lang": "en", "type": "json"},
                headers=_HEADERS,
            )
        if r.status_code != 200:
            return None
        data = r.json()
        entries = data if isinstance(data, list) else ([data] if isinstance(data, dict) and data else [])
        names = []
        for e in entries:
            if isinstance(e, dict):
                n = e.get("ingredient_name") or e.get("active_ingredient_name") or e.get("ingredientName") or ""
                if n:
                    names.append(n.strip())
        result = "; ".join(dict.fromkeys(names)) or None
        cache_set("dpd_ai", cache_key, result or "")
        return result
    except Exception as exc:
        logger.debug("activeingredient API failed for drug_code=%s: %s", drug_code, exc)
        return None


async def _fetch_packaging_api(drug_code: int) -> tuple[Optional[str], Optional[str]]:
    """Return (pack_size, pack_style) from DPD /packaging/ API.

    pack_size  — parsed from product_information (or package_size/package_size_unit
                 when present), cleaned of container words
    pack_style — derived from container-type keyword in product_information text;
                 package_type is NOT used because it is empty for most products

    This is Stage 2 — authoritative. Never overwrite with PDF guesses.
    """
    cache_key = f"pkg:{drug_code}"
    cached = cache_get("dpd_packaging", cache_key)
    if cached is not None:
        d = cached
        return d.get("pack_size") or None, d.get("pack_style") or None

    try:
        client = await _get_shared_client()
        r = await client.get(
                f"{DPD_BASE}/packaging/",
                params={"id": drug_code, "type": "json"},
                headers=_HEADERS,
            )
        if r.status_code != 200:
            logger.debug("packaging API returned %d for drug_code=%s", r.status_code, drug_code)
            return None, None
        data = r.json()
        entries = data if isinstance(data, list) else ([data] if isinstance(data, dict) and data else [])

        sizes: list[str] = []
        styles: list[str] = []

        for e in entries:
            if not isinstance(e, dict):
                continue

            # Standard numeric size fields (often populated for solid dosage forms)
            size_num = (
                e.get("package_size") or e.get("packageSize") or e.get("PACKAGE_SIZE") or ""
            )
            size_unit = (
                e.get("package_size_unit") or e.get("packageSizeUnit") or
                e.get("PACKAGE_SIZE_UNIT") or ""
            )
            if size_num:
                sizes.append(f"{size_num} {size_unit}".strip())

            prod_info = (e.get("product_information") or "").strip()

            if prod_info:
                # pack_style: extract container keyword from free-text
                style = _extract_pack_style_from_text(prod_info, "product_information")
                if style:
                    styles.append(style)

                # pack_size fallback: parse product_information when standard fields empty
                if not size_num:
                    parsed_size = _extract_pack_size_from_product_info(prod_info)
                    if parsed_size:
                        sizes.append(parsed_size)
                    else:
                        logger.debug(
                            "pack_size: product_information parser returned None for %r "
                            "(drug_code=%s) — raw value not appended",
                            prod_info[:80], drug_code,
                        )

        pack_size = "; ".join(dict.fromkeys(s for s in sizes if s)) or None
        pack_style = "; ".join(dict.fromkeys(s for s in styles if s)) or None

        if not pack_size:
            logger.info(
                "packaging API: all size fields empty for drug_code=%s; raw entries: %s",
                drug_code, entries[:2],
            )

        cache_set("dpd_packaging", cache_key, {"pack_size": pack_size, "pack_style": pack_style})
        return pack_size, pack_style

    except Exception as exc:
        logger.debug("packaging API failed for drug_code=%s: %s", drug_code, exc)
        return None, None


async def _scrape_dpd_info_page(drug_code: int) -> dict[str, Optional[str]]:
    """Scrape the DPD info page for:
      - The Labelling / Product Monograph PDF link (and its date)
      - The Description field (cross-check for pack_size when API is sparse)

    Returns: {pdf_url, pdf_date, description}
    """
    cache_key = f"info:{drug_code}"
    cached = cache_get("dpd_info_page", cache_key)
    if cached is not None:
        return cached

    page_url = f"{_DPD_INFO_BASE}?lang=eng&code={drug_code}"
    result: dict[str, Optional[str]] = {"pdf_url": None, "pdf_date": None, "description": None}

    try:
        client = await _get_shared_client()
        r = await client.get(
                page_url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
            )
        if r.status_code != 200:
            logger.debug("DPD info page HTTP %d for drug_code=%s", r.status_code, drug_code)
            cache_set("dpd_info_page", cache_key, result)
            return result
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as exc:
        logger.debug("DPD info page fetch failed for drug_code=%s: %s", drug_code, exc)
        cache_set("dpd_info_page", cache_key, result)
        return result

    for link in soup.find_all("a", href=True):
        href: str = link["href"]
        label_text = link.get_text(strip=True).lower()

        is_pdf = (
            href.lower().endswith(".pdf")
            or "pdf.hres.ca" in href.lower()
            or re.search(r"/pdf/", href, re.IGNORECASE)
        )
        is_labelling_link = (
            "label" in label_text
            or "monograph" in label_text
            or "pm" == label_text
            or is_pdf
        )
        if not is_labelling_link:
            continue

        if not href.startswith("http"):
            href = f"https://health-products.canada.ca{href}"

        if is_pdf or "label" in label_text or "monograph" in label_text:
            result["pdf_url"] = href
            parent = link.find_parent(["td", "li", "div", "p"])
            if parent:
                date_m = re.search(
                    r"(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}|\w+\s+\d{1,2},?\s+\d{4})",
                    parent.get_text(" ", strip=True),
                )
                if date_m:
                    result["pdf_date"] = date_m.group(1)
            break

    if not result["pdf_url"]:
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                if not href.startswith("http"):
                    href = f"https://health-products.canada.ca{href}"
                result["pdf_url"] = href
                logger.debug("drug_code=%s: using last-resort PDF link %s", drug_code, href)
                break

    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            header = cells[0].get_text(strip=True).lower()
            if "description" in header:
                result["description"] = cells[1].get_text(" ", strip=True) or None
                break

    if result["pdf_url"]:
        logger.debug("drug_code=%s: Labelling PDF → %s", drug_code, result["pdf_url"])
    else:
        logger.info("drug_code=%s: no Labelling PDF link found on DPD info page", drug_code)

    cache_set("dpd_info_page", cache_key, result)
    return result


async def fetch_stage2_data(drug_code: int) -> dict[str, Optional[str]]:
    """Fetch all Stage 2 fields for a drug_code.

    Returns:
      active_ingredient, pack_size, pack_style  ← from DPD APIs (authoritative)
      pdf_url, pdf_date, description             ← from DPD info page

    pack_style cascade:
      1. product_information container keyword (in _fetch_packaging_api)
      2. info-page Description container keyword (here)
      Neither falls back to the empty package_type field.
    """
    ai_task = asyncio.create_task(_fetch_active_ingredient_api(drug_code))
    pkg_task = asyncio.create_task(_fetch_packaging_api(drug_code))
    info_task = asyncio.create_task(_scrape_dpd_info_page(drug_code))

    ai, (pack_size, pack_style), info = await asyncio.gather(ai_task, pkg_task, info_task)

    # pack_size fallback: try Description field if API produced nothing
    if not pack_size and info.get("description"):
        desc = info["description"]
        parsed = _extract_pack_size_from_product_info(desc)
        if parsed:
            pack_size = parsed
            logger.debug("drug_code=%s: pack_size from Description fallback: %r", drug_code, pack_size)
        else:
            m = re.search(
                r"(\d+(?:\s*x\s*\d+)?\s*(?:ml|mg|g|tablet|capsule|cap|vial|sachet|ampul)[^\s,]*)",
                desc, re.IGNORECASE,
            )
            if m:
                pack_size = m.group(0).strip()
                logger.debug("drug_code=%s: pack_size from Description regex fallback: %r", drug_code, pack_size)

    # pack_style fallback: try Description if product_information had no keyword
    if not pack_style and info.get("description"):
        pack_style = _extract_pack_style_from_text(info["description"], "description")

    return {
        "active_ingredient": ai,
        "pack_size": pack_size,
        "pack_style": pack_style,
        "pdf_url": info.get("pdf_url"),
        "pdf_date": info.get("pdf_date"),
        "description": info.get("description"),
    }


# Keep old function name for backward compatibility
async def fetch_labeling_pdf_url(drug_code: int) -> Optional[str]:
    info = await _scrape_dpd_info_page(drug_code)
    return info.get("pdf_url")


# ── PDF text extraction with per-page OCR ────────────────────────────────────

def _ocr_single_page(pdf_bytes: bytes, page_num: int) -> Optional[str]:
    """Render page_num (1-indexed) at 300 DPI and run Tesseract OCR.

    Returns OCR text, or None if pdf2image / pytesseract are not installed
    (both install instructions are logged).
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        logger.error(
            "pdf2image is not installed — cannot OCR scanned PDF pages. "
            "Install: pip install pdf2image  and  "
            "brew install poppler  (macOS) / apt-get install poppler-utils  (Linux)."
        )
        return None

    try:
        import pytesseract
    except ImportError:
        logger.error(
            "pytesseract is not installed — cannot OCR scanned PDF pages. "
            "Install: pip install pytesseract  and  "
            "brew install tesseract  (macOS) / apt-get install tesseract-ocr  (Linux)."
        )
        return None

    try:
        images = convert_from_bytes(
            pdf_bytes, dpi=300,
            first_page=page_num, last_page=page_num,
        )
        if not images:
            logger.warning("pdf2image returned no images for page %d", page_num)
            return None
        text: str = pytesseract.image_to_string(images[0], lang="eng")
        return text
    except Exception as exc:
        logger.warning("OCR failed for page %d: %s", page_num, exc)
        return None


def _extract_text_with_ocr(
    pdf_bytes: bytes,
    cache_key: str,
    enable_ocr: bool = True,
) -> tuple[list[tuple[int, str]], bool]:
    """Extract text from PDF, OCR'ing pages whose text layer is below threshold.

    Returns:
      pages     — list of (1-indexed page_num, text_string)
      ocr_used  — True if at least one page was OCR'd
    """
    import io
    import pdfplumber  # deferred — optional dependency

    # Check OCR cache (keyed by PDF URL / caller-supplied key)
    if enable_ocr and cache_key:
        cached = cache_get("ocr_text", cache_key)
        if cached is not None:
            pages = [(p["page"], p["text"]) for p in cached]
            ocr_used = any(p.get("ocr") for p in cached)
            logger.debug("OCR text cache hit for key %r (%d pages)", cache_key, len(pages))
            return pages, ocr_used

    pages: list[tuple[int, str]] = []
    ocr_used = False
    page_meta: list[dict] = []  # for cache serialization

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            # For pages where NM ingredients appear as a table column, pdfplumber's
            # text extraction interleaves columns and produces garbage.  Replace the
            # bare column-header occurrence in-place with the inline form so it lands
            # cleanly inside the §6 section text.
            nm_from_table = _extract_nm_from_table_column(page)
            if nm_from_table:
                # Append (not prepend) a clean NM heading line so that _find_section can
                # collect it as part of §6 text.  Prepending would put the NM line BEFORE
                # the §6 start marker when both are on the same page, causing _find_section
                # to start from the marker and skip the prepended NM entirely.
                # Appending ensures the NM line is always inside whatever section is active.
                raw_text = raw_text + f"\nNon-Medicinal Ingredients: {nm_from_table}"
            used_ocr = False

            if len(raw_text) < _MIN_TEXT_CHARS and enable_ocr:
                logger.info(
                    "Page %d: text layer has only %d chars (threshold=%d) → OCR",
                    i, len(raw_text), _MIN_TEXT_CHARS,
                )
                ocr_text = _ocr_single_page(pdf_bytes, i)
                if ocr_text and len(ocr_text.strip()) > len(raw_text):
                    logger.info(
                        "Page %d: OCR produced %d chars (was %d)",
                        i, len(ocr_text), len(raw_text),
                    )
                    raw_text = ocr_text
                    used_ocr = True
                    ocr_used = True
                else:
                    logger.info("Page %d: OCR returned no improvement; keeping text-layer result", i)
            else:
                source = "text layer" if len(raw_text) >= _MIN_TEXT_CHARS else "text layer (thin, OCR disabled)"
                logger.debug("Page %d: %s (%d chars)", i, source, len(raw_text))

            pages.append((i, raw_text))
            page_meta.append({"page": i, "text": raw_text, "ocr": used_ocr})

    if enable_ocr and cache_key:
        cache_set("ocr_text", cache_key, page_meta, ttl=60 * 60 * 24 * 7)

    return pages, ocr_used


def extract_text_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return list of (1-indexed page_num, page_text) from a PDF.

    For backward compatibility with tests: no OCR, no caching.
    Use _extract_text_with_ocr() in production (called via enrich_labeling).
    """
    pages, _ = _extract_text_with_ocr(pdf_bytes, cache_key="", enable_ocr=False)
    return pages


def is_scanned(pages: list[tuple[int, str]]) -> bool:
    total_chars = sum(len(t) for _, t in pages)
    return total_chars < _MIN_TEXT_CHARS * max(len(pages), 1)


# ── Section finders ───────────────────────────────────────────────────────────

def _find_section(
    pages: list[tuple[int, str]],
    start_markers: list[str],
    end_markers: list[str],
) -> Optional[tuple[int, str]]:
    """Find the first section matching any start_marker, return (page_num, text)."""
    collecting = False
    collected: list[str] = []
    start_page = 0

    for page_num, text in pages:
        for line in text.split("\n"):
            stripped = line.strip()
            if not collecting:
                for marker in start_markers:
                    if re.search(marker, stripped, re.IGNORECASE):
                        collecting = True
                        start_page = page_num
                        collected.append(stripped)
                        break
            else:
                for end_marker in end_markers:
                    if re.search(end_marker, stripped, re.IGNORECASE):
                        return start_page, "\n".join(collected)
                collected.append(stripped)

    if collected:
        return start_page, "\n".join(collected)
    return None


def _find_in_section(section_text: str, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, section_text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return m.group(1).strip()
            except IndexError:
                return m.group(0).strip()
    return None


# ── Strength normalizer ───────────────────────────────────────────────────────

def _normalize_strength(raw: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(.*)", raw.strip())
    if not m:
        return raw.strip()
    num_str = m.group(1)
    unit = m.group(2).strip()
    if "." in num_str:
        num_str = num_str.rstrip("0").rstrip(".")
    return f"{num_str} {unit}".strip() if unit else num_str


# ── Stage 3: LLM provider extraction ─────────────────────────────────────────
#
# In-flight dedup: N DINs sharing one PM fire one provider call, not N.
# Still valuable with any non-null provider; no-op overhead with NullProvider.
_LLM_INFLIGHT: dict[str, "asyncio.Future[dict]"] = {}
_LLM_INFLIGHT_LOCK = asyncio.Lock()


async def _query_provider_cached(section_text: str, page_num: int, field_group: str) -> dict:
    """Query configured LLM provider with persistent cache + in-flight dedup.

    Two-level dedup:
    1. Persistent SQLite cache (cross-run, TTL 7 days).
    2. In-flight asyncio Future (within-run): concurrent DINs sharing identical
       PM text wait for the first caller's result.

    With NullProvider (default), the provider returns {} instantly so this
    function returns {} on every call — no network, no cache writes.
    """
    provider = get_llm_provider()
    key = f"{field_group}:{hashlib.sha256(section_text[:5000].encode()).hexdigest()}"

    # Level 1: persistent cache hit
    cached = cache_get("llm_result", key)
    if cached is not None:
        return cached

    # Level 2: in-flight dedup
    async with _LLM_INFLIGHT_LOCK:
        if key in _LLM_INFLIGHT:
            fut: asyncio.Future[dict] = _LLM_INFLIGHT[key]
            is_leader = False
        else:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            _LLM_INFLIGHT[key] = fut
            is_leader = True

    if not is_leader:
        return await asyncio.shield(fut)

    try:
        result = await provider.extract_appearance_fields(section_text, page_num, field_group)
        if result:
            cache_set("llm_result", key, result, ttl=60 * 60 * 24 * 7)
        fut.set_result(result)
        return result
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        async with _LLM_INFLIGHT_LOCK:
            _LLM_INFLIGHT.pop(key, None)


def _apply_provider_result(
    row: dict,
    provider_out: dict,
    fields: list[str],
    fallback_page: Optional[int],
) -> None:
    """Write provider extraction results into row dict, honouring NOT_IN_PM sentinel."""
    for field in fields:
        if field not in provider_out:
            continue
        entry = provider_out[field]
        if not isinstance(entry, dict):
            continue
        val = entry.get("value")
        found = entry.get("found", False)
        page = entry.get("page") or fallback_page

        if found and val:
            row[field] = str(val).strip()
            row[f"{field}_page"] = page
        else:
            row[field] = NOT_IN_PM
            row[f"{field}_page"] = None


# ── Stage 3: regex fallback ───────────────────────────────────────────────────

_COLOR_WORDS = (
    r"(?:light |pale |dark |bright |deep |off[-\s]?)?(?:white|red|pink|orange|amber|yellow|gold(?:en)?|green|"
    r"blue|purple|violet|brown|beige|grey|gray|black|tan|teal|maroon|ivory|"
    r"peach|coral|lavender|lilac|rose|silver|salmon|olive|turquoise|aqua|indigo)\b"
)
_SHAPE_WORDS = (
    r"\b(?:round|oval(?:oid)?|oblong|capsule[- ]?shaped|caplet|biconvex|"
    r"pentagonal|hexagonal|octagonal|triangular|diamond|shield|kidney[-\s]shaped)\b"
)
_SIZE_PAT = r"(\d+(?:\.\d+)?\s*mm(?:\s*[×xX]\s*\d+(?:\.\d+)?\s*mm)?)"
_WEIGHT_PAT = (
    r"(?:[Tt]ablet\s+[Ww]eight|[Ww]eight\s+of\s+(?:the\s+)?[Tt]ablet|[Tt]otal\s+[Tt]ablet\s+[Ww]eight)"
    r"\s*[:\-]?\s*(\d+(?:\.\d+)?\s*mg)"
)

# Color is only valid when it appears near a dosage-form or appearance vocabulary word.
# This prevents false matches from "white bottle", "white paper", "printed on white", etc.
# Covers: solid oral (tablet/capsule/caplet), topical (gel/cream/ointment/lotion/emulsion),
# parenteral (injection/ampoule/vial), liquids (syrup/suspension/solution/drops),
# and generic descriptors (appearance/description/each/supplied).
_APPEARANCE_CONTEXT = re.compile(
    r"\b(?:tablets?|capsules?|caplets?|gelatin|softgels?|films?|coated|pellets?|granules?|"
    r"lozenges?|suppositories?|injections?|infusions?|ampoules?|vials?|syrups?|"
    r"suspensions?|solutions?|drops|creams?|gels?|ointments?|lotions?|emulsions?|"
    r"pastes?|patches?|inserts?|sprays?|aerosols?|"
    r"topical|ophthalmic|nasal|transdermal|"
    r"each|appearance|description|supplied|colou?rs?)\b",
    re.IGNORECASE,
)

_POISON_PATTERNS = re.compile(
    r"administration\s+strength|strength\s+and\s+dosage|recommended\s+dose|"
    r"how\s+to\s+use|administration\s+and\s+dosage",
    re.IGNORECASE,
)


def _is_poisoned(value: Optional[str]) -> bool:
    if not value:
        return False
    return bool(_POISON_PATTERNS.search(value))


def _extract_active_ingredient_regex(s6_text: str) -> Optional[str]:
    patterns = [
        r"(?:^|\n)[ \t]*Active\s+Ingredient[s]?\s*[:\-][ \t]*(.+?)(?:\n|$)",
        r"(?:contains?|containing)\s+(.+?)\s+as\s+(?:the\s+)?active\s+ingredient",
    ]
    return _find_in_section(s6_text, patterns)


_EXCIPIENT_EXTRA_REJECT = re.compile(
    r"debossed|[Aa]dministration\s+[Ff]orm|[Ff]orm\s*/\s*[Ss]trength|"
    r"[Aa]dministration\s+[Ss]trength|tablets?\s+with\s+['\"]|"
    r"[Ss]trength\s+and\s+[Dd]osage|[Dd]osage\s+[Ff]orm",
    re.IGNORECASE,
)


def _extract_excipients_regex(s6_text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (core_excipients, coating_excipients).

    Key rules:
    1. Core: match only known non-medicinal ingredient headers (Core tablets: prefix
       handled — we strip it to get just the ingredient list).
    2. Coating: MUST start with 'Film Coat' (two words) or 'Film Coating', or a bare
       'Coating:' (colon required). 'Coat' alone is rejected (ambiguous dosage headings).
    3. Hard-reject patterns guard against table-header bleeding (_POISON_PATTERNS) and
       appearance wording (debossed, Form/Strength, etc.) via _EXCIPIENT_EXTRA_REJECT.
    4. Uncoated: when core is found but no coating section exists → coating = "N/A (uncoated)".
    """
    core_pats = [
        # "Core tablets:" sub-label — capture only the ingredient list after it
        r"(?:Non-?[Mm]edicinal\s+[Ii]ngredients?|[Ii]nactive\s+[Ii]ngredients?)"
        r"\s*[:\-]?\s*(?:[Cc]ore\s+[Tt]ablets?\s*[:\-]\s*)(.+?)"
        r"(?:\n(?:[A-Z][a-z]{2,}|[A-Z]{3,})\s*[:\n]|\n\n|[Ff]ilm\s+[Cc]oat|[Cc]oating|$)",
        # Generic non-medicinal ingredients list (no "Core tablets:" prefix)
        r"(?:Non-?[Mm]edicinal\s+[Ii]ngredients?|[Ii]nactive\s+[Ii]ngredients?|"
        r"[Cc]ore\s+[Tt]ablet\s+[Ii]ngredients?|[Cc]ore\s+[Ii]ngredients?)"
        r"\s*[:\-]?\s*(.+?)"
        r"(?:\n(?:[A-Z][a-z]{2,}|[A-Z]{3,})\s*[:\n]|\n\n|[Ff]ilm\s+[Cc]oat|$)",
    ]
    core = _find_in_section(s6_text, core_pats)
    # Clean "Core tablets:" prefix if it leaked into the captured group
    if core and re.match(r"^[Cc]ore\s+[Tt]ablets?\s*[:\-]\s*", core):
        core = re.sub(r"^[Cc]ore\s+[Tt]ablets?\s*[:\-]\s*", "", core).strip()
    if _is_poisoned(core) or (core and _EXCIPIENT_EXTRA_REJECT.search(core)):
        logger.warning("excipients_core regex value discarded (poison guard): %r", core)
        core = None

    coat_pats = [
        # "Film Coat[ing]:" — two-word form, safest
        r"[Ff]ilm\s+[Cc]oat(?:ing)?\s*[:\-]?\s*(.+?)"
        r"(?:\n(?:[A-Z][a-z]{2,}|Administration|Strength|Dose|Dosage|Packaging|Storage)\s*[:\n]|\n\n|$)",
        # Bare "Coating:" (colon required to distinguish from "coat" in dosage phrases)
        r"(?:^|\n)\s*[Cc]oating\s*:\s*(.+?)"
        r"(?:\n(?:[A-Z][a-z]{2,}|Administration|Strength|Dose|Dosage|Packaging|Storage)\s*[:\n]|\n\n|$)",
    ]
    coating = _find_in_section(s6_text, coat_pats)
    if _is_poisoned(coating) or (coating and _EXCIPIENT_EXTRA_REJECT.search(coating)):
        logger.warning("excipients_coating regex value discarded (poison guard): %r", coating)
        coating = None

    # If the core list was found but no coating sub-section exists, the tablet is uncoated.
    if core and not coating:
        coating = "N/A (uncoated)"

    return core, coating


_PACK_STYLE_HEADING_REJECT = re.compile(
    r"the following|dosage strengths|dosage\s+form",
    re.IGNORECASE,
)

# A captured block is also rejected when any line ends with ":" (heading fragment)
_TRAILING_COLON_RE = re.compile(r":\s*$", re.MULTILINE)

# ── General packaging sentence scanner ───────────────────────────────────────
# Instead of enumerating every possible packaging section format, we scan the
# text for sentences that LOOK like packaging info:
#   - contains a container keyword (bottles, vials, blisters, strips, …)
#   - contains a quantity number that is NOT a drug-strength unit (mg/mcg/g/mEq)
#   - is not a clinical-trial or pharmacokinetic context
#
# This handles any PM layout without needing a format-specific pattern.

_PACK_SCAN_CONTAINER_RE = re.compile(
    r"\b(?:bottles?|vials?|blisters?|blister\s+packs?|ampoules?|ampuls?|ampules?|"
    r"syringes?|prefilled?\s+syringes?|cartridges?|auto.?injectors?|"
    r"unit\s*[–-]?\s*dose\s+strips?|strips?|sachets?|tubes?|cartons?|pouches?|"
    r"canisters?|inhalers?|pens?|droppers?)\b",
    re.IGNORECASE,
)

# Numbers immediately followed by drug-strength units are strengths, not counts.
_PACK_STRENGTH_NUM_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|μg|micrograms?|nanograms?|ng\b|g\b|mEq|IU|mmol)\b",
    re.IGNORECASE,
)

# Sentences that belong to clinical/PK/toxicology context, not packaging.
_PACK_CLINICAL_RE = re.compile(
    r"\b(?:patients?|subjects?|participants?|volunteers?|"
    r"study|studies|clinical\s+trial|placebo|randomized|n\s*=\s*\d|"
    r"adverse|efficacy|administered\s+to|half.?life|bioavailability|"
    r"pharmacokinetic|absorption|distribution|elimination)\b",
    re.IGNORECASE,
)

# Sentences describing storage/stability conditions look like packaging sentences
# (they mention container types and numbers) but are not.  Reject them explicitly.
_PACK_STORAGE_RE = re.compile(
    r"\b(?:after\s+opening|discard(?:ed)?\b|days?\s+after|hours?\s+after|"
    r"store(?:d)?\s+at\b|refrigerat|protect\s+from\s+(?:light|heat|moisture)|"
    r"keep\s+out\s+of\s+reach|expir(?:y|ation)|shelf\s*life|"
    r"should\s+be\s+(?:discarded|stored)|to\s+ensure\s+sterility)\b",
    re.IGNORECASE,
)


def _scan_for_pack_sentence(text: str) -> Optional[tuple[Optional[str], str]]:
    """Content-first packaging scanner: no section format assumed.

    Splits text into sentences and finds the first one that contains both a
    container keyword and a non-strength quantity number.  Returns
    (pack_size_text, container_label) or None.

    pack_size_text is verbatim from the PM:
      - "in bottles of 100, 500 and 1000 and in unit dose strips of 100"
      - "in 60 mL and 120 mL plastic bottles"
      - "in aluminium PVC/PCTFE blisters; 56-tablet carton"
    """
    # Split on sentence-ending punctuation only — not bare newlines.
    # Splitting on \n breaks multi-line packaging sentences like:
    #   "available in the following\ncontainers: 500g jars; 20g and 50g tubes."
    # into two fragments, causing both to fail individual checks.
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if len(sent) < 10:
            continue
        if not _PACK_SCAN_CONTAINER_RE.search(sent):
            continue
        if _PACK_CLINICAL_RE.search(sent):
            continue
        # Reject storage/stability sentences: they mention containers + numbers
        # but are describing shelf-life conditions, not pack sizes.
        if _PACK_STORAGE_RE.search(sent):
            continue
        # Must have a number that survives stripping out strength values
        without_strengths = _PACK_STRENGTH_NUM_RE.sub("", sent)
        has_count = bool(re.search(r"\b\d+\b", without_strengths))
        # Second chance: explicit container-enumeration language ("containers:",
        # "available in", "supplied in") — handles weight-based container sizes
        # like "500g jars; 20g and 50g tubes" where all numbers are stripped
        # as gram-strength units yet the sentence is clearly about packaging.
        has_container_context = bool(re.search(
            r"\bcontainers?\b|\bavailable\s+in\b|\bsupplied\s+in\b|\bpackaged\s+in\b",
            sent, re.IGNORECASE,
        ))
        if not has_count and not has_container_context:
            continue
        label = _extract_pack_style_from_text(sent)
        if not label:
            continue
        # Extract the informative slice: "in …" after "available/provided/supplied/…"
        av = re.search(
            r"\b(?:available|provided|supplied|packaged|sold|dispensed)\s+(in\s+\S.+)",
            sent, re.IGNORECASE,
        )
        size_text: Optional[str] = av.group(1).strip() if av else re.sub(r"\s+", " ", sent.strip())
        if size_text and len(size_text) > 250:
            size_text = size_text[:250].rsplit(" ", 1)[0]
        return size_text, label
    return None


def _extract_packaging_from_pdf(s6_text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract (pack_size, pack_style) from §6 text.

    Two-tier strategy — no enumerating format variants:

    Tier 1 — explicit Packaging heading (reliable when present):
      Matches "Packaging" or "Packaging:" as a standalone section header and
      extracts the block that follows.  pack_size is parsed by
      _extract_pack_size_from_product_info (good for terse DPD-style text).
      Reject rules:
        (a) "the following" / "dosage strengths" / "dosage form" → heading fragment.
        (b) any line ends with ":" → heading fragment.
        (c) no container vocabulary keyword → not a packaging block.

    Tier 2 — content-first sentence scan (format-agnostic fallback):
      Finds the first sentence in the text that contains a container keyword AND
      a quantity number that is not a drug-strength unit (mg, mcg, etc.).
      Works for "Available in bottles of 100…", "Supplied in 2 g tubes…",
      "Each blister contains 10 tablets.", and any future layout.
      Stage 2 (DPD API) values override both tiers in enrich_labeling.
    """
    # ── Tier 1: explicit Packaging section header ─────────────────────────────
    for pat in [
        r"(?m)^Packaging\s*$\n(.+?)(?=\n\n|\n[A-Z\d]|\Z)",
        r"(?m)^Packaging\s*:\s*(.+?)(?=\n\n|\n[A-Z\d]|\Z)",
    ]:
        m = re.search(pat, s6_text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        val = m.group(1).strip()
        if len(val) <= 5:
            continue
        if _PACK_STYLE_HEADING_REJECT.search(val):
            continue
        if _TRAILING_COLON_RE.search(val):
            continue
        label = _extract_pack_style_from_text(val)
        if not label:
            continue
        pack_size = _extract_pack_size_from_product_info(val)
        return pack_size, label

    # ── Tier 2: general sentence scan — no format assumed ─────────────────────
    result = _scan_for_pack_sentence(s6_text)
    if result:
        return result

    return None, None


def _extract_pack_style_from_pdf(s6_text: str) -> Optional[str]:
    """Thin wrapper kept for backward compatibility with tests."""
    _, style = _extract_packaging_from_pdf(s6_text)
    return style


_KNOWN_PRESERVATIVES_RE = re.compile(
    r"\b(?:methylparaben|propylparaben|benzyl\s+alcohol|benzalkonium\s+chloride|"
    r"sodium\s+benzoate|potassium\s+benzoate|benzoic\s+acid|sorbic\s+acid|"
    r"phenoxyethanol|phenol|cresol|chlorobutanol|thimerosal)\b",
    re.IGNORECASE,
)

# ── Non-medicinal ingredients verbatim extraction ─────────────────────────────
# Flexible heading match: "Non-medicinal Ingredients", "Nonmedicinal Ingredients",
# "Clinically Relevant Non-medicinal Ingredients", "Nonmedical Ingredients".
#
# IMPORTANT: Must be anchored to line-start ((?:^|\n)[ \t]*).  Without the anchor
# the pattern matches "nonmedicinal ingredients" embedded mid-sentence in the
# patient-information leaflet heading "What the nonmedicinal ingredients are:",
# causing the entire PIL column to be captured as the ingredient list.
_NM_HEADING_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:Clinically\s+Relevant\s+)?Non[-\s]?medicinal\s+Ingredients?"
    r"|(?:^|\n)[ \t]*Nonmedical\s+Ingredients?",
    re.IGNORECASE,
)

# No-line-start variant used ONLY for in-place substitution in _extract_text_async.
# The table column extractor injects the clean NM list back into the raw page text,
# which may have the heading mid-line due to column interleaving.
_NM_HEADING_ANYWHERE_RE = re.compile(
    r"(?:Clinically\s+Relevant\s+)?Non[-\s]?medicinal\s+Ingredients?"
    r"|Nonmedical\s+Ingredients?",
    re.IGNORECASE,
)

# Phrases that appear in patient-information leaflets (PIL) but never in the
# professional PM composition list.  If the captured NM block contains any of
# these, the extraction is contaminated by a parallel PIL column.
_PIL_CONTAMINATION_RE = re.compile(
    r"\b(?:this\s+leaflet|contact\s+your\s+(?:doctor|pharmacist)|side\s+effect|"
    r"how\s+to\s+(?:store|take|use)\b|missed\s+dose|overdose|usual\s+(?:adult\s+)?dose|"
    r"what\s+the\s+medication|what\s+it\s+does|when\s+it\s+should\s+not)\b",
    re.IGNORECASE,
)

# Lines signalling end of nonmedicinal block (Coating/Film Coat are NOT end markers —
# they are sub-headers within the list).
_NM_END_RE = re.compile(
    r"^(?:"
    r"Packaging|Storage|Stability|Shelf\s+Life|Description\b|"
    r"Administration\b|Route\b|Availability\b|Microbiology\b|"
    r"PART\s+(?:I{1,3}|IV|VI{0,3}|IX|X)\b|"  # PART II, PART III, PART IV …
    r"\d+\s+(?:DOSAGE|PHARMACEUTICAL|CLINICAL|NON-CLINICAL|WARNINGS|ADVERSE)"
    r")",
    re.IGNORECASE,
)

# Patterns that locate a non-medicinal ingredient list in §6 text (legacy, kept for
# classify_preservatives which is still used internally)
_NM_INGREDIENT_LIST_RE = re.compile(
    r"(?:Non-?[Mm]edicinal|[Ii]nactive|[Ee]xcipient)\s*(?:[Ii]ngredients?|[Ee]xcipients?)?"
    r"\s*[:\-]?\s*"
    r"(?:Core\s+[Tt]ablets?\s*:\s*)?"  # optional "Core tablets:" prefix
    r"(.{20,}?)(?:\n\n|Film\s+[Cc]oat|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _classify_preservatives(s6_text: str) -> str:
    """Return 'Y', 'N', or NOT_IN_PM based on the non-medicinal ingredient list.

    'Y'       — a known preservative is present in the found composition list.
    'N'       — a composition list was found but contains no known preservative.
    NOT_IN_PM — no composition list found at all.
    """
    # First try an explicit "Preservatives:" label
    explicit = re.search(r"[Pp]reservative[s]?\s*[:\-]\s*(.+?)(?:\n|;|$)", s6_text)
    if explicit:
        val = explicit.group(1).strip()
        if re.search(r"\b(?:none|nil|not applicable|n/a)\b", val, re.IGNORECASE):
            return "N"
        if _KNOWN_PRESERVATIVES_RE.search(val):
            return "Y"

    # Fall back to scanning the whole non-medicinal ingredient list
    nm_match = _NM_INGREDIENT_LIST_RE.search(s6_text)
    if not nm_match:
        return NOT_IN_PM  # no composition info — cannot determine

    ingredient_block = nm_match.group(0)  # includes the header line
    if _KNOWN_PRESERVATIVES_RE.search(ingredient_block):
        return "Y"
    return "N"


def _extract_nm_from_table_column(page: Any) -> Optional[str]:
    """Extract Non-medicinal Ingredients from a pdfplumber table column layout.

    Some Canadian PMs present §6 as a multi-column table:
      | Route | Dosage Form / Strength | Non-medicinal Ingredients |
    pdfplumber's extract_text() interleaves columns, producing garbage.
    This function finds the NM column and returns clean cell content.
    Returns None if no NM column is found.
    """
    try:
        tables = page.extract_tables()
        for table in tables:
            if not table or len(table) < 2:
                continue
            header_row = table[0]
            for col_idx, header_cell in enumerate(header_row):
                if header_cell and _NM_HEADING_RE.search(str(header_cell)):
                    parts: list[str] = []
                    for data_row in table[1:]:
                        if col_idx < len(data_row) and data_row[col_idx]:
                            cell = str(data_row[col_idx]).replace("\n", " ").strip()
                            if cell:
                                parts.append(cell)
                    if parts:
                        return "; ".join(parts)
    except Exception:
        pass
    return None


def _extract_nonmedicinal_ingredients(s6_text: str) -> Optional[str]:
    """Copy the Non-medicinal Ingredients list VERBATIM from §6 text.

    Three layouts handled (priority order):
    1. Inline: heading immediately followed by ": content" on same line.
    2. Block:  heading alone on one line, content on subsequent lines.
    3. "Each gram/mL/unit of X contains:" — common in topical/parenteral PMs
       that list all ingredients (active + inactive) together without a separate
       non-medicinal heading.

    PIL contamination guard: if the captured block contains patient-leaflet phrases
    (e.g. "this leaflet", "contact your doctor"), the column-merge has interleaved
    the consumer section with the professional PM.  Such results are discarded and
    the next layout is tried.

    Returns text exactly as it appears in the PDF — no normalisation.
    """
    def _collect_after(text_after: str) -> Optional[str]:
        # pdfplumber sometimes embeds page-number markers inline:
        # "...D&C yellow No. 10. - 11 - PART II: SCIENTIFIC INFORMATION..."
        # Replace these with newlines so the split below sees them as line breaks.
        text_after = re.sub(r"\s+-\s*\d+\s*-\s+", "\n", text_after)
        lines = text_after.split("\n")
        collected: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if collected:
                    break
                continue
            # Stop on section headings — even the very first line.  Without this guard,
            # a table column header like "Non-medicinal Ingredients" (no colon) would
            # match _NM_HEADING_RE and then the next table column "Administration
            # Strength/Composition" would be captured before the stop check fires.
            if _NM_END_RE.match(stripped):
                break
            collected.append(stripped)
        return " ".join(collected).strip() if collected else None

    def _clean_or_none(result: Optional[str]) -> Optional[str]:
        """Return result unless it is contaminated by PIL text."""
        if not result:
            return None
        if _PIL_CONTAMINATION_RE.search(result):
            logger.debug("NM ingredients: PIL contamination detected — discarding (%d chars)", len(result))
            return None
        return result

    # Priority 1: inline format — heading + ": content" on same line
    # Use line-start anchor so "What the nonmedicinal ingredients are:" (PIL) is skipped.
    inline_re = re.compile(
        r"(?:^|\n)[ \t]*(?:Clinically\s+Relevant\s+)?Non[-\s]?medicinal\s+Ingredients?\s*:\s*"
        r"|(?:^|\n)[ \t]*Nonmedical\s+Ingredients?\s*:\s*",
        re.IGNORECASE,
    )
    for m in inline_re.finditer(s6_text):
        after = s6_text[m.end():]
        result = _clean_or_none(_collect_after(after))
        if result:
            return result

    # Priority 1b: "non-medicinal excipients:" embedded in a sentence — old Health Canada PM
    # format where the heading reads "...contains the following non-medicinal excipients: ..."
    # No line-start anchor needed; this phrasing doesn't appear in PIL sections.
    excipients_re = re.compile(
        r"non[-\s]?medicinal\s+excipients?\s*:\s*",
        re.IGNORECASE,
    )
    for m1b in excipients_re.finditer(s6_text):
        after = s6_text[m1b.end():]
        result = _clean_or_none(_collect_after(after))
        if result:
            return result

    # Priority 1c: "Composition: [product name] contains: [NM list]" — old Health Canada PM
    # format where the Composition section embeds the excipient list after "contains:".
    # e.g. "Composition: TEVA-KETOCONAZOLE (ketoconazole) contains: pregelatinized starch, ..."
    # LINE-START anchor is mandatory: prevents matching "Administration Strength/Composition"
    # table headings where "Composition" is embedded mid-line and "contains" appears later in
    # the active-ingredient sentence (which would capture the wrong block).
    composition_contains_re = re.compile(
        r"(?:^|\n)\s*Composition\s*[:\-]\s*[^\n]{0,120}\bcontains?\s*[:\-]\s*",
        re.IGNORECASE,
    )
    m1c = composition_contains_re.search(s6_text)
    if m1c:
        after = s6_text[m1c.end():]
        result = _clean_or_none(_collect_after(after))
        if result:
            return result

    # Priority 2: block format — heading on its own line (line-start anchor guards PIL).
    m = _NM_HEADING_RE.search(s6_text)
    if m:
        after = re.sub(r"^\s*[:\-]?\s*", "", s6_text[m.end():])
        result = _clean_or_none(_collect_after(after))
        if result:
            return result

    # Priority 3: "Each gram/mL/unit of X contains:" format — topical/parenteral PMs
    # that list all ingredients (active + non-medicinal) together.
    each_re = re.compile(
        r"[Ee]ach\s+(?:gram|g\b|mL|ml|litre|liter|tablet|capsule|unit|vial|ampoule|sachet)"
        r"(?:\s+of\s+\S+(?:\s+\S+){0,3})?"
        r"\s+contains?\s*[:\-]?\s*",
        re.IGNORECASE,
    )
    m3 = each_re.search(s6_text)
    if m3:
        after = s6_text[m3.end():]
        result = _clean_or_none(_collect_after(after))
        if result:
            return result

    return None


def _extract_ph_regex(s13_text: str) -> str:
    conc_entries = re.findall(
        r"pH\s+\d+(?:\.\d+)?\s*[:\-]\s*(?:<|>)?[\d.]+\s*(?:mg|µg|ug)/",
        s13_text, re.IGNORECASE,
    )
    has_ph_dep = bool(re.search(
        r"pH[-\s]dependent|solubility[^\n]+pH[-\s]dependent|pH[-\s]dependent[^\n]+solubility",
        s13_text, re.IGNORECASE,
    ))
    if len(conc_entries) >= 2 or (has_ph_dep and conc_entries):
        return PH_SOLUBILITY_ONLY

    single_pats = [
        r"(?:^|\n)[ \t]*pH(?:\s+range)?\s*[:\-]\s*(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s*(?:\n|$)",
        r"\bpH\b[^\n]*?[:\-]\s*(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)(?!\s*[:\-]\s*[\d<>])",
    ]
    val = _find_in_section(s13_text, single_pats)
    if val:
        val = val.strip()
        # Validate: all numeric parts must be physically possible pH (0–14).
        # OCR often drops decimal points (e.g. "6.7" → "67"), which this rejects.
        nums = re.findall(r'\d+(?:\.\d+)?', val)
        if nums and all(0.0 <= float(n) <= 14.0 for n in nums):
            return val
        logger.warning("pH value out of valid range 0-14: %r — discarded", val)
    return NOT_IN_PM


def _extract_appearance_regex(
    desc_text: str,
    target_strength: Optional[str],
) -> dict[str, Optional[str]]:
    """Extract color/shape/size/weight, scoped to target_strength block if possible."""
    out: dict[str, Optional[str]] = {"color": None, "shape": None, "size_mm": None, "weight": None}

    block_text = desc_text
    if target_strength:
        norm = _normalize_strength(target_strength)
        strength_pat = re.escape(norm).replace(r"\ ", r"\s*")
        block_re = re.compile(
            r"(?:^|[•\-*◦–—])\s*" + strength_pat + r"\s*[:\-]?\s*(.+?)"
            r"(?=(?:[•\-*◦–—]\s*\d|\n\n|\Z))",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        m = block_re.search(desc_text)
        if m:
            block_text = m.group(1)
        # If no bulleted strength block found, use full description text.
        # A single-line fallback is too narrow: color/shape words often appear
        # on a different line than the strength (e.g. "amber-colored oblong capsule\n
        # each containing icosapent ethyl 1 g").

    # Color: iterate all matches; take the first one that has a dosage-form context word
    # within 200 chars. Using finditer instead of search prevents the first color word
    # in a non-appearance context (e.g. "yellow jaundice", "white blood cell") from
    # blocking the correct match that appears later in the text (e.g. "amber gel").
    for cm in re.finditer(_COLOR_WORDS, block_text, re.IGNORECASE):
        pos = cm.start()
        window = block_text[max(0, pos - 200): pos + 200]
        if not _APPEARANCE_CONTEXT.search(window):
            continue
        # Skip drug-substance descriptions: "white to slightly beige coloured powder"
        # "powder" or "crystalline" within 120 chars after the match signals drug-substance
        after_match = block_text[pos: pos + 120]
        if re.search(r"\bpowder\b|\bcrystalline\b", after_match, re.IGNORECASE):
            continue
        color = cm.group(0).strip()
        # Extend to compound color phrases: "white to slightly grey", "light pink to pale red"
        _ext = re.match(
            r"\s+to\s+(?:slightly\s+|almost\s+|very\s+|pale\s+|light\s+|dark\s+|off-?)?"
            r"(?:white|red|pink|orange|amber|yellow|gold(?:en)?|green|blue|purple|violet|"
            r"brown|beige|grey|gray|black|cream|tan|ivory|maroon)\b",
            block_text[pos + len(cm.group(0)):],
            re.IGNORECASE,
        )
        if _ext:
            color = (color + _ext.group(0)).strip()
        out["color"] = color
        break

    # Shape: also apply context check to avoid false matches from medical terminology
    # (e.g. "kidney" in "kidney infection", "diamond" in drug names).
    for sm in re.finditer(_SHAPE_WORDS, block_text, re.IGNORECASE):
        pos = sm.start()
        window = block_text[max(0, pos - 200): pos + 200]
        if _APPEARANCE_CONTEXT.search(window):
            out["shape"] = sm.group(0).strip()
            break

    szm = re.search(_SIZE_PAT, block_text, re.IGNORECASE)
    if szm:
        out["size_mm"] = szm.group(1).strip()

    wm = re.search(_WEIGHT_PAT, block_text, re.IGNORECASE)
    if wm:
        out["weight"] = wm.group(1).strip()

    return out


# ── §6 / §13 markers ─────────────────────────────────────────────────────────

_S6_MARKERS = [
    # Matches "6 DOSAGE FORMS, COMPOSITION..." and "6. Dosage Forms, Strengths, Composition..."
    # Negative lookahead rejects TOC entries which have dot leaders like "..........9"
    r"^6[\.\s]+DOSAGE FORMS?(?!.*\.{4,}).*COMPOSITION",
    r"^6[\.\s]+PHARMACEUTICAL INFORMATION(?!.*\.{4,})",
    r"^PART II.*SCIENTIFIC INFORMATION(?!.*\.{4,})",
    # Older/generic PMs use unnumbered heading: "DOSAGE FORMS, COMPOSITION AND PACKAGING"
    r"^DOSAGE FORMS?,\s*(?:STRENGTHS?,\s*)?COMPOSITION(?!.*\.{4,})",
]
_S6_END = [r"^7[\.\s]+", r"^PART\s+III", r"^CLINICAL\s+PHARMACOLOGY"]

_S13_MARKERS = [
    r"^13[\.\s]+PHARMACEUTICAL INFORMATION",
    r"^PHARMACEUTICAL INFORMATION(?!.*\.{4,})",
]
_S13_END = [r"^14[\.\s]+", r"^NON-CLINICAL", r"^TOXICOLOGY"]

_DESC_MARKERS = [r"[Dd]escription", r"[Pp]hysical\s+[Dd]escription"]
_DESC_END = [r"[Cc]omposition", r"[Pp]ackaging", r"[Ss]torage", r"MICROBIOLOGY", r"\n\n\n"]


# ── Main extraction entry point ───────────────────────────────────────────────

async def parse_labeling_fields_async(
    pages: list[tuple[int, str]],
    din_strength: Optional[str],
) -> dict:
    """Extract Stage 3 label fields from pre-extracted PDF pages.

    Only extracts: excipients_core, excipients_coating, preservatives,
                   ph, color, shape, size_mm, weight.
    Does NOT extract: active_ingredient, pack_size, pack_style
                      (those come from Stage 2 / DPD API).

    Extraction strategy (per field group):
      1. Ask configured LLM provider (no-op with NullProvider — the default).
      2. If provider returns {}, use regex deterministic path.

    cite-or-blank rule is enforced on both paths: only populate verbatim/cited
    content; leave fields NOT_IN_PM when not found — never guess.

    Returns a flat dict. Every scalar field has a companion _page field.
    """
    row: dict = {}

    # ── Full-document fallback text ───────────────────────────────────────────
    # When no section marker matches (novel PM template), we fall back to
    # searching the entire document.  Individual extractors each have their own
    # keyword guards (explicit headings, context proximity checks, range
    # validation) so searching the full text is safe — they will still return
    # NOT_IN_PM for fields that are genuinely absent.
    #
    # TOC avoidance: TOC pages are almost always the first 1–2 pages.  In full-
    # doc fallback mode we skip them for the Description subsection search so
    # "Description.....5" in the TOC does not win over the actual section body.
    full_doc_text = "\n".join(text for _, text in pages)
    full_doc_page = pages[0][0] if pages else 1
    # Pages with more than 40 % of lines ending in dots are treated as TOC pages.
    _dot_re = re.compile(r"\.{4,}\s*\d*\s*$")
    non_toc_pages = [
        (pn, txt) for pn, txt in pages
        if not (
            txt.strip()
            and sum(1 for ln in txt.splitlines() if _dot_re.search(ln))
            / max(len(txt.splitlines()), 1) > 0.4
        )
    ] or pages

    s6 = _find_section(pages, _S6_MARKERS, _S6_END)
    s6_page = s6[0] if s6 else None
    s6_text = s6[1] if s6 else ""
    s6_fallback = not bool(s6_text)
    if s6_fallback:
        # No known §6 template matched — search the full document.
        s6_text = full_doc_text
        s6_page = full_doc_page
        logger.info("No §6 marker matched — using full-document text for §6 field extraction")

    s13 = _find_section(pages, _S13_MARKERS, _S13_END)
    s13_page = s13[0] if s13 else None
    s13_text = s13[1] if s13 else ""
    if not s13_text:
        # No known §13 template matched — pH searched against full document.
        s13_text = full_doc_text
        s13_page = full_doc_page
        logger.info("No §13 marker matched — using full-document text for pH extraction")

    # ── Description subsection ────────────────────────────────────────────────
    # Search for Description subsection within §6 text to avoid matching TOC
    # entries (e.g. "Description...........5" in the table of contents triggers
    # the marker, then "Composition...........5" immediately ends it, leaving
    # desc_text as a single useless TOC line with no color/shape content).
    if not s6_fallback:
        # §6 was found via marker — search for Description inside it (TOC-safe).
        desc_in_s6 = _find_section([(s6_page or 1, s6_text)], _DESC_MARKERS, _DESC_END)
        desc_page = desc_in_s6[0] if desc_in_s6 else s6_page
        desc_text = desc_in_s6[1] if desc_in_s6 else s6_text
    else:
        # Full-doc fallback — skip TOC pages when hunting for Description.
        desc_section = _find_section(non_toc_pages, _DESC_MARKERS, _DESC_END)
        desc_page = desc_section[0] if desc_section else s6_page
        # If no Description subsection found, use full doc (color context check guards it).
        desc_text = desc_section[1] if desc_section else full_doc_text

    # ── Non-medicinal Ingredients (verbatim, always deterministic) ───────────
    nm_val = _extract_nonmedicinal_ingredients(s6_text)
    row["nonmedicinal_ingredients"] = nm_val if nm_val else NOT_IN_PM
    row["nonmedicinal_ingredients_page"] = s6_page if nm_val else None

    # Active ingredient PDF fallback (DPD API is authoritative; overridden in enrich_labeling)
    ai_pdf = _extract_active_ingredient_regex(s6_text)
    row["active_ingredient"] = ai_pdf if ai_pdf else NOT_IN_PM
    row["active_ingredient_page"] = s6_page if ai_pdf else None

    # ── Appearance ────────────────────────────────────────────────────────────
    if din_strength and desc_text:
        norm = _normalize_strength(din_strength)
        provider_app = await _query_provider_cached(
            f"Product: {norm}\n\n{desc_text}", desc_page or 1, "appearance"
        )
    else:
        provider_app = await _query_provider_cached(desc_text or s6_text, desc_page or 1, "appearance")

    if provider_app:
        _apply_provider_result(row, provider_app, ["color", "shape", "size_mm", "weight"], desc_page)
    else:
        if desc_text:
            norm_strength = _normalize_strength(din_strength) if din_strength else None
            app = _extract_appearance_regex(desc_text, norm_strength)
        else:
            app = {"color": None, "shape": None, "size_mm": None, "weight": None}

        for field in ("color", "shape", "size_mm", "weight"):
            val = app.get(field)
            row[field] = val if val else NOT_IN_PM
            row[f"{field}_page"] = desc_page if val else None

    # ── Consumer section (Part III) fallback for appearance ───────────────────
    # Some PMs (e.g. topical gels) only describe the physical appearance in the
    # consumer information section ("XOLEGEL comes in a smooth, clear amber gel")
    # while the professional section lists only dye codes.  If appearance fields
    # remain NOT_IN_PM after professional PM extraction, search Part III.
    _appearance_fields_missing = all(
        row.get(f) == NOT_IN_PM for f in ("color", "shape")
    )
    if _appearance_fields_missing and not provider_app:
        part3 = _find_section(
            pages,
            [r"^PART\s+III", r"^CONSUMER\s+INFORMATION"],
            [],  # no end marker — collect to end of document
        )
        if part3:
            part3_page, part3_text = part3
            # Use a targeted product-description sentence pattern rather than the
            # general appearance regex.  The consumer section has color words in
            # adverse-reaction text ("skin may appear red") that pass the broad
            # context check.  A product-description sentence always uses "comes in",
            # "is a", "is supplied as", or "available as" before the color.
            _prod_desc_re = re.compile(
                r"(?:comes?\s+in\s+(?:a\s+)?(?:[^\s.!?]+\s+){0,6}"
                r"|is\s+(?:supplied\s+as\s+a\s+)?(?:[^\s.!?]+\s+){0,4}"
                r"|available\s+as\s+(?:a\s+)?(?:[^\s.!?]+\s+){0,4})"
                r"(" + _COLOR_WORDS + r")",
                re.IGNORECASE,
            )
            for pm in _prod_desc_re.finditer(part3_text):
                if row.get("color") == NOT_IN_PM:
                    row["color"] = pm.group(1).strip()
                    row["color_page"] = part3_page
                    logger.info("color %r sourced from consumer section (Part III)", row["color"])
                    break

    # ── "AVAILABILITY OF DOSAGE FORMS" fallback (old-style PMs) ─────────────────
    # Old Health Canada PMs describe tablet/capsule appearance in a dedicated
    # "AVAILABILITY OF DOSAGE FORMS" section instead of a "Description" subsection
    # inside §6.  If appearance fields are still missing after the main and Part III
    # searches, try this section directly.
    if any(row.get(f) == NOT_IN_PM for f in ("color", "shape")) and not provider_app:
        avail_section = _find_section(
            non_toc_pages,
            [r"AVAILABILITY\s+OF\s+DOSAGE\s+FORMS?", r"^SUPPLIED\b"],
            [r"MICROBIOLOGY", r"CLINICAL\s+(?:PHARMACOLOGY|TRIALS)", r"PHARMACOKINETICS",
             r"TOXICOLOGY", r"REFERENCES"],
        )
        if avail_section:
            avail_page, avail_text = avail_section
            norm_s = _normalize_strength(din_strength) if din_strength else None
            app_avail = _extract_appearance_regex(avail_text, norm_s)
            for field in ("color", "shape"):
                if app_avail.get(field) and row.get(field) == NOT_IN_PM:
                    row[field] = app_avail[field]
                    row[f"{field}_page"] = avail_page
                    logger.info("%s %r sourced from AVAILABILITY OF DOSAGE FORMS section", field, row[field])

    # ── pH ────────────────────────────────────────────────────────────────────
    if s13_text:
        provider_ph = await _query_provider_cached(s13_text, s13_page or 1, "ph")
        if provider_ph:
            _apply_provider_result(row, provider_ph, ["ph"], s13_page)
        else:
            row["ph"] = _extract_ph_regex(s13_text)
            row["ph_page"] = s13_page if row["ph"] not in (NOT_IN_PM, PH_SOLUBILITY_ONLY) else None
    else:
        row["ph"] = NOT_IN_PM
        row["ph_page"] = None

    # ── pack_size + pack_style from PDF §6 (Stage 2 DPD API overrides in enrich_labeling) ─
    pdf_pack_size, pdf_pack_style = _extract_packaging_from_pdf(s6_text)
    if "pack_style" not in row:
        row["pack_style"] = pdf_pack_style if pdf_pack_style else NOT_IN_PM
        row["pack_style_page"] = s6_page if pdf_pack_style else None
    if "pack_size" not in row:
        row["pack_size"] = pdf_pack_size if pdf_pack_size else NOT_IN_PM
        row["pack_size_page"] = s6_page if pdf_pack_size else None

    row["needs_ocr"] = 0  # overridden by enrich_labeling() if OCR was used
    row["has_unverified"] = 0
    return row


def parse_labeling_fields(
    pages: list[tuple[int, str]],
    din_strength: Optional[str],
) -> dict:
    """Synchronous wrapper around parse_labeling_fields_async.

    Must only be called from a non-async context; raises RuntimeError otherwise.
    """
    return asyncio.run(parse_labeling_fields_async(pages, din_strength))


# ── Public named wrappers for test introspection ──────────────────────────────

def _extract_strength_block(description: str, target_strength: str) -> dict[str, Optional[str]]:
    """Public alias used by tests: extract color/shape/size/weight for one strength."""
    return _extract_appearance_regex(description, target_strength)


def _extract_ph(s13_text: str) -> str:
    """Public alias used by tests: extract pH from §13 text."""
    return _extract_ph_regex(s13_text)


async def enrich_labeling(
    din: str,
    drug_code: int,
    strength: Optional[str],
    pdf_bytes: Optional[bytes] = None,
    enable_ocr: Optional[bool] = None,
) -> Optional[dict]:
    """Enrich a single DIN: Stage 2 (DPD API) then Stage 3 (PDF + OCR if needed).

    If pdf_bytes is provided (e.g. from a test), skip the download.
    Returns the extracted row dict (also stored in the labeling table).
    """
    # Stage 2: DPD API
    stage2 = await fetch_stage2_data(drug_code)
    pdf_url = stage2.pop("pdf_url", None)
    stage2.pop("pdf_date", None)
    stage2.pop("description", None)

    # Download PDF (unless provided by caller)
    if pdf_bytes is None:
        if pdf_url:
            pdf_bytes = await _download_pdf(pdf_url)
        if pdf_bytes is None:
            logger.info("No PDF for drug_code=%s din=%s — storing Stage 2 data only", drug_code, din)
            row: dict = {}
            for field in _LABELING_FIELDS:
                if field in _STAGE2_FIELDS:
                    val = stage2.get(field)
                    row[field] = val if val else NO_PM_AVAILABLE
                    row[f"{field}_page"] = None
                else:
                    row[field] = NO_PM_AVAILABLE  # PM file not available for this DIN
                    row[f"{field}_page"] = None
            row["needs_ocr"] = 0
            row["has_unverified"] = 0
            row["drug_code"] = drug_code
            upsert_labeling(din, row)
            return row

    # Stage 3: PDF extraction — OCR applied per-page when text layer is thin.
    # _extract_text_async runs pdfplumber + Tesseract in a thread pool so the
    # event loop stays free for concurrent labeling tasks; it also serialises
    # concurrent callers sharing the same PDF URL via a per-URL Lock so the
    # work is done exactly once and the second caller gets an OCR-cache hit.
    ocr_flag = ENABLE_OCR if enable_ocr is None else enable_ocr
    try:
        cache_key = pdf_url or f"pdf_bytes:{din}"
        pages, ocr_used = await _extract_text_async(
            pdf_bytes, cache_key=cache_key, enable_ocr=ocr_flag,
        )
    except ImportError:
        logger.error("pdfplumber is not installed — cannot extract labeling fields.")
        return None
    except Exception as exc:
        logger.warning("PDF parse failed for din=%s: %s", din, exc)
        return None

    pdf_row = await parse_labeling_fields_async(pages, strength)

    # Mark OCR usage
    if ocr_used:
        pdf_row["needs_ocr"] = 1
        logger.info("din=%s: OCR was used for at least one page → needs_ocr=1", din)

    # Merge: Stage 2 API values are authoritative; Stage 3 fills the rest
    final_row: dict = dict(pdf_row)
    for field in _STAGE2_FIELDS:
        api_val = stage2.get(field)
        if api_val:
            final_row[field] = api_val
            final_row[f"{field}_page"] = None  # API has no page number
        else:
            logger.info(
                "Stage 2 API empty for %s din=%s drug_code=%s — PDF value kept: %r",
                field, din, drug_code, final_row.get(field),
            )

    final_row["drug_code"] = drug_code
    if pdf_url:
        final_row["pdf_url"] = pdf_url

    upsert_labeling(din, final_row)
    return final_row


async def enrich_labeling_batch(
    din_map: dict[str, tuple[int, Optional[str]]],
    enable_ocr: Optional[bool] = None,
) -> dict[str, Optional[dict]]:
    """Enrich labeling for multiple DINs. din_map: {din → (drug_code, strength)}"""
    results = await asyncio.gather(*[
        enrich_labeling(din, drug_code, strength, enable_ocr=enable_ocr)
        for din, (drug_code, strength) in din_map.items()
    ])
    return dict(zip(din_map.keys(), results))


async def _extract_text_async(
    pdf_bytes: bytes,
    cache_key: str,
    enable_ocr: bool = True,
) -> tuple[list[tuple[int, str]], bool]:
    """Thread-pool + per-URL-dedup wrapper for _extract_text_with_ocr.

    Two speedups in one:
    1. Thread pool: pdfplumber/Tesseract run in _PDF_THREAD_POOL so the event
       loop stays responsive to other concurrent labeling tasks.
    2. Per-URL serialisation: if two coroutines request the same cache_key
       concurrently, only the first runs the extraction; the second blocks on
       the per-URL Lock, then gets a fast OCR-cache hit when it runs.

    Output is byte-for-byte identical to _extract_text_with_ocr — same
    pdfplumber calls, same Tesseract invocations, same cache reads/writes.
    """
    loop = asyncio.get_running_loop()

    if not cache_key:
        # No URL key (test-injected bytes with empty key) — no dedup needed.
        return await loop.run_in_executor(
            _PDF_THREAD_POOL,
            functools.partial(_extract_text_with_ocr, pdf_bytes, cache_key, enable_ocr),
        )

    # Acquire a per-URL lock to serialise concurrent extractions for the same
    # PDF URL.  The lock is created lazily under a module-level meta-lock.
    async with _PDF_EXTRACT_LOCKS_META:
        if cache_key not in _PDF_EXTRACT_LOCKS:
            _PDF_EXTRACT_LOCKS[cache_key] = asyncio.Lock()
        url_lock = _PDF_EXTRACT_LOCKS[cache_key]

    async with url_lock:
        # _extract_text_with_ocr checks the OCR text cache at its own start;
        # a second caller will hit that cache immediately and return in <1 ms.
        return await loop.run_in_executor(
            _PDF_THREAD_POOL,
            functools.partial(_extract_text_with_ocr, pdf_bytes, cache_key, enable_ocr),
        )


async def enrich_labeling_batch_fast(
    din_map: dict[str, tuple[int, Optional[str]]],
    enable_ocr: Optional[bool] = None,
    concurrency: int = 8,
    on_progress: Optional[Callable] = None,
) -> dict[str, Optional[dict]]:
    """Fast batch labeling with two levels of deduplication.

    Speedup 1 — deduplicate by drug_code:
      Multiple DINs can share one drug_code (different strengths of the same
      product).  Stage 2 DPD API calls (active ingredient, packaging, info page)
      are made once per unique drug_code, not once per DIN.

    Speedup 2 — deduplicate by pdf_url:
      DINs sharing the same Product Monograph URL trigger one PDF download and
      one pdfplumber/OCR extraction pass.  parse_labeling_fields_async() is
      then called per-DIN with its specific strength so per-strength appearance
      fields (color, shape, size) are extracted correctly.

    Speedup 3 — thread pool (via _extract_text_async):
      PDF extraction is CPU-bound and synchronous.  Running it in the thread
      pool unblocks the event loop for concurrent PDF downloads and API calls.

    Output is identical to calling enrich_labeling() for each DIN individually —
    same Stage 2 → Stage 3 merge logic, same store writes, same sentinels.
    """
    if not din_map:
        return {}

    ocr_flag = ENABLE_OCR if enable_ocr is None else enable_ocr
    results: dict[str, Optional[dict]] = {}
    done_count = 0
    done_lock = asyncio.Lock()

    # ── Step 1: fetch Stage 2 data, deduplicating by drug_code ───────────────
    unique_drug_codes = {dc for dc, _ in din_map.values()}
    stage2_by_dc: dict[int, dict] = {}

    async def _fetch_s2(dc: int) -> None:
        stage2_by_dc[dc] = await fetch_stage2_data(dc)

    await asyncio.gather(*[_fetch_s2(dc) for dc in unique_drug_codes])

    # ── Step 2: group DINs by pdf_url ─────────────────────────────────────────
    # Key: pdf_url (str) or None for DINs with no PM.
    # Value: list of (din, drug_code, strength) triples.
    pdf_url_groups: dict[Optional[str], list[tuple[str, int, Optional[str]]]] = {}
    for din, (drug_code, strength) in din_map.items():
        pdf_url = stage2_by_dc.get(drug_code, {}).get("pdf_url")
        pdf_url_groups.setdefault(pdf_url, []).append((din, drug_code, strength))

    # ── Step 3: bounded concurrent processing, one task per unique pdf_url ────
    pdf_sem = asyncio.Semaphore(concurrency)

    async def _process_group(
        pdf_url: Optional[str],
        dins_in_group: list[tuple[str, int, Optional[str]]],
    ) -> None:
        nonlocal done_count
        async with pdf_sem:
            await _process_pdf_group(pdf_url, dins_in_group)

    async def _process_pdf_group(
        pdf_url: Optional[str],
        dins_in_group: list[tuple[str, int, Optional[str]]],
    ) -> None:
        nonlocal done_count

        # ── No PM path ────────────────────────────────────────────────────────
        pdf_bytes: Optional[bytes] = None
        if pdf_url:
            pdf_bytes = await _download_pdf(pdf_url)

        if pdf_bytes is None:
            # Stage 2 data only — identical logic to the no-PM branch in enrich_labeling().
            for din, drug_code, _strength in dins_in_group:
                s2 = _stage2_fields_only(stage2_by_dc.get(drug_code, {}))
                row: dict = {}
                for field in _LABELING_FIELDS:
                    if field in _STAGE2_FIELDS:
                        val = s2.get(field)
                        row[field] = val if val else NO_PM_AVAILABLE
                        row[f"{field}_page"] = None
                    else:
                        row[field] = NO_PM_AVAILABLE
                        row[f"{field}_page"] = None
                row["needs_ocr"] = 0
                row["has_unverified"] = 0
                row["drug_code"] = drug_code
                upsert_labeling(din, row)
                results[din] = row
                await _tick_progress(din)
            return

        # ── PDF path: extract text once, parse per-DIN ────────────────────────
        cache_key = pdf_url or f"pdf_bytes:{dins_in_group[0][0]}"
        try:
            pages, ocr_used = await _extract_text_async(
                pdf_bytes, cache_key=cache_key, enable_ocr=ocr_flag,
            )
        except ImportError:
            logger.error("pdfplumber not installed — cannot extract labeling fields.")
            for din, dc, _ in dins_in_group:
                results[din] = None
                await _tick_progress(din)
            return
        except Exception as exc:
            logger.warning("PDF parse failed for url=%s: %s", pdf_url, exc)
            for din, dc, _ in dins_in_group:
                results[din] = None
                await _tick_progress(din)
            return

        # Run all DINs in this group concurrently (each needs its own strength).
        async def _parse_one(din: str, drug_code: int, strength: Optional[str]) -> None:
            s2 = _stage2_fields_only(stage2_by_dc.get(drug_code, {}))
            pdf_row = await parse_labeling_fields_async(pages, strength)
            if ocr_used:
                pdf_row["needs_ocr"] = 1
                logger.info("din=%s: OCR was used for at least one page → needs_ocr=1", din)

            # Merge: Stage 2 API values override Stage 3 PDF values (authoritative).
            final_row = dict(pdf_row)
            for field in _STAGE2_FIELDS:
                api_val = s2.get(field)
                if api_val:
                    final_row[field] = api_val
                    final_row[f"{field}_page"] = None
                else:
                    logger.info(
                        "Stage 2 API empty for %s din=%s drug_code=%s — PDF value kept: %r",
                        field, din, drug_code, final_row.get(field),
                    )
            final_row["drug_code"] = drug_code
            if pdf_url:
                final_row["pdf_url"] = pdf_url

            upsert_labeling(din, final_row)
            results[din] = final_row
            await _tick_progress(din)

        await asyncio.gather(*[_parse_one(d, dc, st) for d, dc, st in dins_in_group])

    async def _tick_progress(din: str) -> None:
        nonlocal done_count
        async with done_lock:
            done_count += 1
            dc = done_count
        if on_progress is not None:
            cb = on_progress(dc, len(din_map), din)
            if asyncio.iscoroutine(cb):
                await cb

    await asyncio.gather(*[
        _process_group(pdf_url, dins)
        for pdf_url, dins in pdf_url_groups.items()
    ])

    return results


def _stage2_fields_only(stage2: dict) -> dict:
    """Return a copy of stage2 with only the mergeable fields (drops pdf_url etc.)."""
    s2 = dict(stage2)
    s2.pop("pdf_url", None)
    s2.pop("pdf_date", None)
    s2.pop("description", None)
    return s2


async def _download_pdf(url: str) -> Optional[bytes]:
    import base64
    cache_key = f"pdf:{url}"

    # Fast path — avoid acquiring the per-URL lock if already cached.
    cached = cache_get("labeling_pdf", cache_key)
    if cached is not None:
        return base64.b64decode(cached)

    # Slow path: serialise concurrent downloads for the same URL so only
    # the first request hits the network; the rest get a cache hit on retry.
    async with _PDF_DL_LOCKS_META:
        if url not in _PDF_DL_LOCKS:
            _PDF_DL_LOCKS[url] = asyncio.Lock()
        url_lock = _PDF_DL_LOCKS[url]

    async with url_lock:
        # Re-check after acquiring the lock — another coroutine may have
        # downloaded and cached the PDF while we were waiting.
        cached = cache_get("labeling_pdf", cache_key)
        if cached is not None:
            return base64.b64decode(cached)

        try:
            client = await _get_shared_client()
            r = await client.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=60.0,
                )
            if r.status_code != 200:
                logger.warning("PDF download HTTP %d for %s", r.status_code, url)
                return None
            pdf_bytes = r.content
            cache_set("labeling_pdf", cache_key, base64.b64encode(pdf_bytes).decode(),
                      ttl=60 * 60 * 24 * 7)
            return pdf_bytes
        except Exception as exc:
            logger.warning("PDF download failed for %s: %s", url, exc)
            return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_pack_style_validation() -> None:
    """Validate Fix 1 & Fix 2: pack_style and pack_size extraction from product_information."""
    test_cases = [
        {
            "din": "02413736",
            "label": "BENLYSTA 80mg/mL",
            "product_information": "FOR I.V. INFUSION ONLY. 80MG/ML(RECONST.) - 5ML VIAL.",
            "expected_pack_style": "Vial",
            "expected_pack_size": "5 mL",
        },
        {
            "din": "02229091",
            "label": "Tablet product 24/50/100/200 count",
            "product_information": "24/50/100/200",
            "expected_pack_style": None,
            "expected_pack_size": "24, 50, 100, 200 count",
        },
        {
            "din": "02048779",
            "label": "Prefilled syringe product",
            "product_information": "1ML PREFILLED SYRINGE",
            "expected_pack_style": "Prefilled Syringe",
            "expected_pack_size": "1 mL",
        },
        {
            "din": "02248700",
            "label": "Blister pack 100/500",
            "product_information": "BLISTER PACK 100/500",
            "expected_pack_style": "Blister Pack",
            "expected_pack_size": "100, 500 count",
        },
        {
            "din": "02XXX000",
            "label": "Icosapent 1g capsule — blister + bottle",
            "product_information": "8 COUNT BLISTERS AND BOTTLES OF 120 CAPSULES",
            "expected_pack_style": "Blister; Bottle",
            "expected_pack_size": "8-count blister; 120-count bottle",
        },
        {
            "din": "02XXX001",
            "label": "Icosapent — blister only (no capsules keyword)",
            "product_information": "8 COUNT BLISTERS",
            "expected_pack_style": "Blister",
            "expected_pack_size": "8 count",
        },
    ]

    print("\n=== Fix 1 & 2 Validation: pack_style / pack_size from product_information ===\n")
    print(f"{'DIN':<12} {'product_information':<45} {'pack_style':<20} {'pack_size':<25} {'style_ok':<10} {'size_ok'}")
    print("-" * 130)

    all_ok = True
    for tc in test_cases:
        pi = tc["product_information"]
        style = _extract_pack_style_from_text(pi, "")
        size = _extract_pack_size_from_product_info(pi)

        style_ok = style == tc["expected_pack_style"]
        size_ok = size == tc["expected_pack_size"]
        if not style_ok or not size_ok:
            all_ok = False

        print(
            f"{tc['din']:<12} "
            f"{pi[:43]:<45} "
            f"{str(style):<20} "
            f"{str(size):<25} "
            f"{'✓' if style_ok else '✗ (want: ' + str(tc['expected_pack_style']) + ')':<10} "
            f"{'✓' if size_ok else '✗ (want: ' + str(tc['expected_pack_size']) + ')'}"
        )

    print()
    # Verify container words don't leak into pack_size *unintentionally*.
    # Multi-container cases (e.g. "8-count blister; 120-count bottle") deliberately
    # include the container name — skip those whose expected value already contains it.
    for tc in test_cases:
        size = _extract_pack_size_from_product_info(tc["product_information"])
        expected = tc.get("expected_pack_size") or ""
        if size:
            for _, label in _CONTAINER_VOCAB_ORDERED:
                if label.upper() in (size or "").upper():
                    if label.lower() not in expected.lower():
                        print(f"FAIL: container word '{label}' leaked into pack_size={size!r} for DIN {tc['din']}")
                        all_ok = False

    if all_ok:
        print("All Fix 1 & 2 assertions passed.")
    else:
        print("SOME ASSERTIONS FAILED — see above.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract labeling fields for a DIN.")
    parser.add_argument("--drug-code", type=int)
    parser.add_argument("--din")
    parser.add_argument("--strength", default=None, help="e.g. '50 mg'")
    parser.add_argument("--validate", action="store_true",
                        help="Run pack_style/pack_size validation demo (no network needed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.validate:
        _run_pack_style_validation()
    elif args.drug_code and args.din:
        result = asyncio.run(enrich_labeling(args.din, args.drug_code, args.strength))
        if result:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("No labeling data extracted.")
    else:
        parser.print_help()
