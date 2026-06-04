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
    ph, colour, shape, size_mm, weight
  Per-strength matching: use the DIN's strength to scope §6 Description block.
  Section location by keyword → only that section passed to Ollama or regex.
  If Ollama (llama3) is available it is preferred; regex is the fallback.

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
import json
import logging
import re
import time
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.cache import cache_get, cache_set
from app.config import (
    DPD_BASE, ENABLE_OCR, HTTP_TIMEOUT, OLLAMA_BASE_URL, OLLAMA_MODEL, USER_AGENT,
)
from app.enrichment.store import get_labeling_for_din, upsert_labeling

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
_DPD_INFO_BASE = "https://health-products.canada.ca/dpd-bdpp/info"

NOT_IN_PM = "Not in PM"
NOT_STATED = NOT_IN_PM        # alias used by tests and external callers
NO_PM_AVAILABLE = "No PM available"  # no PM file exists for the DIN
NEEDS_OCR = "needs OCR / manual check"
PH_SOLUBILITY_ONLY = "Not stated (pH-dependent solubility only)"

# Minimum characters on a page to consider it selectable text (not a scanned image)
_MIN_TEXT_CHARS = 50

_LABELING_FIELDS = (
    "active_ingredient", "excipients_core", "excipients_coating",
    "preservatives", "pack_size", "pack_style",
    "colour", "shape", "size_mm", "weight", "ph",
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

# Pre-built regex patterns for speed (word-boundary anchored, case-insensitive)
_CONTAINER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), label)
    for kw, label in _CONTAINER_VOCAB_ORDERED
]


def _extract_pack_style_from_text(text: str, source_label: str = "") -> Optional[str]:
    """Return Title-Case container label if any vocabulary keyword is found in text."""
    for pattern, label in _CONTAINER_PATTERNS:
        if pattern.search(text):
            if source_label:
                logger.info(
                    "pack_style=%r found in %s (text snippet: %r)",
                    label, source_label, text[:100],
                )
            return label
    return None


def _extract_pack_size_from_product_info(prod_info: str) -> Optional[str]:
    """Parse product_information free-text into a clean pack_size string.

    Container-type keywords are stripped before size parsing so they cannot
    bleed into the result.

    Examples:
      "FOR I.V. INFUSION ONLY. 80MG/ML(RECONST.) - 5ML VIAL." → "5 mL"
      "24/50/100/200"                                           → "24, 50, 100, 200 count"
      "100/500"                                                 → "100, 500 count"
      "100 TABLETS"                                             → "100 count"
      "5ML"                                                     → "5 mL"
    """
    # Strip container keywords
    text = prod_info.upper()
    for pattern, _ in _CONTAINER_PATTERNS:
        text = pattern.sub(" ", text)
    text = text.strip()

    # Volume: standalone N mL / N L — NOT a concentration like 80MG/ML
    # The negative lookahead rejects things like "80MG/ML"
    vol_m = re.search(
        r'(?:^|[\s\-\.,(])(\d+(?:\.\d+)?)\s*(ML|L)\b(?!\s*/)',
        text,
    )
    if vol_m:
        num = float(vol_m.group(1))
        unit = "mL" if vol_m.group(2) == "ML" else "L"
        return f"{num:g} {unit}"

    # Slash-separated pure-integer counts: "24/50/100/200" or "100/500"
    slash_m = re.search(r'\b(\d+(?:/\d+)+)\b', text)
    if slash_m:
        parts = slash_m.group(1).split("/")
        if all(p.isdigit() for p in parts):
            return ", ".join(parts) + " count"

    # Explicit count with tablet/capsule/unit word
    count_m = re.search(r'\b(\d+)\s+(?:TABLETS?|CAPSULES?|CAPS?|UNITS?)\b', text)
    if count_m:
        return f"{count_m.group(1)} count"

    return None


# ── Stage 2: DPD API + info page ─────────────────────────────────────────────

async def _fetch_active_ingredient_api(drug_code: int) -> Optional[str]:
    """Fetch active ingredient name(s) from DPD /activeingredient/ API."""
    cache_key = f"ai:{drug_code}"
    cached = cache_get("dpd_ai", cache_key)
    if cached is not None:
        return cached or None

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{DPD_BASE}/activeingredient/",
                params={"id": drug_code, "lang": "en", "type": "json"},
                headers=_HEADERS,
                timeout=HTTP_TIMEOUT,
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
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{DPD_BASE}/packaging/",
                params={"id": drug_code, "type": "json"},
                headers=_HEADERS,
                timeout=HTTP_TIMEOUT,
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
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(
                page_url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=HTTP_TIMEOUT,
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


# ── Stage 3: Ollama extraction ────────────────────────────────────────────────

async def _is_ollama_available() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


async def _query_ollama(section_text: str, page_num: int, field_group: str) -> dict:
    """Query Ollama llama3 for one field group from a section of PM text.

    field_group: "excipients" | "appearance" | "ph"
    Returns dict of field → {"value": str|None, "found": bool, "page": int|None}

    Rules enforced in the prompt:
      - One group per call (never all 8 fields at once)
      - temperature=0, num_ctx=8192
      - Copy verbatim; do NOT invent or paraphrase
      - found=false + value=null means "searched, absent" — valid
    """
    if field_group == "excipients":
        fields_desc = {
            "excipients_core": (
                "Non-medicinal / inactive ingredients in the tablet CORE only. "
                "Do NOT include coating ingredients. Typical labels: "
                "'Non-medicinal ingredients', 'Core tablet', 'Tablet core', 'Inactive ingredients'."
            ),
            "excipients_coating": (
                "Ingredients in the FILM COAT or COATING ONLY. "
                "Only populate if there is an explicit 'Film coat', 'Film coating', or 'Coating:' subsection. "
                "If no coating subsection exists, set found=false."
            ),
            "preservatives": (
                "Preservative(s) specifically listed (e.g. benzalkonium chloride, methylparaben). "
                "Return null if none are listed; set found=false if no preservative section."
            ),
        }
    elif field_group == "appearance":
        fields_desc = {
            "colour": "Colour(s) of the tablet/capsule/product (e.g. 'white', 'light blue').",
            "shape": "Shape (e.g. 'round', 'oval', 'oblong', 'biconvex'). Null if absent.",
            "size_mm": "Dimensions in mm (e.g. '9.5 mm', '11 × 7 mm'). Null if absent.",
            "weight": "Weight of the dosage unit in mg (e.g. '325 mg tablet weight'). Null if absent.",
        }
    elif field_group == "ph":
        fields_desc = {
            "ph": (
                "Standalone pH value or range (e.g. '6.8', '4.5–7.0'). "
                "If there is ONLY a pH-solubility table (no standalone pH property), "
                "return the exact string 'Not stated (pH-dependent solubility only)'. "
                "Return null if pH is not mentioned at all."
            ),
        }
    else:
        return {}

    fields_json = json.dumps({k: v for k, v in fields_desc.items()}, indent=2)
    prompt = (
        f"Extract pharmaceutical data from this product monograph text (page {page_num}).\n\n"
        f"Fields to extract:\n{fields_json}\n\n"
        f"For each field return:\n"
        f'  "value": exact verbatim text from the document, or null\n'
        f'  "found": true if present, false if not found\n'
        f'  "page": {page_num} if found, null if not\n\n'
        f"RULES:\n"
        f"- Copy values VERBATIM. Do NOT paraphrase, abbreviate, or invent.\n"
        f"- If absent: value=null, found=false, page=null.\n"
        f"- Return ONLY valid JSON — no prose, no markdown.\n\n"
        f"TEXT (page {page_num}):\n"
        f"{section_text[:5000]}\n\n"
        f"JSON response:"
    )

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0, "num_ctx": 8192},
                },
                timeout=120.0,
            )
        if r.status_code != 200:
            logger.debug("Ollama returned %d for field_group=%s", r.status_code, field_group)
            return {}
        raw = r.json().get("response", "")
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("Ollama response not valid JSON for %s: %s", field_group, exc)
        return {}
    except Exception as exc:
        logger.debug("Ollama query failed for %s: %s", field_group, exc)
        return {}


def _apply_ollama_result(
    row: dict,
    ollama_out: dict,
    fields: list[str],
    fallback_page: Optional[int],
) -> None:
    """Write Ollama results into row dict, honouring NOT_IN_PM sentinel."""
    for field in fields:
        if field not in ollama_out:
            continue
        entry = ollama_out[field]
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

_COLOUR_WORDS = (
    r"(?:light |pale |dark |bright |deep |off-?)?(?:white|red|pink|orange|yellow|green|"
    r"blue|purple|violet|brown|beige|grey|gray|black|cream|tan|teal|maroon|ivory)"
)
_SHAPE_WORDS = (
    r"(?:round|oval(?:oid)?|oblong|capsule[- ]?shaped|caplet|biconvex|"
    r"pentagonal|hexagonal|octagonal|triangular|diamond|shield|kidney)"
)
_SIZE_PAT = r"(\d+(?:\.\d+)?\s*mm(?:\s*[×xX]\s*\d+(?:\.\d+)?\s*mm)?)"
_WEIGHT_PAT = r"(\d+(?:\.\d+)?\s*mg\s*(?:tablet\s+weight|weight\s+of\s+tablet|per\s+tablet)?)"

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
    r"the following dosage strengths|dosage strengths|:\s*$|^\s*dosage\s+form",
    re.IGNORECASE,
)


def _extract_pack_style_from_pdf(s6_text: str) -> Optional[str]:
    """Extract packaging description from §6 Packaging subsection.

    Only returns a value when it:
      (a) contains a container-vocabulary keyword, AND
      (b) is not a section heading / dosage-strengths fragment.

    Stage 2 (DPD API) overrides this in enrich_labeling.
    """
    pats = [
        # "Packaging" on its own line, followed by content
        r"(?m)^Packaging\s*$\n(.+?)(?=\n\n|\n[A-Z\d]|\Z)",
        # "Packaging:" as a label
        r"(?m)^Packaging\s*:\s*(.+?)(?=\n\n|\n[A-Z\d]|\Z)",
        # "provided in ..." sentence
        r"(?:provided|available)\s+in\s+(.+?)(?:\.|;|\n|$)",
    ]
    for pat in pats:
        m = re.search(pat, s6_text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        val = m.group(1).strip()
        if len(val) <= 5:
            continue
        # Hard reject: heading fragments and missing container keyword
        if _PACK_STYLE_HEADING_REJECT.search(val):
            continue
        if not _extract_pack_style_from_text(val):
            continue
        return val
    return None


_KNOWN_PRESERVATIVES_RE = re.compile(
    r"\b(?:methylparaben|propylparaben|benzyl\s+alcohol|benzalkonium\s+chloride|"
    r"sodium\s+benzoate|potassium\s+benzoate|benzoic\s+acid|sorbic\s+acid|"
    r"phenoxyethanol|phenol|cresol|chlorobutanol|thimerosal)\b",
    re.IGNORECASE,
)

# Patterns that locate a non-medicinal ingredient list in §6 text
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
    return val.strip() if val else NOT_IN_PM


def _extract_appearance_regex(
    desc_text: str,
    target_strength: Optional[str],
) -> dict[str, Optional[str]]:
    """Extract colour/shape/size/weight, scoped to target_strength block if possible."""
    out: dict[str, Optional[str]] = {"colour": None, "shape": None, "size_mm": None, "weight": None}

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
        else:
            lm = re.search(
                r"^.*" + strength_pat + r".*$",
                desc_text, re.IGNORECASE | re.MULTILINE,
            )
            if lm:
                block_text = lm.group(0)

    cm = re.search(_COLOUR_WORDS, block_text, re.IGNORECASE)
    if cm:
        out["colour"] = cm.group(0).strip()

    sm = re.search(_SHAPE_WORDS, block_text, re.IGNORECASE)
    if sm:
        out["shape"] = sm.group(0).strip()

    szm = re.search(_SIZE_PAT, block_text, re.IGNORECASE)
    if szm:
        out["size_mm"] = szm.group(1).strip()

    wm = re.search(_WEIGHT_PAT, block_text, re.IGNORECASE)
    if wm:
        out["weight"] = wm.group(1).strip()

    return out


# ── §6 / §13 markers ─────────────────────────────────────────────────────────

_S6_MARKERS = [
    r"^6[\.\s]+DOSAGE FORMS?,?\s*COMPOSITION",
    r"^6[\.\s]+PHARMACEUTICAL INFORMATION",
    r"^PART II.*SCIENTIFIC INFORMATION",
]
_S6_END = [r"^7[\.\s]+", r"^PART\s+III", r"^CLINICAL\s+PHARMACOLOGY"]

_S13_MARKERS = [
    r"^13[\.\s]+PHARMACEUTICAL INFORMATION",
    r"^PHARMACEUTICAL INFORMATION",
]
_S13_END = [r"^14[\.\s]+", r"^NON-CLINICAL", r"^TOXICOLOGY"]

_DESC_MARKERS = [r"[Dd]escription", r"[Pp]hysical\s+[Dd]escription"]
_DESC_END = [r"[Cc]omposition", r"[Pp]ackaging", r"[Ss]torage", r"\n\n\n"]


# ── Main extraction entry point ───────────────────────────────────────────────

async def parse_labeling_fields_async(
    pages: list[tuple[int, str]],
    din_strength: Optional[str],
) -> dict:
    """Extract Stage 3 label fields from pre-extracted PDF pages.

    Only extracts: excipients_core, excipients_coating, preservatives,
                   ph, colour, shape, size_mm, weight.
    Does NOT extract: active_ingredient, pack_size, pack_style
                      (those come from Stage 2 / DPD API).

    Scanned / low-text PDFs are handled by the caller (_extract_text_with_ocr)
    before this function is called. This function always attempts extraction
    regardless of text density. needs_ocr is set by enrich_labeling().

    Returns a flat dict. Every scalar field has a companion _page field.
    """
    row: dict = {}

    s6 = _find_section(pages, _S6_MARKERS, _S6_END)
    s6_page = s6[0] if s6 else None
    s6_text = s6[1] if s6 else ""

    s13 = _find_section(pages, _S13_MARKERS, _S13_END)
    s13_page = s13[0] if s13 else None
    s13_text = s13[1] if s13 else ""

    desc_section = _find_section(pages, _DESC_MARKERS, _DESC_END)
    desc_page = desc_section[0] if desc_section else s6_page
    desc_text = desc_section[1] if desc_section else s6_text

    use_ollama = await _is_ollama_available()

    if use_ollama:
        # --- Ollama path ---
        excip_section_text = s6_text
        nm_match = re.search(
            r"(?:Non-?[Mm]edicinal|[Ii]nactive|[Ee]xcipient)[^\n]{0,60}\n(.{50,}?)(?=\n\n|\Z)",
            s6_text, re.DOTALL,
        )
        if nm_match:
            excip_section_text = nm_match.group(0)

        ollama_a = await _query_ollama(excip_section_text, s6_page or 1, "excipients")
        _apply_ollama_result(row, ollama_a, ["excipients_core", "excipients_coating", "preservatives"], s6_page)

        if din_strength and desc_text:
            norm = _normalize_strength(din_strength)
            ollama_b = await _query_ollama(
                f"Product: {norm}\n\n{desc_text}", desc_page or 1, "appearance"
            )
        else:
            ollama_b = await _query_ollama(desc_text or s6_text, desc_page or 1, "appearance")
        _apply_ollama_result(row, ollama_b, ["colour", "shape", "size_mm", "weight"], desc_page)

        if s13_text:
            ollama_c = await _query_ollama(s13_text, s13_page or 1, "ph")
            _apply_ollama_result(row, ollama_c, ["ph"], s13_page)
        else:
            row["ph"] = NOT_IN_PM
            row["ph_page"] = None

    else:
        # --- Regex fallback path ---
        logger.info("Ollama offline — using regex fallback for Stage 3 extraction")

        # active_ingredient: best-effort from §6 (Stage 2 DPD API overrides in enrich_labeling)
        ai_pdf = _extract_active_ingredient_regex(s6_text)
        row["active_ingredient"] = ai_pdf if ai_pdf else NOT_IN_PM
        row["active_ingredient_page"] = s6_page if ai_pdf else None

        core, coating = _extract_excipients_regex(s6_text)
        row["excipients_core"] = core if core else NOT_IN_PM
        row["excipients_core_page"] = s6_page if core else None
        row["excipients_coating"] = coating if coating else NOT_IN_PM
        row["excipients_coating_page"] = s6_page if coating else None

        pres = _classify_preservatives(s6_text)
        row["preservatives"] = pres
        row["preservatives_page"] = s6_page if pres not in (NOT_IN_PM,) else None

        if desc_text:
            norm_strength = _normalize_strength(din_strength) if din_strength else None
            app = _extract_appearance_regex(desc_text, norm_strength)
        else:
            app = {"colour": None, "shape": None, "size_mm": None, "weight": None}

        for field in ("colour", "shape", "size_mm", "weight"):
            val = app.get(field)
            row[field] = val if val else NOT_IN_PM
            row[f"{field}_page"] = desc_page if val else None

        row["ph"] = _extract_ph_regex(s13_text) if s13_text else NOT_IN_PM
        row["ph_page"] = s13_page if row["ph"] not in (NOT_IN_PM, PH_SOLUBILITY_ONLY) else None

        # pack_style: packaging description from §6 Packaging subsection
        # (Stage 2 DPD API overrides in enrich_labeling when available)
        ps_pdf = _extract_pack_style_from_pdf(s6_text)
        row["pack_style"] = ps_pdf if ps_pdf else NOT_IN_PM
        row["pack_style_page"] = s6_page if ps_pdf else None

    row["needs_ocr"] = 0  # overridden by enrich_labeling() if OCR was used
    row["has_unverified"] = 0
    return row


def parse_labeling_fields(
    pages: list[tuple[int, str]],
    din_strength: Optional[str],
) -> dict:
    """Synchronous wrapper around parse_labeling_fields_async for backward compatibility.

    Must only be called from a non-async context; raises RuntimeError otherwise.
    """
    return asyncio.run(parse_labeling_fields_async(pages, din_strength))


# ── Public named wrappers for test introspection ──────────────────────────────

def _extract_strength_block(description: str, target_strength: str) -> dict[str, Optional[str]]:
    """Public alias used by tests: extract colour/shape/size/weight for one strength."""
    return _extract_appearance_regex(description, target_strength)


def _extract_ph(s13_text: str) -> str:
    """Public alias used by tests: extract pH from §13 text."""
    return _extract_ph_regex(s13_text)


async def enrich_labeling(
    din: str,
    drug_code: int,
    strength: Optional[str],
    pdf_bytes: Optional[bytes] = None,
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

    # Stage 3: PDF extraction — OCR applied per-page when text layer is thin
    try:
        cache_key = pdf_url or f"pdf_bytes:{din}"
        pages, ocr_used = _extract_text_with_ocr(
            pdf_bytes, cache_key=cache_key, enable_ocr=ENABLE_OCR,
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
) -> dict[str, Optional[dict]]:
    """Enrich labeling for multiple DINs. din_map: {din → (drug_code, strength)}"""
    results = await asyncio.gather(*[
        enrich_labeling(din, drug_code, strength)
        for din, (drug_code, strength) in din_map.items()
    ])
    return dict(zip(din_map.keys(), results))


async def _download_pdf(url: str) -> Optional[bytes]:
    cache_key = f"pdf:{url}"
    cached = cache_get("labeling_pdf", cache_key)
    if cached is not None:
        import base64
        return base64.b64decode(cached)

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=60.0,
            )
        if r.status_code != 200:
            logger.warning("PDF download HTTP %d for %s", r.status_code, url)
            return None
        pdf_bytes = r.content
        import base64
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
    # Verify container words don't leak into pack_size
    for tc in test_cases:
        size = _extract_pack_size_from_product_info(tc["product_information"])
        if size:
            for _, label in _CONTAINER_VOCAB_ORDERED:
                if label.upper() in (size or "").upper():
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
