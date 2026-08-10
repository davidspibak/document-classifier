"""
Client for a locally-run GROBID service (Java, runs as `docker run grobid` or
a local install — see docs/architecture.md). GROBID is purpose-built for
academic-paper metadata extraction and should be tried BEFORE the general
LLM-based extraction fallback, since it's faster and more reliable for
standard scholarly formatting.
"""
import requests
import xml.etree.ElementTree as ET

GROBID_URL = "http://localhost:8070"  # default local GROBID port
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def is_grobid_available() -> bool:
    try:
        resp = requests.get(f"{GROBID_URL}/api/isalive", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def extract_header_metadata(pdf_path: str) -> dict | None:
    """
    Calls GROBID's header-extraction endpoint and parses the returned TEI XML
    into a flat dict. Returns None on failure so the caller can fall back to
    LLM-based extraction.
    """
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={"input": f},
                data={"consolidateHeader": "1"},
                timeout=60,
            )
        if resp.status_code != 200:
            return None
        return _parse_tei_header(resp.text)
    except requests.exceptions.RequestException:
        return None


def _parse_tei_header(tei_xml: str) -> dict:
    root = ET.fromstring(tei_xml)
    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    title = title_el.text if title_el is not None else None

    authors = []
    for pers_name in root.findall(".//tei:sourceDesc//tei:author/tei:persName", TEI_NS):
        forename = pers_name.find("tei:forename", TEI_NS)
        surname = pers_name.find("tei:surname", TEI_NS)
        name_parts = [p.text for p in (forename, surname) if p is not None and p.text]
        if name_parts:
            authors.append(" ".join(name_parts))

    date_el = root.find(".//tei:publicationStmt/tei:date", TEI_NS)
    published_date = date_el.get("when") if date_el is not None else None

    keywords = [kw.text for kw in root.findall(".//tei:keywords//tei:term", TEI_NS) if kw.text]

    abstract_el = root.find(".//tei:abstract", TEI_NS)
    abstract = "".join(abstract_el.itertext()).strip() if abstract_el is not None else None

    return {
        "title": title, "authors": authors, "published_date": published_date,
        "keywords": keywords, "abstract": abstract,
    }
