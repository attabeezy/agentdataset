"""
AgentDataset Discovery Agent
Search & Fetch Research Documents
"""

import logging
import os
import tempfile
from typing import List
import requests
from ddgs import DDGS
import trafilatura
from agentdataset.models.schemas import DiscoveryResult

logger = logging.getLogger(__name__)

# Prefix used to signal that fetch_content returned a local file path, not inline text
PDF_PATH_PREFIX = "pdf://"

# Browser-like headers — many hosts (academia.edu, researchgate.net) return 403
# to requests' default python-requests User-Agent.
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
}


class SearchError(Exception):
    """Raised when the search backend itself fails (distinct from 'no results')."""


class DiscoveryAgent:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[DiscoveryResult]:
        """Search web for relevant documents."""
        results = []
        try:
            with DDGS() as ddgs:
                # Search for PDFs specifically
                pdf_query = f"{query} filetype:pdf"
                for r in ddgs.text(pdf_query, max_results=self.max_results):
                    results.append(DiscoveryResult(
                        title=r['title'],
                        url=r['href'],
                        source_type="pdf",
                        relevance_score=1.0,  # Placeholder
                        snippet=r['body']
                    ))

                # General web search for HTML
                for r in ddgs.text(query, max_results=self.max_results):
                    if not r['href'].endswith(".pdf"):
                        results.append(DiscoveryResult(
                            title=r['title'],
                            url=r['href'],
                            source_type="html",
                            relevance_score=0.8,  # Placeholder
                            snippet=r['body']
                        ))
        except Exception as e:
            logger.error("Search failed for query %r: %s", query, e, exc_info=True)
            # Distinguish a backend failure from a genuine empty result set: if we
            # already collected some results, return them; otherwise signal the error.
            if not results:
                raise SearchError(f"Search backend failed for query {query!r}: {e}") from e

        return results

    def fetch_content(self, result: DiscoveryResult) -> str:
        """Fetch and convert content to text.

        For HTML sources returns extracted text directly.
        For PDF sources downloads the file to a temp path and returns
        ``pdf://<path>`` so the caller can pass it to Extractor.pdf_to_markdown().
        Falls back to the search snippet on any network error.
        """
        if result.source_type == "html":
            try:
                downloaded = trafilatura.fetch_url(result.url)
                if downloaded:
                    extracted = trafilatura.extract(downloaded)
                    if extracted:
                        return extracted
                    logger.warning("trafilatura extracted no text from %s — falling back to snippet", result.url)
            except Exception as e:
                logger.warning("Failed to fetch HTML %s: %s — falling back to snippet", result.url, e)
            # Fall back to the search snippet on empty extraction or any error.
            return result.snippet or ""

        elif result.source_type == "pdf":
            try:
                response = requests.get(result.url, timeout=15, stream=True, headers=_HTTP_HEADERS)
                response.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp.close()
                logger.info("Downloaded PDF to %s", tmp.name)
                return PDF_PATH_PREFIX + tmp.name
            except Exception as e:
                logger.warning("PDF download failed for %s: %s — falling back to snippet", result.url, e)
                return result.snippet or ""

        return ""
