"""
AgentDataset Discovery Agent
Search & Fetch Research Documents
"""

import json
import logging
import os
import re
import tempfile
from typing import List, Any
from urllib.parse import urlparse
import requests
import trafilatura
from agentdataset.core.retry import retry_with_backoff, CircuitBreaker
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

# Safety caps for fetched content.
MAX_PDF_BYTES = 25 * 1024 * 1024  # 25 MiB


class SearchError(Exception):
    """Raised when the search backend itself fails (distinct from 'no results')."""


class DiscoveryAgent:
    def __init__(
        self,
        max_results: int = 5,
        api_key: str = "",
        model: str = "openrouter/auto",
    ):
        self.max_results = max_results
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.model = model
        self._circuit = CircuitBreaker()

    @staticmethod
    def _safe_url(url: Any) -> bool:
        """Only allow http(s) URLs with a host, to avoid file:// and SSRF-style targets."""
        try:
            parsed = urlparse(str(url).strip())
        except Exception:
            return False
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)

    def search(self, query: str, max_results: int | None = None) -> List[DiscoveryResult]:
        """Search the web via OpenRouter's Exa-powered web search plugin."""
        limit = max_results if max_results is not None else self.max_results
        results = []
        prompt = (
            "Search the web for: " + query + "\n"
            "Return ONLY a JSON array (no prose, no markdown fences) of up to "
            + str(limit)
            + " objects with keys: title, url, snippet. "
            "Use real URLs from the search results."
        )
        if not self._circuit.allow():
            raise SearchError(f"OpenRouter search circuit is open for query {query!r}")

        def _do_request():
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "plugins": [{"id": "web"}],
                    "temperature": 0,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()

        def _retryable(e):
            import requests as _r
            if isinstance(e, (_r.exceptions.Timeout, _r.exceptions.ConnectionError)):
                return True
            if isinstance(e, _r.exceptions.HTTPError):
                code = getattr(getattr(e, "response", None), "status_code", None)
                return code in (429, 500, 502, 503, 504)
            return False

        try:
            data = retry_with_backoff(_do_request, retryable=_retryable)
            self._circuit.record_success()
        except Exception as e:
            self._circuit.record_failure()
            logger.error("OpenRouter search failed for query %r: %s", query, e, exc_info=True)
            raise SearchError(f"OpenRouter search backend failed for query {query!r}: {e}") from e

        content = ""
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            content = ""

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None

        if not isinstance(parsed, list):
            raise SearchError(f"OpenRouter search returned unparseable results for query {query!r}")

        from datetime import datetime as _dt
        retrieved_at = _dt.now().isoformat(timespec="seconds")
        seen_urls = set()
        rank = 0
        for r in parsed:
            if not isinstance(r, dict):
                continue
            url = r.get("url")
            if not url or not self._safe_url(url):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            rank += 1
            if rank > limit:
                break
            snippet = r.get("snippet")
            if isinstance(snippet, str) and len(snippet) > 2000:
                snippet = snippet[:2000]
            results.append(DiscoveryResult(
                title=(str(r.get("title") or url))[:200],
                url=url,
                source_type="pdf" if str(url).lower().endswith(".pdf") else "html",
                relevance_score=1.0,
                snippet=snippet,
                query=query,
                rank=rank,
                retrieved_at=retrieved_at,
                dedup_decision="kept",
            ))

        if not results:
            raise SearchError(f"OpenRouter search returned no usable results for query {query!r}")
        return results

    def fetch_content(self, result: DiscoveryResult) -> str:
        """Fetch and convert content to text.

        For HTML sources returns extracted text directly.
        For PDF sources downloads the file to a temp path and returns
        ``pdf://<path>`` so the caller can pass it to Extractor.pdf_to_markdown().
        Falls back to the search snippet on any network error.
        """
        if not self._safe_url(result.url):
            logger.warning("Refusing to fetch unsafe URL %s — falling back to snippet", result.url)
            return result.snippet or ""

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
            tmp_path = None
            try:
                response = requests.get(result.url, timeout=15, stream=True, headers=_HTTP_HEADERS)
                response.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                tmp_path = tmp.name
                total = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise ValueError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                    tmp.write(chunk)
                tmp.close()
                logger.info("Downloaded PDF to %s", tmp.name)
                return PDF_PATH_PREFIX + tmp.name
            except Exception as e:
                # Clean up any partial temp file.
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                logger.warning("PDF download failed for %s: %s — falling back to snippet", result.url, e)
                return result.snippet or ""

        return ""
