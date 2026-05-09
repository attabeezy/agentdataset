"""
AgentDataset Discovery Agent
Search & Fetch Research Documents
"""

import logging
from typing import List
from duckduckgo_search import DDGS
import trafilatura
from agentdataset.models.schemas import DiscoveryResult

logger = logging.getLogger(__name__)

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
            logger.error("Search failed for query %r: %s", query, e)

        return results

    def fetch_content(self, result: DiscoveryResult) -> str:
        """Fetch and convert content to Markdown."""
        if result.source_type == "html":
            try:
                downloaded = trafilatura.fetch_url(result.url)
                if downloaded:
                    return trafilatura.extract(downloaded) or ""
            except Exception as e:
                logger.error("Failed to fetch %s: %s", result.url, e)
        elif result.source_type == "pdf":
            # TODO: download and parse PDF with extractor.pdf_to_markdown()
            # For now, return the search snippet so the extractor has some text to work with
            return result.snippet or ""
        return ""
