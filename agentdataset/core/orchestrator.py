"""
AgentDataset Orchestrator
The Autonomous Engine (Brain)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, List
import pandas as pd
from agentdataset.models.schemas import SessionContext, Parameters, FidelityReport, DiscoveryResult
from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator

logger = logging.getLogger(__name__)

MAX_NOISE = 2.0

class Orchestrator:
    def __init__(self, session_id: str, base_dir: str = "sessions", model: str = "gpt-4o", api_key: str = ""):
        self.context = SessionContext(
            session_id=session_id,
            path=str(Path(base_dir) / session_id)
        )
        os.makedirs(self.context.path, exist_ok=True)

        self.discovery = DiscoveryAgent()
        self.extractor = Extractor(model=model, api_key=api_key)
        self.synthesizer = Synthesizer()
        self.validator = Validator()

        self.best_score = 0.0
        self.best_params: Optional[Parameters] = None
        self.best_data: Optional[pd.DataFrame] = None

    def run_discovery(self, query: str) -> List[DiscoveryResult]:
        """Phase 0: Discovery."""
        return self.discovery.search(query)

    def process_source(self, result: DiscoveryResult) -> Parameters:
        """Phase 1: Extraction."""
        content = self.discovery.fetch_content(result)

        if content.startswith(PDF_PATH_PREFIX):
            pdf_path = content[len(PDF_PATH_PREFIX):]
            try:
                text = self.extractor.pdf_to_markdown(pdf_path)
            finally:
                # Clean up temp file regardless of extraction outcome
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
        else:
            text = content

        return self.extractor.extract_parameters(text, result.title)

    def run_optimization_loop(self, parameters: Parameters, iterations: int = 5):
        """Phase 2 & 3: The Engine (Synthesis-Validation Loop)."""
        current_params = parameters
        noise_level = 0.1

        for i in range(iterations):
            logger.info("Loop %d/%d...", i + 1, iterations)

            # Synthesis
            df = self.synthesizer.synthesize(current_params, noise_level=noise_level)

            # Validation
            report = self.validator.validate(df, current_params)
            logger.info("  Fidelity Score: %s", report.overall_score)

            # Ratchet Logic
            if report.overall_score > self.best_score:
                logger.info("  [KEEP] New best score!")
                self.best_score = report.overall_score
                self.best_params = current_params
                self.best_data = df

                # Save artifacts
                df.to_csv(Path(self.context.path) / "data.csv", index=False)
                with open(Path(self.context.path) / "parameters.json", "w") as f:
                    f.write(current_params.model_dump_json(indent=2))

                # Generate and save DATACARD
                datacard = self.validator.generate_datacard(report, current_params, df)
                with open(Path(self.context.path) / "DATACARD.md", "w") as f:
                    f.write(datacard)
            else:
                logger.info("  [DISCARD] Score did not improve.")
                # Strategy Pivot: increase noise, capped to avoid runaway degradation
                noise_level = min(noise_level * 1.1, MAX_NOISE)

        return self.best_score, self.best_data
