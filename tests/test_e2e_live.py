from pathlib import Path
import os
import time

import pytest

from agentdataset.core.orchestrator import AGENTDATASET_CACHE_DIR, Orchestrator


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / AGENTDATASET_CACHE_DIR


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@pytest.mark.live_e2e
def test_live_e2e_generates_inspection_report():
    _load_dotenv(ROOT / ".env")

    env_var = os.environ.get("AGENTDATASET_E2E_ENV_VAR", "OPENAI_API_KEY")
    model = os.environ.get("AGENTDATASET_E2E_MODEL", "gpt-4o")
    api_key = os.environ.get(env_var, "")
    if not api_key:
        pytest.skip(f"{env_var} is not set in the environment or repo-root .env.")

    source_text = """
    A lending portfolio study reports three borrower measurements.
    Monthly income is normally distributed with mean 5200 and standard deviation 1100.
    Loan balance is normally distributed with mean 18000 and standard deviation 4200.
    Credit score is normally distributed with mean 690 and standard deviation 45.
    The correlation between income and loan_balance is 0.42.
    The correlation between credit_score and loan_balance is -0.31.
    """

    session_id = f"live_e2e_{int(time.time())}"
    orchestrator = Orchestrator(
        session_id=session_id,
        base_dir=str(CACHE_ROOT / "sessions"),
        model=model,
        api_key=api_key,
        env_var=env_var,
    )

    params = orchestrator.extractor.extract_parameters(source_text, "live_e2e_fixture")

    assert params.meta.extraction_method == "llm"
    assert len(params.variables) >= 2

    best_score, best_data = orchestrator.run_optimization_loop(params, iterations=2)

    session_dir = Path(orchestrator.context.path)
    data_path = session_dir / "data.csv"
    params_path = session_dir / "parameters.json"
    datacard_path = session_dir / "DATACARD.md"

    assert best_data is not None
    assert not best_data.empty
    assert data_path.exists()
    assert params_path.exists()
    assert datacard_path.exists()

    report_dir = CACHE_ROOT / "e2e"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "live_e2e_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Live E2E Report",
                "",
                f"- Model: `{model}`",
                f"- Env var: `{env_var}`",
                f"- Extraction method: `{params.meta.extraction_method}`",
                f"- Variables: {', '.join(params.variables.keys())}",
                f"- Correlations: {', '.join(params.correlations.keys()) or 'none'}",
                f"- Best score: {best_score}",
                f"- Session directory: `{session_dir}`",
                f"- Data: `{data_path}`",
                f"- Parameters: `{params_path}`",
                f"- Datacard: `{datacard_path}`",
                "",
                "## Datacard Preview",
                "",
                datacard_path.read_text(encoding="utf-8"),
            ]
        ),
        encoding="utf-8",
    )

    assert report_path.exists()
