"""Terminal front-end for AgentDataset (§10.5).

Shares the same discovery/extraction/review services as the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.review import ClaimReview, parameters_from_claims


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def cmd_discover(args) -> int:
    orch = Orchestrator(
        session_id=args.session_id or "cli_discover",
        model=args.model,
        api_key=_env(args.env_var),
        env_var=args.env_var,
        pdf_backend=getattr(args, "pdf_backend", "pymupdf4llm"),
    )
    results = orch.run_discovery(args.query, max_candidates=args.limit)
    out = []
    for r in results:
        out.append({
            "title": r.title,
            "url": r.url,
            "type": r.source_type,
            "rank": r.rank,
            "query": r.query,
            "score_hint": None,
        })
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.source_type}] {r.title}")
            print(f"   {r.url}")
            if r.snippet:
                print(f"   {r.snippet[:160]}")
    return 0


def cmd_run(args) -> int:
    session_id = args.session_id or f"cli_run"
    orch = Orchestrator(
        session_id=session_id,
        model=args.model,
        api_key=_env(args.env_var),
        env_var=args.env_var,
        pdf_backend=getattr(args, "pdf_backend", "pymupdf4llm"),
    )
    results = orch.run_discovery(args.query, max_candidates=args.sources)
    print(f"Discovered {len(results)} candidates.", file=sys.stderr)
    # Process selected sources (top N unless --all).
    selected = results if args.all else results[: args.sources]
    claims_all = []
    for r in selected:
        try:
            params = orch.process_source(r)
            orch.record_source(r)
            # claims from extractor
            claims = getattr(orch.extractor, "last_claims", []) or []
            orch.record_claims(claims)
            claims_all.extend(claims)
            print(f"  extracted {len(params.variables)} vars from {r.title}", file=sys.stderr)
        except Exception as e:
            print(f"  failed {r.title}: {e}", file=sys.stderr)
    # Write claims file for review.
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    claims_path = out_dir / "claims.json"
    # The agent reviews every extracted claim before any parameters reach synthesis.
    review = orch.review_claims_with_agent(claims_all)
    summary = (
        f"Agent review: {len(review.accepted_claims())} accepted, "
        f"{len(review.rejected_claims())} rejected, "
        f"{len(review.pending_claims())} pending"
    )
    print(summary, file=sys.stderr)
    review.save(claims_path)

    params = parameters_from_claims(review.accepted_claims())
    params_path = out_dir / "parameters.json"
    params_path.write_text(params.model_dump_json(indent=2), encoding="utf-8")
    orch.write_run_manifest({
        "claims_path": str(claims_path),
        "parameters_path": str(params_path),
        "review_mode": "agent",
    })
    print(f"Claims written to {claims_path}")
    print(f"Reviewed parameters written to {params_path}")
    return 0


def cmd_review(args) -> int:
    path = Path(args.claims_file)
    review = ClaimReview.load(path)
    print(f"Loaded {len(review.claims)} claims ({len(review.pending_claims())} pending).")
    if args.agent:
        orch = Orchestrator(session_id="cli_agent_review")
        review = orch.review_claims_with_agent(review.claims)
        review.save(path)
        print(
            f"Agent review: {len(review.accepted_claims())} accepted, "
            f"{len(review.rejected_claims())} rejected, "
            f"{len(review.pending_claims())} pending."
        )
        return 0
    if args.accept_all:
        for c in review.pending_claims():
            review.accept(c.claim_id)
    elif args.reject_all:
        for c in review.pending_claims():
            review.reject(c.claim_id)
    elif args.interactive:
        for c in review.pending_claims():
            print(f"\nClaim {c.claim_id}: {c.variable_name} = {c.value} ({c.claim_type}, conf {c.confidence:.2f})")
            if c.evidence and c.evidence.passage:
                print(f"  passage: {c.evidence.passage[:200]}")
            choice = input("  [a]ccept / [r]eject / [d]efer / [s]kip: ").strip().lower()
            if choice == "a":
                review.accept(c.claim_id)
            elif choice == "r":
                review.reject(c.claim_id)
            elif choice == "d":
                review.defer(c.claim_id)
    else:
        print("No action specified. Use --accept-all, --reject-all, or --interactive.")
    review.save(path)
    print(f"Accepted: {len(review.accepted_claims())}, rejected: {len(review.rejected_claims())}, pending: {len(review.pending_claims())}")
    return 0


def cmd_synthesize(args) -> int:
    import json
    from pathlib import Path
    import pandas as pd
    from agentdataset.models.schemas import Parameters

    params_path = Path(args.params)
    params = Parameters.model_validate_json(params_path.read_text(encoding="utf-8"))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(
        session_id=args.session_id or "cli_synth",
        base_dir=str(out_dir / "sessions"),
        model=args.model,
        api_key=_env(args.env_var),
        env_var=args.env_var,
        pdf_backend=getattr(args, "pdf_backend", "pymupdf4llm"),
    )
    score, df = orch.run_agent_loop(params, iterations=args.iterations)
    print(f"Best fidelity score: {score}")
    if df is not None:
        df.to_csv(out_dir / "data.csv", index=False)
        print(f"Wrote {out_dir / 'data.csv'}")
    return 0


def cmd_validate(args) -> int:
    import json
    from pathlib import Path
    import pandas as pd
    from agentdataset.models.schemas import Parameters

    df = pd.read_csv(args.data)
    params = Parameters.model_validate_json(Path(args.params).read_text(encoding="utf-8"))
    orch = Orchestrator(
        session_id=args.session_id or "cli_validate",
        model=args.model,
        api_key=_env(args.env_var),
        env_var=args.env_var,
        pdf_backend=getattr(args, "pdf_backend", "pymupdf4llm"),
    )
    report = orch.validator.validate(df, params, source_text=args.source_text)
    print(f"overall={report.overall_score} trust={report.trust_score}")
    print(f"critique: {report.critique}")
    return 0


def cmd_doctor(args) -> int:
    import os
    from pathlib import Path
    import sys

    ROOT = Path.cwd()
    _load_dotenv = None
    # load .env if present
    env_path = ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

    checks = []
    def record(name, ok, detail=""):
        checks.append((name, ok, detail))
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))

    # 1. env keys
    key = None
    for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(k):
            key = k
            break
    record("API key present", bool(key), key or "none found")

    # 2. imports
    for mod in ("pydantic", "numpy", "scipy", "pandas", "pymupdf", "pymupdf4llm",
                "requests", "trafilatura", "litellm"):
        try:
            __import__(mod)
            record(f"import {mod}", True)
        except Exception as e:
            record(f"import {mod}", False, str(e))

    # 3. pdf backend
    try:
        import tempfile, os as _os
        import pymupdf
        fd, p = tempfile.mkstemp(suffix=".pdf")
        _os.close(fd)
        import pymupdf4llm
        # write a tiny pdf via pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "doctor test")
        doc.save(p)
        doc.close()
        md = pymupdf4llm.to_markdown(p)
        _os.remove(p)
        record("pdf backend (pymupdf4llm)", bool(md and md.strip()), (md or "")[:40])
    except Exception as e:
        record("pdf backend (pymupdf4llm)", False, str(e))

    # 4. cache dir writable
    try:
        from agentdataset.core.cache import ContentCache
        c = ContentCache(base_dir=str(ROOT / ".agentdataset_cache" / "cache"))
        c.set("doctor_probe", {"ok": True})
        assert c.get("doctor_probe") == {"ok": True}
        record("cache dir writable", True)
    except Exception as e:
        record("cache dir writable", False, str(e))

    # 5. network + model reachability
    try:
        import requests
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        record("network (openrouter)", r.status_code < 500, f"HTTP {r.status_code}")
    except Exception as e:
        record("network (openrouter)", False, str(e))

    if key:
        try:
            from agentdataset.core.extractor import Extractor
            ext = Extractor(model="openrouter/auto", api_key=os.environ[key], env_var=key)
            out = ext.llm_call("Reply with the single word: ok", purpose="doctor")
            record("model reachability", bool(out and out.strip()), (out or "")[:40])
        except Exception as e:
            record("model reachability", False, str(e)[:80])

    # auto-fix
    if getattr(args, "fix", False):
        print("\n-- auto-fix --")
        if not env_path.exists():
            env_path.write_text("# Add your key below\nOPENROUTER_API_KEY=\n", encoding="utf-8")
            print("Created .env template (fill in OPENROUTER_API_KEY).")
        cache_dir = ROOT / ".agentdataset_cache" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("Ensured cache directory exists.")

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentdataset", description="AgentDataset CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("discover", help="Discover candidate sources for a query")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model", default="openrouter/auto")
    p.add_argument("--env-var", default="OPENROUTER_API_KEY")
    p.add_argument("--session-id", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--pdf-backend", default="pymupdf4llm")
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("run", help="Run discovery + extraction and write claims for review")
    p.add_argument("query")
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", default=".agentdataset_cache/cli_run")
    p.add_argument("--model", default="openrouter/auto")
    p.add_argument("--env-var", default="OPENROUTER_API_KEY")
    p.add_argument("--session-id", default=None)
    p.add_argument("--pdf-backend", default="pymupdf4llm")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("review", help="Review extracted claims")
    p.add_argument("claims_file")
    p.add_argument("--accept-all", action="store_true")
    p.add_argument("--reject-all", action="store_true")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--agent", action="store_true", help="Have the LLM agent review the claims")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("synthesize", help="Synthesize a dataset from a reviewed params JSON")
    p.add_argument("params")
    p.add_argument("--output", default=".agentdataset_cache/cli_synth")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--model", default="openrouter/auto")
    p.add_argument("--env-var", default="OPENROUTER_API_KEY")
    p.add_argument("--session-id", default=None)
    p.add_argument("--pdf-backend", default="pymupdf4llm")
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("validate", help="Validate a generated dataset")
    p.add_argument("data")
    p.add_argument("params")
    p.add_argument("--source-text", default=None)
    p.add_argument("--model", default="openrouter/auto")
    p.add_argument("--env-var", default="OPENROUTER_API_KEY")
    p.add_argument("--session-id", default=None)
    p.add_argument("--pdf-backend", default="pymupdf4llm")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("doctor", help="Check environment and dependencies; --fix repairs basics")
    p.add_argument("--fix", action="store_true")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())