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
from agentdataset.core.review import ClaimReview


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def cmd_discover(args) -> int:
    orch = Orchestrator(
        session_id=args.session_id or "cli_discover",
        model=args.model,
        api_key=_env(args.env_var),
        env_var=args.env_var,
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
    review = ClaimReview(claims_all)
    review.save(claims_path)
    orch.write_run_manifest({"claims_path": str(claims_path)})
    print(f"Claims written to {claims_path}")
    print("Run 'agentdataset review <file>' to approve claims before synthesis.")
    return 0


def cmd_review(args) -> int:
    path = Path(args.claims_file)
    review = ClaimReview.load(path)
    print(f"Loaded {len(review.claims)} claims ({len(review.pending_claims())} pending).")
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
    p.set_defaults(func=cmd_discover)

    p = sub.add_parser("run", help="Run discovery + extraction and write claims for review")
    p.add_argument("query")
    p.add_argument("--sources", type=int, default=3)
    p.add_argument("--all", action="store_true")
    p.add_argument("--output", default=".agentdataset_cache/cli_run")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--env-var", default="OPENAI_API_KEY")
    p.add_argument("--session-id", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("review", help="Review extracted claims")
    p.add_argument("claims_file")
    p.add_argument("--accept-all", action="store_true")
    p.add_argument("--reject-all", action="store_true")
    p.add_argument("--interactive", action="store_true")
    p.set_defaults(func=cmd_review)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())