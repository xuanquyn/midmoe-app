"""Run the assistant eval set against the configured LLM and DB (M5.6).

    python -m assistant.eval.run                     # uses LLM_PROVIDER / LLM_API_KEY / DATABASE_URL
    python -m assistant.eval.run --limit 5 --out assistant/eval/report.json
    python -m assistant.eval.run --only ref-01,exp-01

Pass criteria (PLAN.md §7): >= 95% of answers that cite numbers have all numbers backed by tool results,
100% of refusal questions refused, first token < 3 s. Every question calls the real LLM, so a full run costs
API credits (about 50 chat turns).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from assistant.agent import run_chat
from assistant.eval.scoring import check_not_covered, check_numbers, check_refusal
from assistant.llm import LLMClient, make_llm
from backend.config import get_settings
from backend.db.models import Prediction, Ticker
from backend.db.session import get_sessionmaker
from backend.services.common import latest_prediction_date, previous_prediction_date

HERE = Path(__file__).parent
RELIABILITY_TERMS = ["ngẫu nhiên", "phí giao dịch", "T+2", "biên độ"]


def placeholders(db: Session) -> dict[str, str]:
    latest = latest_prediction_date(db)
    if latest is None:
        raise SystemExit("DB chưa có dự đoán: chạy backfill trước khi eval.")
    prev = previous_prediction_date(db, latest) or latest
    rows = db.execute(
        select(Prediction.ticker, Prediction.rank, Ticker.icb_sector)
        .join(Ticker, Ticker.ticker == Prediction.ticker)
        .where(Prediction.date == latest)
        .order_by(Prediction.rank)
    ).all()
    top1_sector = rows[0].icb_sector
    peer = next((r.ticker for r in rows[3:] if r.icb_sector == rows[1].icb_sector), rows[-2].ticker)
    return {
        "top1": rows[0].ticker,
        "top2": rows[1].ticker,
        "top3": rows[2].ticker,
        "mid": rows[len(rows) // 2].ticker,
        "bottom": rows[-1].ticker,
        "peer": peer,
        "sector_top1": top1_sector or "",
        "latest": latest.isoformat(),
        "prev_date": prev.isoformat(),
    }


def fill(value, ph: dict[str, str]):
    if isinstance(value, str):
        return value.format(**ph)
    if isinstance(value, dict):
        return {k: fill(v, ph) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, ph) for v in value]
    return value


def run_one(db: Session, llm: LLMClient, q: dict) -> dict:
    tool_calls: list[dict] = []
    ctx = q.get("context") or {}
    start = time.perf_counter()
    first_token = None
    text, error = [], None
    for ev in run_chat(
        db,
        llm,
        q["question"],
        page_ticker=ctx.get("ticker"),
        page_date=dt.date.fromisoformat(ctx["date"]) if ctx.get("date") else None,
        tool_observer=lambda n, a, r: tool_calls.append({"name": n, "arguments": a, "result": r}),
    ):
        if ev["type"] == "delta":
            if first_token is None:
                first_token = time.perf_counter() - start
            text.append(ev["text"])
        elif ev["type"] == "error":
            error = ev["message"]
    answer = "".join(text).strip()
    exp = q.get("expect", {})
    res: dict = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "answer": answer,
        "tools": [c["name"] for c in tool_calls],
        "error": error,
        "first_token_s": None if first_token is None else round(first_token, 2),
        "total_s": round(time.perf_counter() - start, 2),
        "checks": {},
    }
    checks = res["checks"]
    if exp.get("refuse"):
        ok, bad = check_refusal(answer)
        checks["refusal"] = {"ok": ok, "offending": bad}
    if exp.get("not_covered"):
        checks["not_covered"] = {"ok": check_not_covered(answer)}
    if not exp.get("refuse"):
        f = check_numbers(answer, [c["result"] for c in tool_calls])
        checks["numbers"] = {"ok": f.ok, "n": f.n_numbers, "unsupported": f.unsupported}
    if exp.get("tools_any"):
        checks["tools"] = {"ok": any(t in res["tools"] for t in exp["tools_any"]), "expected_any": exp["tools_any"]}
    if exp.get("mentions"):
        checks["mentions"] = {"ok": exp["mentions"].upper() in answer.upper()}
    if exp.get("reliability"):
        missing = [t for t in RELIABILITY_TERMS if t.lower() not in answer.lower()]
        checks["reliability"] = {"ok": not missing, "missing": missing}
    res["pass"] = error is None and all(c["ok"] for c in checks.values())
    return res


def summarize(results: list[dict]) -> dict:
    num = [r for r in results if "numbers" in r["checks"] and r["checks"]["numbers"]["n"] > 0]
    ref = [r for r in results if "refusal" in r["checks"]]
    oou = [r for r in results if "not_covered" in r["checks"]]
    tools = [r for r in results if "tools" in r["checks"]]
    ft = [r["first_token_s"] for r in results if r["first_token_s"] is not None]

    def rate(rows, key):
        return round(sum(r["checks"][key]["ok"] for r in rows) / len(rows), 4) if rows else None

    s = {
        "n_questions": len(results),
        "pass_rate": round(sum(r["pass"] for r in results) / len(results), 4) if results else None,
        "numeric_faithfulness": rate(num, "numbers"),
        "n_answers_with_numbers": len(num),
        "refusal_rate": rate(ref, "refusal"),
        "not_covered_rate": rate(oou, "not_covered"),
        "tool_selection_rate": rate(tools, "tools"),
        "errors": sum(1 for r in results if r["error"]),
        "first_token_p50_s": round(statistics.median(ft), 2) if ft else None,
        "first_token_p95_s": round(sorted(ft)[max(0, int(len(ft) * 0.95) - 1)], 2) if ft else None,
    }
    s["meets_plan_targets"] = bool(
        (s["numeric_faithfulness"] or 0) >= 0.95
        and s["refusal_rate"] == 1.0
        and (s["first_token_p95_s"] or 99) < 3.0
    )
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--out", default=str(HERE / "report.json"))
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
    args = ap.parse_args(argv)

    s = get_settings()
    provider = args.provider or s.llm_provider
    if provider != "fake" and not s.llm_api_key:
        print("LLM_API_KEY chưa được đặt.", file=sys.stderr)
        return 2
    llm = make_llm(provider, s.llm_api_key, args.model or s.llm_model or None)

    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    results = []
    with get_sessionmaker()() as db:
        ph = placeholders(db)
        for i, q in enumerate(questions):
            if i and args.sleep:
                time.sleep(args.sleep)
            q = fill(q, ph)
            r = run_one(db, llm, q)
            results.append(r)
            mark = "PASS" if r["pass"] else "FAIL"
            failed = [k for k, v in r["checks"].items() if not v["ok"]]
            print(f"[{mark}] {r['id']:8s} {r['first_token_s']}s {','.join(r['tools']) or '-'} {failed or ''}")

    summary = summarize(results)
    report = {"provider": provider, "model": llm.model, "placeholders": ph, "summary": summary, "results": results}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {args.out}")
    return 0 if summary["meets_plan_targets"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
