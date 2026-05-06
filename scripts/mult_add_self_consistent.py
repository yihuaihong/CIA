"""Backfill `self_consistent` (and pp1/pp2/final) into mult corruption jsonls
that were produced before the field was saved. Idempotent — skips records
that already have the field.

Self-consistent = pp1 + pp2*10 == final, parsed from full_generation.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import jsonlines

PP_RE = re.compile(r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)", re.MULTILINE)
FINAL_RE = re.compile(r"FINAL ANSWER:?\s*(\d+)", re.IGNORECASE)


def parse(full_generation: str) -> tuple[int | None, int | None, int | None, bool]:
    pps = PP_RE.findall(full_generation)
    m_f = FINAL_RE.search(full_generation)
    final = int(m_f.group(1)) if m_f else None
    pp1 = pp2 = None
    if len(pps) >= 2:
        try:
            pp1 = int(pps[0][0]); pp2 = int(pps[1][0])
        except Exception:
            pass
    self_consistent = (pp1 is not None and pp2 is not None and final is not None
                       and final == pp1 + pp2)
    return pp1, pp2, final, self_consistent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="corruption_labeled.jsonl files")
    args = ap.parse_args()

    for path in args.inputs:
        path = Path(path)
        records = list(jsonlines.open(str(path)))
        n_added = n_already = n_skip = 0
        n_sc = 0
        for r in records:
            if "self_consistent" in r:
                n_already += 1
                if r["self_consistent"]:
                    n_sc += 1
                continue
            full = r.get("full_generation", "") or ""
            if not full:
                n_skip += 1
                continue
            pp1, pp2, final, sc = parse(full)
            r["self_consistent"] = bool(sc)
            r["pp1"] = pp1
            r["pp2"] = pp2
            r["final"] = final
            n_added += 1
            if sc: n_sc += 1
        with jsonlines.open(str(path), "w") as w:
            for r in records:
                w.write(r)
        print(f"{path.name}: added={n_added} already={n_already} skip={n_skip} "
              f"self_consistent_total={n_sc}/{len(records)}")


if __name__ == "__main__":
    main()
