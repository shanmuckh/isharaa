"""Evaluate the English -> ASL gloss translator against reference glosses.

    uv run python -m text2sign.evaluate            (from src/)
    uv run python src/text2sign/evaluate.py --errors

Metrics (standard for gloss translation, cf. Camgoz et al. 2018/2020):
  * BLEU-1..4  corpus BLEU over gloss tokens (Papineni et al. 2002, +1 smoothing
               for n>1 so short sentences don't zero out)
  * WER        gloss-level word error rate (Levenshtein / reference length)
  * exact      fraction of sentences whose gloss matches the reference exactly
  * coverage   share of output glosses realised by a lexicon sign rather than
               fingerspelling
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from text2sign import translate, load_json  # noqa: E402
else:
    from . import translate, load_json


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ngrams(toks: list, n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps: list[list], refs: list[list], max_n: int = 4) -> dict:
    scores = {}
    hyp_len = sum(len(h) for h in hyps)
    ref_len = sum(len(r) for r in refs)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    log_p = 0.0
    for n in range(1, max_n + 1):
        match = total = 0
        for h, r in zip(hyps, refs):
            hc, rc = ngrams(h, n), ngrams(r, n)
            match += sum(min(c, rc[g]) for g, c in hc.items())
            total += max(len(h) - n + 1, 0)
        p = (match + (n > 1)) / (total + (n > 1)) if total else 0.0
        log_p += math.log(p) if p > 0 else -1e9
        scores[f"bleu{n}"] = round(100 * bp * math.exp(log_p / n), 2)
    return scores


def evaluate(items: list[dict]) -> dict:
    hyps, refs, rows = [], [], []
    signed = spelled = edits = 0
    for it in items:
        out = translate(it["en"])
        hyp = [g for s in out["sentences"] for g in s["gloss"]]
        ref = it["ref"].split()
        hyps.append(hyp)
        refs.append(ref)
        e = edit_distance(hyp, ref)
        edits += e
        signed += out["stats"]["signs"]
        spelled += out["stats"]["fingerspelled"]
        rows.append({"en": it["en"], "hyp": " ".join(hyp), "ref": it["ref"], "edits": e})
    res = corpus_bleu(hyps, refs)
    res["wer"] = round(100 * edits / sum(len(r) for r in refs), 2)
    res["exact"] = round(100 * sum(r["edits"] == 0 for r in rows) / len(rows), 2)
    res["coverage"] = round(100 * signed / max(signed + spelled, 1), 2)
    res["n"] = len(rows)
    return {"metrics": res, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="eval_set.json", help="file in text2sign/data/")
    ap.add_argument("--errors", action="store_true", help="list sentences that differ from the reference")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    args = ap.parse_args()
    result = evaluate(load_json(args.set)["items"])
    if args.json:
        print(json.dumps(result, indent=2))
        return
    m = result["metrics"]
    print(f"Isharaa English->ASL gloss  (n={m['n']})")
    print(f"  BLEU-1 {m['bleu1']:6.2f}   BLEU-2 {m['bleu2']:6.2f}   BLEU-3 {m['bleu3']:6.2f}   BLEU-4 {m['bleu4']:6.2f}")
    print(f"  WER    {m['wer']:6.2f}%  exact match {m['exact']:6.2f}%  lexicon coverage {m['coverage']:6.2f}%")
    if args.errors:
        print("\nMismatches:")
        for r in result["rows"]:
            if r["edits"]:
                print(f"  {r['en']:<34} hyp: {r['hyp']:<30} ref: {r['ref']}")


if __name__ == "__main__":
    main()
