"""Ukur waktu respons p50/p95 per lapisan dari 100 berita VALIDASI (Anggota C).

Rencana Kerja (Anggota C butir 5) dan Proposal Tabel 3.6: waktu respons
p50/p95 per lapisan di laptop CPU. Jalankan di laptop yang dipakai demo,
karena angkanya bergantung pada mesin.

Contoh:
    python scripts/ukur_latensi.py --data data/DATASET_V3_1.csv
Keluaran:
    results/latency.csv          ringkasan p50/p95 per lapisan
    results/latency_detail.csv   waktu setiap berita
"""
from __future__ import annotations

import argparse
import csv
import platform
import random
from pathlib import Path

import numpy as np

from _common import ROOT, read_rows
from src.agent import MODELS_DIR, load_default_agent

LAYERS = ["preprocess", "scope_guard", "similar", "model", "total"]


def split_body(title: str, text_raw: str) -> str:
    t, r = (title or "").strip(), (text_raw or "").strip()
    return r[len(t):].strip() if t and r.startswith(t) else r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split-col", default="random_split")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "latency.csv")
    args = ap.parse_args()

    agent = load_default_agent(args.models_dir)
    rows, _ = read_rows(args.data, ["raw_id", "title", "text_raw"], args.split_col, splits=("VALIDASI",))
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    agent.predict(sample[0]["title"], split_body(sample[0]["title"], sample[0]["text_raw"]))  # pemanasan

    detail = []
    for r in sample:
        res = agent.predict(r["title"], split_body(r["title"], r["text_raw"]), include_timing=True)
        detail.append({"raw_id": r["raw_id"], "lapisan_penentu": res["layer"],
                       **{f"{k}_ms": res["timing_ms"].get(k) for k in LAYERS}})

    summary = []
    for k in LAYERS:
        vals = [d[f"{k}_ms"] for d in detail if d[f"{k}_ms"] is not None]
        summary.append({
            "split": "VALIDASI", "lapisan": k, "n": len(vals),
            "p50_ms": round(float(np.percentile(vals, 50)), 3) if vals else None,
            "p95_ms": round(float(np.percentile(vals, 95)), 3) if vals else None,
            "mesin": f"{platform.system()} {platform.machine()} | Python {platform.python_version()}",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    det_path = args.out.with_name(args.out.stem + "_detail.csv")
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    for s in summary:
        print(f"{s['lapisan']:<12} n={s['n']:>3}  p50={s['p50_ms']} ms  p95={s['p95_ms']} ms")
    print(f"-> {args.out} dan {det_path}")


if __name__ == "__main__":
    main()
