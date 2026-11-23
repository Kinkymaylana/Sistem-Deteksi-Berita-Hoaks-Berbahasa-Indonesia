"""Uji kesetaraan preprocessing aplikasi dengan text_clean dataset (Anggota C).

Rencana Kerja (Anggota A butir 3): minimal 99% identik untuk CNN, Kompas,
Tempo; TurnBackHoax dilaporkan terpisah karena memakai ekstraksi NARASI.
Skrip ini memeriksa src/preprocess.py sementara buatan C. Uji resmi tetap
tests/test_preprocess.py milik A.

Rekonstruksi input dari kolom dataset (mengikuti kode V3 A):
  text_raw = (title + " " + body).strip()      untuk berita VALID
  text_raw = (title + " " + NARASI).strip()    untuk HOAKS
sehingga body = text_raw tanpa awalan judul.
Hanya LATIH dan VALIDASI yang dibaca.

Contoh:
    python scripts/uji_paritas_preprocess.py --data data/DATASET_V3_1_KANDIDAT_OPERASIONAL.csv --text-col text_clean_v31
    python scripts/uji_paritas_preprocess.py --data data/DATASET_V3_KANDIDAT_OPERASIONAL.csv --text-col text_clean --v3

Dua mode diukur:
  sumber_diketahui : clean_news(..., source=<kolom source>), meniru pembuatan dataset
  aplikasi         : clean_news(..., source=None), kondisi input pengguna di aplikasi
                     (aturan Kompas diterapkan, keputusan C)
  unknown_mode_A   : clean_news(..., source="UNKNOWN"), sama dengan
                     clean_v31_for_unknown_source() di file Anggota A
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from _common import ROOT, read_rows
from src.preprocess import clean_news


def split_body(title: str, text_raw: str) -> str:
    t = (title or "").strip()
    r = (text_raw or "").strip()
    return r[len(t):].strip() if t and r.startswith(t) else r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--v3", action="store_true", help="Uji bentuk V3 (tanpa aturan V3.1)")
    ap.add_argument("--per-source", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "paritas_preprocess.csv")
    args = ap.parse_args()

    rows, _ = read_rows(args.data, ["raw_id", "source", "label", "title", "text_raw", args.text_col])
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)

    rng = random.Random(args.seed)
    summary, detail = {}, []
    for src in sorted(by_src):
        sample = rng.sample(by_src[src], min(args.per_source, len(by_src[src])))
        summary[src] = {"n": len(sample)}
        for mode, source in (("sumber_diketahui", src), ("aplikasi", None), ("unknown_mode_A", "UNKNOWN")):
            same = 0
            for r in sample:
                got = clean_news(r["title"], split_body(r["title"], r["text_raw"]), v31=not args.v3, source=source)
                ok = got == r[args.text_col]
                same += ok
                if not ok:
                    detail.append({"mode": mode, "raw_id": r["raw_id"], "source": src, "label": r["label"],
                                   "harapan": r[args.text_col][:300], "hasil": got[:300]})
            pct = 100 * same / len(sample)
            summary[src][mode] = {"identik": same, "persen": round(pct, 2),
                                  "lulus_99": None if src == "TurnBackHoax" else pct >= 99.0}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "raw_id", "source", "label", "harapan", "hasil"])
        w.writeheader()
        w.writerows(detail)
    print(json.dumps({"kolom": args.text_col, "versi": "V3" if args.v3 else "V3.1", "ringkasan": summary,
                      "baris_tidak_identik": str(args.out)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
