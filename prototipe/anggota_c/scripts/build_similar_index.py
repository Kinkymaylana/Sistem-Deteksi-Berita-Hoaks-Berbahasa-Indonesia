"""Bangun indeks hoaks serupa dan setel tau_mirip (pemilik: Anggota C).

Desain Metode 2.5.2 dan Rencana Kerja (Anggota C butir 3):
  1. Indeks TF-IDF dari teks HOAKS (label = 1) di data LATIH.
  2. Setiap berita VALIDASI (kedua kelas) dicari satu tetangga terdekatnya.
  3. Untuk setiap ambang t, precision = proporsi berita VALIDASI dengan skor >= t
     yang memang HOAKS.
  4. tau_mirip = ambang terkecil >= 0,90 dengan precision >= 0,99.
UJI tidak dibaca.

Contoh:
    python scripts/build_similar_index.py --data data/DATASET_V3_1.csv --text-col text_clean

Keluaran:
    models/similar_index.joblib              indeks + tau_mirip terpilih di meta
    results/tau_similar_validasi.csv         tabel precision per ambang (VALIDASI)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from _common import ROOT, read_rows
from src.reactive import SimilarIndex, choose_tau_similar, tau_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--text-col", required=True, help="Kolom teks bersih yang dipakai model (konfirmasi ke A/B)")
    ap.add_argument("--split-col", default="random_split")
    ap.add_argument("--floor", type=float, default=0.90, help="Ambang awal (Desain 2.5.2)")
    ap.add_argument("--min-precision", type=float, default=0.99)
    ap.add_argument("--out-index", type=Path, default=ROOT / "models" / "similar_index.joblib")
    ap.add_argument("--out-table", type=Path, default=ROOT / "results" / "tau_similar_validasi.csv")
    args = ap.parse_args()

    cols = ["raw_id", "label", "title", "url", args.text_col]
    rows, counts = read_rows(args.data, cols, args.split_col)
    latih_hoaks = [r for r in rows if r[args.split_col] == "LATIH" and r["label"] == "1"]
    valid = [r for r in rows if r[args.split_col] == "VALIDASI"]
    if not latih_hoaks or not valid:
        raise SystemExit(f"Data LATIH-HOAKS atau VALIDASI kosong. Isi split: {counts}")

    index = SimilarIndex.build(
        [r[args.text_col] for r in latih_hoaks],
        [r["title"] for r in latih_hoaks],
        [r["url"] for r in latih_hoaks],
        [r["raw_id"] for r in latih_hoaks],
        meta={"data": args.data.name, "text_col": args.text_col, "split_col": args.split_col,
              "dibangun": datetime.now().isoformat(timespec="seconds")},
    )

    _, scores = index.scores_many([r[args.text_col] for r in valid])
    labels = [int(r["label"]) for r in valid]
    grid = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0,50 .. 0,99
    table = tau_table(scores.tolist(), labels, grid)
    tau = choose_tau_similar(table, floor=args.floor, min_precision=args.min_precision)

    index.meta.update({
        "tau_similar": tau,
        "tau_rule": f"terkecil >= {args.floor} dengan precision >= {args.min_precision} di VALIDASI",
        "n_validasi": len(valid),
    })
    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    index.save(args.out_index)

    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", *table[0].keys(), "terpilih"])
        w.writeheader()
        for r in table:
            w.writerow({"split": "VALIDASI", **r, "terpilih": int(tau is not None and r["tau"] == tau)})

    view = [r for r in table if r["tau"] >= 0.80]
    print(json.dumps({
        "baris_per_split": counts,
        "hoaks_latih_diindeks": len(latih_hoaks),
        "berita_validasi": len(valid),
        "tau_mirip_terpilih": tau,
        "catatan": None if tau is not None else
            "Tidak ada ambang >= floor yang memenuhi precision; lapisan memakai 0,90. Diskusikan dengan kelompok.",
        "tabel_mulai_0_80": view,
        "indeks": str(args.out_index),
        "tabel": str(args.out_table),
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
