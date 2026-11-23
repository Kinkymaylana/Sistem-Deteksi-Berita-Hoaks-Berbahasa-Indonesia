"""Bangun aturan bahasa scope_guard dari data LATIH (pemilik: Anggota C).

Hanya baris dengan <split-col> == LATIH yang dibaca. VALIDASI dan UJI tidak
dipakai sama sekali.

Contoh:
    python scripts/build_language_rule.py ^
        --data data/DATASET_V3_1.csv ^
        --text-col text_clean ^
        --split-col random_split ^
        --out models/language_rule.json

--text-col sengaja wajib diisi: dokumen rencana menyebut kolom V3.1 sama
dengan V3 (text_clean), sedangkan notebook A2 menyebut kolom baru
text_clean_v31. Pastikan dulu ke Anggota A kolom mana yang final.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG  # noqa: E402
from src.reactive import build_language_rule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--split-col", default="random_split")
    ap.add_argument("--train-value", default="LATIH")
    ap.add_argument("--top-n", type=int, default=DEFAULT_CONFIG.lang_top_n)
    ap.add_argument("--percentile", type=float, default=DEFAULT_CONFIG.lang_percentile)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "language_rule.json")
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    texts, valid_texts = [], []
    split_counts: dict[str, int] = {}
    with open(args.data, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for col in (args.text_col, args.split_col):
            if col not in (reader.fieldnames or []):
                sys.exit(f"Kolom '{col}' tidak ada. Kolom tersedia: {reader.fieldnames}")
        for row in reader:
            split = row[args.split_col]
            split_counts[split] = split_counts.get(split, 0) + 1
            if split == args.train_value:
                texts.append(row[args.text_col])
            elif split == "VALIDASI":
                valid_texts.append(row[args.text_col])

    if not texts:
        sys.exit(f"Tidak ada baris {args.split_col} == {args.train_value}. Isi split: {split_counts}")

    rule = build_language_rule(
        texts, top_n=args.top_n, percentile=args.percentile,
        source=f"{args.data.name}; kolom={args.text_col}; {args.split_col}={args.train_value}",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rule.save(args.out)
    # Pemeriksaan saja (tidak dipakai menyetel tau): seluruh dataset berbahasa
    # Indonesia, jadi idealnya hanya sekitar 1% teks VALIDASI yang tertolak.
    rejected_valid = (sum(rule.score(t) < rule.tau for t in valid_texts) / len(valid_texts)) if valid_texts else None
    print(json.dumps({
        "persen_validasi_tertolak": None if rejected_valid is None else round(100 * rejected_valid, 2),
        "baris_per_split": split_counts,
        "teks_latih_dipakai": rule.n_texts,
        "top_n": rule.top_n,
        "percentile": rule.percentile,
        "tau_bahasa": rule.tau,
        "meta": rule.meta,
        "disimpan_ke": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
