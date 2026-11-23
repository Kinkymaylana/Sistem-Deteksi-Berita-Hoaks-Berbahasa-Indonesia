"""Fungsi bersama untuk skrip Anggota C: membaca dataset tanpa menyentuh UJI."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_SPLITS = {"LATIH", "VALIDASI"}


def read_rows(path, columns, split_col="random_split", splits=("LATIH", "VALIDASI")):
    """Baca baris dataset untuk split tertentu saja.

    UJI ditolak secara eksplisit: data UJI hanya dijalankan D satu kali lewat
    --final (Rencana Kerja bagian 2), tidak dipakai untuk penyetelan apa pun.
    """
    bad = set(splits) - ALLOWED_SPLITS
    if bad:
        raise ValueError(f"Split {bad} tidak boleh dibaca oleh skrip Anggota C.")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows, counts = [], {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in [*columns, split_col] if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"Kolom {missing} tidak ada di {path}. Kolom tersedia: {reader.fieldnames}")
        for row in reader:
            s = row[split_col]
            counts[s] = counts.get(s, 0) + 1
            if s in splits:
                rows.append({c: row[c] for c in [*columns, split_col]})
    return rows, counts
