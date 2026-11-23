"""Pemrosesan banyak berita dari CSV dan pengurutan antrean (pemilik: Anggota C).

Desain Metode 2.8.2 "Urutan Antrean":
  - Berita yang dinilai model diurutkan dari peluang hoaks terkalibrasi tertinggi.
  - Berita yang cocok dengan hoaks lama ditampilkan pada kelompok terpisah.
  - Berita yang ditolak penjaga cakupan tidak masuk antrean.
"""
from __future__ import annotations

import pandas as pd

MAX_ROWS = 1000

# Nama kolom yang diterima (huruf besar/kecil diabaikan).
TITLE_COLS = ("title", "judul")
BODY_COLS = ("body", "isi", "text", "teks", "content", "narasi")

OUT_COLS = ["no", "judul", "tindakan", "lapisan", "p_hoaks", "p_hoaks_persen", "alasan",
            "hoaks_mirip_judul", "hoaks_mirip_url", "hoaks_mirip_skor"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah CSV pengguna menjadi kolom title dan body."""
    lower = {c.lower().strip(): c for c in df.columns}
    t = next((lower[c] for c in TITLE_COLS if c in lower), None)
    b = next((lower[c] for c in BODY_COLS if c in lower), None)
    if t is None and b is None:
        raise ValueError(
            "CSV harus punya kolom judul (title/judul) dan/atau isi (body/isi/text/teks). "
            f"Kolom yang ada: {list(df.columns)}"
        )
    if len(df) > MAX_ROWS:
        raise ValueError(f"CSV berisi {len(df)} baris; batas aplikasi {MAX_ROWS} baris per unggahan.")
    out = pd.DataFrame({
        "title": df[t].fillna("").astype(str) if t else "",
        "body": df[b].fillna("").astype(str) if b else "",
    })
    return out.reset_index(drop=True)


def process(agent, df: pd.DataFrame) -> pd.DataFrame:
    """Jalankan agent.predict untuk setiap baris."""
    rows = []
    for i, r in normalize_columns(df).iterrows():
        res = agent.predict(r["title"], r["body"])
        sim = res.get("similar") or {}
        p = res["p_hoaks"]
        rows.append({
            "no": i + 1,
            "judul": r["title"] or r["body"][:80],
            "tindakan": res["action"],
            "lapisan": res["layer"],
            "p_hoaks": p,
            "p_hoaks_persen": None if p is None else round(100 * p, 1),
            "alasan": res["reason"],
            "hoaks_mirip_judul": sim.get("title"),
            "hoaks_mirip_url": sim.get("url"),
            "hoaks_mirip_skor": None if not sim else round(sim["score"], 3),
        })
    return pd.DataFrame(rows, columns=OUT_COLS)


def queue(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Kelompokkan hasil sesuai Desain Metode 2.8.2."""
    model = results[results["lapisan"] == "model"].sort_values("p_hoaks", ascending=False, kind="stable")
    similar = results[results["lapisan"] == "similar"].sort_values("hoaks_mirip_skor", ascending=False, kind="stable")
    outside = results[results["lapisan"] == "scope_guard"]
    return {
        "antrean": model.reset_index(drop=True),
        "cocok_hoaks_lama": similar.reset_index(drop=True),
        "di_luar_cakupan": outside.reset_index(drop=True),
    }
