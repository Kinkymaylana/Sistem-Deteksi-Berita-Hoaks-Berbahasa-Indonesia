"""Perlengkapan uji bersama (Anggota C).

SEMUA DATA DI SINI SINTETIS. Kalimat dibuat dari templat untuk menguji
alur kode, bukan isi dataset kelompok. Model pengganti dibangun dengan
struktur yang sama seperti B3_calibrate.py milik Anggota B
(Pipeline tfidf + LinearSVC, lalu CalibratedClassifierCV sigmoid prefit),
tetapi angkanya tidak bermakna dan tidak boleh dilaporkan.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_VALID_T = [
    "Pemerintah kota {k} meresmikan {o} baru untuk warga pada {h} pagi",
    "Menteri {m} menyampaikan rencana anggaran {o} di gedung DPR {h} lalu",
    "Polisi menangkap tersangka korupsi proyek {o} di kota {k} pekan ini",
    "Gubernur {k} meninjau pembangunan {o} dan meminta pekerjaan selesai tepat waktu",
]
_HOAKS_T = [
    "Viral pesan berantai sebarkan vaksin {o} mengandung chip magnet segera bagikan",
    "Awas beredar kabar {o} di {k} gratis untuk semua klik tautan sekarang juga",
    "Sebarkan info penting presiden bagikan uang {o} kepada warga {k} malam ini",
]
_K = ["Malang", "Surabaya", "Bandung", "Medan", "Makassar", "Semarang"]
_O = ["jembatan", "rumah sakit", "sekolah", "pasar", "jalan tol", "bendungan"]
_H = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]
_M = ["keuangan", "kesehatan", "pendidikan", "perhubungan"]


def make_rows(seed: int = 7, n_valid: int = 60, n_hoaks: int = 30) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    def fill(t):
        return t.format(k=rng.choice(_K), o=rng.choice(_O), h=rng.choice(_H), m=rng.choice(_M))

    for i in range(n_valid + n_hoaks):
        label = int(i >= n_valid)
        text = fill(rng.choice(_HOAKS_T if label else _VALID_T))
        title = " ".join(text.split()[:5])
        split = ["LATIH"] * 6 + ["VALIDASI"] * 2 + ["UJI"] * 2
        rows.append({
            "raw_id": str(1000 + i), "source": "TurnBackHoax" if label else "Kompas",
            "label": str(label), "title": title,
            "url": f"https://contoh.invalid/{1000 + i}",
            "text_raw": text, "text_clean": text,
            "random_split": split[i % 10],
        })
    return rows


def build_standin_model(rows: list[dict]):
    """Model pengganti dengan struktur B3_calibrate.py (data sintetis)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    tr = [r for r in rows if r["random_split"] == "LATIH"]
    va = [r for r in rows if r["random_split"] == "VALIDASI"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)),
    ]).fit([r["text_clean"] for r in tr], [int(r["label"]) for r in tr])
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
    except ImportError:  # scikit-learn < 1.6 (B memakai cv="prefit")
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    return cal.fit([r["text_clean"] for r in va], [int(r["label"]) for r in va])


@pytest.fixture(scope="session")
def rows():
    return make_rows()


@pytest.fixture(scope="session")
def standin_model(rows):
    return build_standin_model(rows)


@pytest.fixture()
def dataset_csv(tmp_path, rows):
    path = tmp_path / "dataset_sintetis.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
