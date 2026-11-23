"""Uji integrasi dengan model ASLI Anggota B (LinearSVM_calibrated_v31.joblib).

Hanya memeriksa bahwa agen, preprocessing, dan penjelasan bekerja dengan
objek model B. Tidak ada angka kinerja di sini. Jika scikit-learn terpasang
bukan 1.6.1 (versi model card), peluang yang keluar tidak boleh dilaporkan.
Dilewati otomatis jika file model belum ada di models/.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HAS_MODEL = any((MODELS / f).exists() for f in ("LinearSVM_calibrated_v31.joblib", "LinearSVM_calibrated.joblib"))
pytestmark = pytest.mark.skipif(not HAS_MODEL, reason="Model Anggota B belum diletakkan di models/")

CONTRACT_KEYS = {"action", "layer", "p_hoaks", "reason", "similar", "model"}
TITLE = "Pemerintah umumkan jadwal vaksinasi lanjutan"
BODY = ("Jakarta, CNN Indonesia -- Kementerian Kesehatan mengumumkan jadwal vaksinasi lanjutan untuk warga "
        "di seluruh provinsi mulai pekan depan. Lihat Juga :Judul berita lain Vaksinasi dilakukan di puskesmas.")


@pytest.fixture(scope="module")
def agent():
    from src.agent import load_default_agent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # InconsistentVersionWarning dicatat di agent.warnings
        return load_default_agent(MODELS)


def test_struktur_model_sesuai_b3(agent):
    m = agent.model
    assert type(m).__name__ == "CalibratedClassifierCV" and m.method == "sigmoid"
    assert list(m.classes_) == [0, 1]
    assert list(m.calibrated_classifiers_[0].estimator.named_steps) == ["tfidf", "classifier"]


def test_versi_preprocessing_dari_model_card(agent):
    assert "V3.1" in agent.model_card["data_version"]
    assert agent.status()["preprocess_version"] == "V3.1"


def test_predict_format_dan_konsistensi_p(agent):
    from src.agent import decide
    from src.preprocess import clean_news
    res = agent.predict(TITLE, BODY)
    assert set(res) == CONTRACT_KEYS and res["layer"] == "model"
    clean = clean_news(TITLE, BODY, v31=True)
    assert "Lihat Juga" not in clean and "CNN Indonesia --" not in clean
    p = float(agent.model.predict_proba([clean])[0][1])
    assert math.isclose(res["p_hoaks"], p)
    assert res["action"] == decide(p)[0]


def test_teks_pendek_tidak_sampai_model(agent):
    res = agent.predict("Vaksin bikin magnet", "")
    assert res["layer"] == "scope_guard" and res["p_hoaks"] is None


def test_kontribusi_kata_svm_konsisten_dengan_model(agent):
    from src.explain import _unwrap, svm_word_contributions
    from src.preprocess import clean_news
    clean = clean_news(TITLE, BODY)
    ex = svm_word_contributions(agent.model, clean, top_k=10)
    pipe, *_ = _unwrap(agent.model)
    assert math.isclose(ex["skor_svm"], float(pipe.decision_function([clean])[0]), rel_tol=1e-9)
    assert ex["platt_a"] < 0  # skor SVM naik -> p(HOAKS) naik
    assert ex["n_kata_dikenal"] > 0 and (ex["ke_hoaks"] or ex["ke_valid"])
