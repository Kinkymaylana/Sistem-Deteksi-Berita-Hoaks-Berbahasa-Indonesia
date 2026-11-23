"""Uji find_similar, penyetelan tau_mirip, penjelasan, dan antrean CSV (Anggota C).

Data dan model di sini sintetis (lihat conftest.py). Angka tidak bermakna.
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

from src.agent import Agent
from src.batch import normalize_columns, process, queue
from src.explain import lime_explanation, svm_word_contributions
from src.preprocess import clean_news
from src.reactive import SimilarIndex, choose_tau_similar, tau_table


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------
@pytest.fixture()
def index():
    texts = ["vaksin covid mengandung chip magnet pelacak",
             "presiden bagikan uang tunai kepada warga lewat tautan",
             "air rebusan bawang menyembuhkan semua penyakit"]
    return SimilarIndex.build(texts, ["H1", "H2", "H3"], ["u1", "u2", "u3"], ["1", "2", "3"])


def test_teks_identik_skor_satu(index):
    hit = index.query("vaksin covid mengandung chip magnet pelacak")
    assert hit["title"] == "H1" and math.isclose(hit["score"], 1.0)


def test_tetangga_terdekat_dipilih(index):
    hit = index.query("katanya presiden bagikan uang tunai ke warga")
    assert hit["title"] == "H2" and 0 < hit["score"] < 1


def test_tanpa_kata_bersama_none(index):
    assert index.query("zzz qqq") is None


def test_skor_sama_dengan_cosine_manual(index):
    from sklearn.metrics.pairwise import cosine_similarity
    q = "air rebusan bawang untuk penyakit"
    expect = cosine_similarity(index.vectorizer.transform([q]), index.matrix).max()
    assert math.isclose(index.query(q)["score"], float(expect), rel_tol=1e-9)


def test_scores_many_sama_dengan_query(index):
    qs = ["vaksin chip magnet", "uang tunai warga", "bawang penyakit"]
    _, s = index.scores_many(qs, chunk=2)
    assert np.allclose(s, [index.query(q)["score"] for q in qs])


def test_simpan_muat(index, tmp_path):
    p = tmp_path / "idx.joblib"
    index.meta["tau_similar"] = 0.93
    index.save(p)
    loaded = SimilarIndex.load(p)
    assert loaded.meta["tau_similar"] == 0.93
    for q in ["vaksin chip magnet", "uang tunai warga lewat tautan", "bawang penyakit semua"]:
        a, b = loaded.query(q), index.query(q)
        assert a["title"] == b["title"] and math.isclose(a["score"], b["score"], rel_tol=1e-12)
    # format portabel: tidak ada objek scikit-learn yang di-pickle
    import joblib
    raw = joblib.load(p)
    assert raw["format"] == "similar_index_portabel_v1" and "vectorizer" not in raw


# ---------------------------------------------------------------------------
# Penyetelan tau_mirip
# ---------------------------------------------------------------------------
def test_tabel_precision():
    scores = [0.95, 0.92, 0.91, 0.70]
    labels = [1, 1, 0, 1]
    t = {r["tau"]: r for r in tau_table(scores, labels, [0.90, 0.92, 0.96])}
    assert t[0.90]["n_cocok"] == 3 and math.isclose(t[0.90]["precision"], 2 / 3)
    assert t[0.92]["precision"] == 1.0 and t[0.92]["n_cocok"] == 2
    assert t[0.96]["precision"] is None
    assert math.isclose(t[0.92]["coverage_hoaks"], 2 / 3)


def test_pilih_ambang_terkecil_yang_memenuhi_precision():
    scores = [0.95, 0.92, 0.91, 0.70]
    labels = [1, 1, 0, 1]
    table = tau_table(scores, labels, [round(0.5 + 0.01 * i, 2) for i in range(50)])
    assert choose_tau_similar(table, floor=0.90, min_precision=0.99) == 0.92


def test_tidak_ada_yang_memenuhi_mengembalikan_none():
    table = tau_table([0.95, 0.91], [0, 0], [0.90, 0.95])
    assert choose_tau_similar(table) is None
    assert choose_tau_similar(tau_table([0.5], [1], [0.90])) is None  # tidak ada yang cocok


# ---------------------------------------------------------------------------
# Penjelasan
# ---------------------------------------------------------------------------
def test_kontribusi_svm_menjumlah_ke_skor_keputusan(standin_model):
    text = "viral pesan berantai sebarkan vaksin jembatan mengandung chip magnet"
    ex = svm_word_contributions(standin_model, text, top_k=50)
    from src.explain import _unwrap
    pipe, *_ = _unwrap(standin_model)
    assert math.isclose(ex["skor_svm"], float(pipe.decision_function([text])[0]), rel_tol=1e-9)
    assert ex["platt_a"] is not None and ex["platt_a"] < 0  # f naik -> p naik
    assert ex["ke_hoaks"] and all(c > 0 for _, c in ex["ke_hoaks"])
    assert all(c < 0 for _, c in ex["ke_valid"])
    top_words = {w for w, _ in ex["ke_hoaks"]}
    assert top_words & {"viral", "sebarkan", "chip", "magnet", "pesan"}


def test_kontribusi_svm_top_k(standin_model):
    ex = svm_word_contributions(standin_model, "pemerintah kota malang meresmikan sekolah baru", top_k=3)
    assert len(ex["ke_hoaks"]) <= 3 and len(ex["ke_valid"]) <= 3


def test_lime_tanpa_pustaka_memberi_pesan_jelas(standin_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "lime", None)
    monkeypatch.setitem(sys.modules, "lime.lime_text", None)
    with pytest.raises(ImportError, match="pip install lime"):
        lime_explanation(standin_model, "teks apa saja")


def test_lime_pemanggilan_benar_dengan_pustaka_tiruan(standin_model, monkeypatch):
    """Hanya menguji sambungan ke API lime (pustaka asli tidak terpasang di sini)."""
    seen = {}

    class FakeExp:
        def as_list(self, label):
            seen["label_as_list"] = label
            return [("viral", 0.3), ("kota", -0.1)]

    class FakeExplainer:
        def __init__(self, class_names, random_state):
            seen["class_names"] = class_names

        def explain_instance(self, text, classifier_fn, num_features, labels, num_samples):
            seen["proba_shape"] = np.asarray(classifier_fn([text, text])).shape
            seen.update(labels=labels, num_features=num_features, num_samples=num_samples)
            return FakeExp()

    mod = types.ModuleType("lime.lime_text")
    mod.LimeTextExplainer = FakeExplainer
    monkeypatch.setitem(sys.modules, "lime", types.ModuleType("lime"))
    monkeypatch.setitem(sys.modules, "lime.lime_text", mod)
    out = lime_explanation(standin_model, "viral pesan", num_features=5, num_samples=200)
    assert out == [("viral", 0.3), ("kota", -0.1)]
    assert seen["proba_shape"] == (2, 2) and seen["labels"] == (1,) and seen["label_as_list"] == 1
    assert seen["class_names"] == ["VALID", "HOAKS"]


# ---------------------------------------------------------------------------
# CSV dan antrean (Desain Metode 2.8.2)
# ---------------------------------------------------------------------------
class _ScriptedAgent:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def predict(self, title, body):
        return self.outputs.pop(0)


def _res(action, layer, p=None, sim=None):
    return {"action": action, "layer": layer, "p_hoaks": p, "reason": "r", "similar": sim, "model": "m"}


def test_kolom_csv_dikenali():
    df = normalize_columns(pd.DataFrame({"Judul": ["a"], "Isi": ["b"]}))
    assert list(df.columns) == ["title", "body"] and df.iloc[0].tolist() == ["a", "b"]
    df = normalize_columns(pd.DataFrame({"text": ["hanya isi"]}))
    assert df.iloc[0]["title"] == "" and df.iloc[0]["body"] == "hanya isi"
    with pytest.raises(ValueError, match="kolom"):
        normalize_columns(pd.DataFrame({"x": [1]}))


def test_antrean_diurutkan_dan_dikelompokkan():
    outs = [
        _res("VALID", "model", 0.05),
        _res("HOAKS", "similar", None, {"title": "H", "url": "u", "score": 0.97}),
        _res("HOAKS", "model", 0.91),
        _res("PERLU VERIFIKASI", "scope_guard"),
        _res("PERLU VERIFIKASI", "model", 0.35),
    ]
    df = pd.DataFrame({"title": list("abcde"), "body": ["x"] * 5})
    res = process(_ScriptedAgent(outs), df)
    q = queue(res)
    assert q["antrean"]["p_hoaks"].tolist() == [0.91, 0.35, 0.05]
    assert q["antrean"]["p_hoaks_persen"].tolist() == [91.0, 35.0, 5.0]
    assert q["cocok_hoaks_lama"]["judul"].tolist() == ["b"]
    assert q["di_luar_cakupan"]["judul"].tolist() == ["d"]
    assert len(res) == 5


def test_csv_ujung_ke_ujung_dengan_model_pengganti(standin_model, rows):
    ag = Agent(model=standin_model, preprocess=clean_news)
    sample = [r for r in rows if r["random_split"] == "VALIDASI"]
    df = pd.DataFrame({"judul": [r["title"] for r in sample], "isi": [r["text_raw"] for r in sample]})
    q = queue(process(ag, df))
    assert sum(len(v) for v in q.values()) == len(sample) > 0
    ps = q["antrean"]["p_hoaks"].tolist()
    assert ps == sorted(ps, reverse=True)
