"""Uji agen dengan peluang tiruan (pemilik: Anggota C).

Sesuai rencana (Anggota C butir 2), decide dan scope_guard diuji dengan nilai
p tiruan, tanpa menunggu model. Semua teks di file ini adalah kalimat sintetis
buatan untuk pengujian, bukan isi dataset kelompok.

Jalankan dari folder PROTOTIPE_KEL5:
    python -m pytest tests/test_agent.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import agent as agent_mod  # noqa: E402
from src.agent import Agent, combine_text, decide  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CONFIG, HOAKS, PERLU_VERIFIKASI, VALID, AgentConfig, thresholds_from_costs,
)
from src.reactive import (  # noqa: E402
    LanguageRule, build_language_rule, count_words, percentile_linear, scope_guard,
)

CONTRACT_KEYS = {"action", "layer", "p_hoaks", "reason", "similar", "model"}

# Kalimat sintetis berbahasa Indonesia (>= 8 kata).
ID_TEXT = "Pemerintah kota mengumumkan jadwal vaksinasi gratis untuk warga di seluruh kecamatan pekan depan"
EN_TEXT = "The quick brown fox jumps over the lazy dog while the sun is shining today"


# ---------------------------------------------------------------------------
# Komponen tiruan
# ---------------------------------------------------------------------------
class FakeModel:
    """Model tiruan: selalu mengembalikan p yang sama dan mencatat pemanggilan."""

    def __init__(self, p: float, classes=(0, 1)):
        self.p = p
        self.classes_ = list(classes)
        self.calls = 0

    def predict_proba(self, texts):
        self.calls += 1
        row = {0: 1 - self.p, 1: self.p}
        return [[row[c] for c in self.classes_] for _ in texts]


def fake_clean(title, body) -> str:
    return combine_text(title, body).lower()


def make_agent(p=0.5, **kw) -> Agent:
    kw.setdefault("preprocess", fake_clean)
    return Agent(model=FakeModel(p), **kw)


# ---------------------------------------------------------------------------
# 1. Ambang dan biaya
# ---------------------------------------------------------------------------
def test_ambang_dari_biaya_sama_dengan_rencana():
    t_valid, t_hoaks = thresholds_from_costs(2, 5, 1)
    assert t_valid == 0.20
    assert t_hoaks == 0.50


@pytest.mark.parametrize("c_fn, expected_valid", [(3, 1 / 3), (5, 0.20), (10, 0.10)])
def test_ambang_valid_ikut_c_fn_untuk_risk_coverage(c_fn, expected_valid):
    t_valid, t_hoaks = thresholds_from_costs(2, c_fn, 1)
    assert math.isclose(t_valid, expected_valid)
    assert t_hoaks == 0.50


def test_biaya_tidak_valid_ditolak():
    with pytest.raises(ValueError):
        thresholds_from_costs(0, 5, 1)
    with pytest.raises(ValueError):
        thresholds_from_costs(2, 5, 3)  # C_R terlalu mahal, tidak ada zona verifikasi


# ---------------------------------------------------------------------------
# 2. decide(p) dengan p tiruan, termasuk tepat di batas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p, expected", [
    (0.00, VALID), (0.10, VALID), (0.20, VALID),
    (0.2000001, PERLU_VERIFIKASI), (0.35, PERLU_VERIFIKASI), (0.4999999, PERLU_VERIFIKASI),
    (0.50, HOAKS), (0.83, HOAKS), (1.00, HOAKS),
])
def test_decide_batas(p, expected):
    action, reason = decide(p)
    assert action == expected
    assert reason


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), "0.5", None, True])
def test_decide_menolak_p_tidak_valid(bad):
    with pytest.raises(ValueError):
        decide(bad)


def test_decide_alasan_memakai_koma_desimal():
    _, reason = decide(0.83)
    assert "0,83" in reason and "0,50" in reason


def test_decide_mengikuti_config():
    cfg = AgentConfig(c_fn=10)  # ambang VALID turun ke 0,10
    assert decide(0.15, cfg)[0] == PERLU_VERIFIKASI
    assert decide(0.15)[0] == VALID


# ---------------------------------------------------------------------------
# 3. scope_guard: input kosong dan teks pendek (dihitung pada teks bersih)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [None, "", "   ", "!!! ... ???"])
def test_scope_guard_input_kosong(text):
    g = scope_guard(text)
    assert g["action"] == PERLU_VERIFIKASI
    assert "kosong" in g["reason"].lower()


def test_scope_guard_tujuh_kata_ditolak_delapan_lolos():
    tujuh = "satu dua tiga empat lima enam tujuh"
    delapan = tujuh + " delapan"
    assert count_words(tujuh) == 7 and count_words(delapan) == 8
    g = scope_guard(tujuh)
    assert g["action"] == PERLU_VERIFIKASI and "7 kata" in g["reason"]
    assert scope_guard(delapan) is None


def test_hitung_kata_sama_dengan_definisi_v3():
    # Kata berulang dengan tanda hubung dihitung satu, seperti WORD_RE di pipeline V3.
    assert count_words("anak-anak bermain di taman") == 4
    assert count_words("Jum'at, 13/11/2026") == 4


# ---------------------------------------------------------------------------
# 4. Aturan bahasa
# ---------------------------------------------------------------------------
LATIH_SINTETIS = [
    "pemerintah kota mengumumkan jadwal vaksinasi untuk warga di kecamatan",
    "warga kota diminta datang ke kantor kecamatan untuk vaksinasi gratis",
    "kantor pemerintah membuka layanan baru untuk warga yang datang pagi",
    "jadwal layanan di kantor kecamatan berubah mulai pekan depan untuk warga",
    "pesan berantai menyebut vaksinasi berbahaya padahal pemerintah membantah kabar itu",
    "beredar kabar kantor pemerintah tutup padahal layanan tetap buka untuk warga",
] * 5


def test_persentil_sama_dengan_numpy():
    np = pytest.importorskip("numpy")
    vals = [0.3, 0.9, 0.5, 0.7, 0.71, 0.1, 0.95]
    for q in (0, 1, 25, 50, 99, 100):
        assert math.isclose(percentile_linear(vals, q), float(np.percentile(vals, q)))


def test_aturan_bahasa_dibangun_dari_latih():
    rule = build_language_rule(LATIH_SINTETIS, top_n=30, percentile=1.0)
    assert len(rule.vocab) == 30
    assert rule.n_texts == len(LATIH_SINTETIS)
    scores = [rule.score(t) for t in LATIH_SINTETIS]
    assert math.isclose(rule.tau, percentile_linear(scores, 1.0))
    # Teks Inggris skornya rendah, teks Indonesia dari domain yang sama lolos.
    assert rule.score(EN_TEXT) < rule.tau
    assert scope_guard(EN_TEXT, language_rule=rule)["action"] == PERLU_VERIFIKASI
    assert "bahasa" in scope_guard(EN_TEXT, language_rule=rule)["reason"].lower()
    assert scope_guard(LATIH_SINTETIS[0], language_rule=rule) is None


def test_aturan_bahasa_simpan_dan_muat(tmp_path):
    rule = build_language_rule(LATIH_SINTETIS, top_n=30)
    p = tmp_path / "language_rule.json"
    rule.save(p)
    loaded = LanguageRule.load(p)
    assert loaded.vocab == rule.vocab and loaded.tau == rule.tau


def test_tanpa_aturan_bahasa_teks_inggris_tidak_ditolak_scope_guard():
    # Mendokumentasikan perilaku saat language_rule.json belum dibangun.
    assert scope_guard(EN_TEXT) is None


# ---------------------------------------------------------------------------
# 5. predict(title, body): format keluaran yang disepakati
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p, expected", [(0.05, VALID), (0.35, PERLU_VERIFIKASI), (0.90, HOAKS)])
def test_predict_lapisan_model(p, expected):
    res = make_agent(p).predict("Judul berita", ID_TEXT)
    assert set(res) == CONTRACT_KEYS
    assert res["action"] == expected
    assert res["layer"] == "model"
    assert math.isclose(res["p_hoaks"], p)
    assert res["similar"] is None          # find_similar tidak dipasang
    assert res["model"] == "LinearSVM-platt"


def test_predict_teks_pendek_tidak_memanggil_model():
    ag = make_agent(0.99)
    res = ag.predict("Vaksin bikin magnet", "")
    assert res["action"] == PERLU_VERIFIKASI
    assert res["layer"] == "scope_guard"
    assert res["p_hoaks"] is None
    assert ag.model.calls == 0


def test_predict_input_kosong_tidak_memanggil_preprocess():
    called = []
    ag = Agent(model=FakeModel(0.5), preprocess=lambda t, b: called.append(1) or "")
    res = ag.predict("", "   ")
    assert res["action"] == PERLU_VERIFIKASI and res["layer"] == "scope_guard"
    assert "kosong" in res["reason"].lower()
    assert called == []


def test_scope_guard_dihitung_setelah_dibersihkan():
    # Desain Metode 2.5.1: "Jika isi teks SETELAH dibersihkan kurang dari 8 kata".
    # Teks mentah 9+ token (URL dipecah jadi banyak kata), setelah clean_news tinggal 5 kata.
    from src.preprocess import clean_news
    title = "[SALAH]"
    body = "https://contoh.invalid/a/b/c/d satu dua tiga empat lima"
    assert count_words(combine_text(title, body)) >= 8
    ag = Agent(model=FakeModel(0.9), preprocess=clean_news)
    res = ag.predict(title, body)
    assert res["layer"] == "scope_guard" and "5 kata" in res["reason"]
    assert ag.model.calls == 0


def test_predict_judul_saja_yang_cukup_panjang_dinilai_model():
    res = make_agent(0.1).predict(ID_TEXT, None)
    assert res["layer"] == "model" and res["action"] == VALID


def test_predict_lapisan_similar_di_atas_ambang():
    hit = {"title": "Hoaks lama", "url": "https://contoh.invalid/hoaks", "score": 0.95}
    ag = make_agent(0.05, similar_finder=lambda t: hit)
    res = ag.predict("Judul", ID_TEXT)
    assert res["action"] == HOAKS
    assert res["layer"] == "similar"
    assert res["p_hoaks"] is None
    assert res["similar"] == hit
    assert ag.model.calls == 0


def test_predict_similar_di_bawah_ambang_tetap_ditampilkan_sebagai_pembanding():
    # Desain Metode 2.8.1 butir 4: hoaks terdekat tetap ditampilkan,
    # tetapi keputusan diambil model.
    hit = {"title": "Hoaks lama", "url": "https://contoh.invalid/hoaks", "score": 0.85}
    res = make_agent(0.05, similar_finder=lambda t: hit).predict("Judul", ID_TEXT)
    assert res["layer"] == "model" and res["action"] == VALID
    assert res["similar"] == hit


def test_predict_ambang_similar_mengikuti_config():
    hit = {"title": "H", "url": "u", "score": 0.93}
    cfg = AgentConfig(tau_similar=0.95)
    res = make_agent(0.05, similar_finder=lambda t: hit, config=cfg).predict("Judul", ID_TEXT)
    assert res["layer"] == "model"


def test_predict_kelas_model_terbalik():
    ag = Agent(model=FakeModel(0.9, classes=(1, 0)), preprocess=fake_clean)
    assert math.isclose(ag.predict("Judul", ID_TEXT)["p_hoaks"], 0.9)


def test_predict_hasil_preprocess_kosong():
    ag = Agent(model=FakeModel(0.9), preprocess=lambda t, b: "  ")
    res = ag.predict("Judul", ID_TEXT)
    assert res["action"] == PERLU_VERIFIKASI and res["p_hoaks"] is None
    assert ag.model.calls == 0


def test_predict_pesan_jelas_jika_komponen_belum_ada():
    with pytest.raises(RuntimeError, match="Anggota B"):
        Agent(model=None, preprocess=fake_clean).predict("Judul", ID_TEXT)
    with pytest.raises(RuntimeError, match="preprocessing"):
        Agent(model=FakeModel(0.5), preprocess=None).predict("Judul", ID_TEXT)


def test_predict_timing_opsional():
    res = make_agent(0.5).predict("Judul", ID_TEXT, include_timing=True)
    assert {"preprocess", "scope_guard", "model", "total"} <= set(res["timing_ms"])
    assert res["clean_text"]
    plain = make_agent(0.5).predict("Judul", ID_TEXT)
    assert "timing_ms" not in plain and "clean_text" not in plain


def test_status_lapisan():
    st = make_agent(0.5).status()
    assert st["model"] and st["preprocess"]
    assert not st["find_similar"] and not st["scope_guard_bahasa"]


def test_gabung_judul_dan_isi():
    assert combine_text("  Judul ", " isi berita ") == "Judul isi berita"
    assert combine_text(None, "isi") == "isi"


def test_load_default_agent_tanpa_model_menyebut_anggota_b(tmp_path):
    with pytest.raises(FileNotFoundError, match="Anggota B"):
        agent_mod.load_default_agent(models_dir=tmp_path)
