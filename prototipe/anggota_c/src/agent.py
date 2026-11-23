"""Agen hybrid: decide(p) dan predict(title, body) (pemilik: Anggota C).

Alur predict() mengikuti Desain Metode Bab 2 dan Rencana Kerja Tahap 3 bagian 4:

    title, body
        |
    [0] Persepsi     clean_news(title, body)      -- src/preprocess.py
        |
    [1] scope_guard  kosong / < 8 kata SETELAH dibersihkan / bukan bahasa Indonesia
        |            -> PERLU VERIFIKASI, layer="scope_guard", p_hoaks=None
        |
    [2] find_similar hoaks LATIH terdekat, skor >= tau_mirip
        |            -> HOAKS, layer="similar", p_hoaks=None
        |
    [3] model        p = P(HOAKS) dari Linear SVM terkalibrasi Platt (Anggota B)
        |
    decide(p)        VALID / PERLU VERIFIKASI / HOAKS, layer="model"

Untuk berita yang dinilai model, hoaks lama terdekat tetap dikembalikan di
kunci "similar" sebagai pembanding, tetapi tidak memengaruhi keputusan
(Desain Metode 2.8.1 butir 4).

Komponen disuntikkan lewat konstruktor Agent, sehingga agen bisa diuji dengan
komponen tiruan dan dirakit dari file asli lewat load_default_agent().
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .config import (
    DEFAULT_CONFIG,
    HOAKS,
    LAYER_MODEL,
    LAYER_SCOPE_GUARD,
    LAYER_SIMILAR,
    PERLU_VERIFIKASI,
    VALID,
    AgentConfig,
    thresholds_from_costs,
)
from .reactive import LanguageRule, SimilarIndex, count_words, scope_guard

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
# Nama file dari Anggota B (repo: results/anggota_b/Biner Model SVM Calibrated/)
# dan nama di Rencana Kerja bagian 4. File pertama yang ada yang dipakai.
MODEL_FILES = ("LinearSVM_calibrated_v31.joblib", "LinearSVM_calibrated.joblib")
MODEL_FILE = MODEL_FILES[0]
MODEL_CARD_FILES = ("model_card.json", "B3_MODEL_CARD_BASELINE.json")
MODEL_CARD_FILE = MODEL_CARD_FILES[0]
LANGUAGE_RULE_FILE = "language_rule.json"
SIMILAR_INDEX_FILE = "similar_index.joblib"


def _fmt(x: float) -> str:
    """Angka dengan koma desimal, dua angka di belakang koma (0,20)."""
    return f"{x:.2f}".replace(".", ",")


# ---------------------------------------------------------------------------
# decide(p)
# ---------------------------------------------------------------------------
def decide(p: float, config: AgentConfig = DEFAULT_CONFIG) -> tuple[str, str]:
    """Ubah peluang HOAKS p menjadi tindakan dan alasannya.

    VALID jika p <= 0,20, HOAKS jika p >= 0,50, PERLU VERIFIKASI jika di antaranya.
    Ambang dihitung dari biaya di config (lihat src/config.py).
    Melempar ValueError jika p bukan angka di rentang [0, 1].
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError(f"p harus angka, bukan {type(p).__name__}.")
    p = float(p)
    if math.isnan(p) or not 0.0 <= p <= 1.0:
        raise ValueError(f"p harus di rentang [0, 1], didapat {p!r}.")

    t_valid, t_hoaks = thresholds_from_costs(config.c_fp, config.c_fn, config.c_r)
    if p <= t_valid:
        return VALID, f"p = {_fmt(p)} tidak melebihi ambang VALID {_fmt(t_valid)}"
    if p >= t_hoaks:
        return HOAKS, f"p = {_fmt(p)} mencapai ambang HOAKS {_fmt(t_hoaks)}"
    return (
        PERLU_VERIFIKASI,
        f"p = {_fmt(p)} berada di antara {_fmt(t_valid)} dan {_fmt(t_hoaks)}; "
        f"model tidak cukup yakin",
    )


def combine_text(title: Optional[str], body: Optional[str]) -> str:
    """Gabungkan judul dan isi dengan satu spasi, judul di depan (Proposal 3.4.1)."""
    parts = [s.strip() for s in (title or "", body or "") if s and s.strip()]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class Agent:
    """Agen hybrid dengan komponen yang bisa diganti.

    model          : objek dengan predict_proba(list[str]); kelas HOAKS = 1.
                     Bentuk dari Anggota B: CalibratedClassifierCV yang membungkus
                     Pipeline([('tfidf', ...), ('classifier', LinearSVC)]).
    preprocess     : fungsi (title, body) -> teks bersih, misalnya
                     src.preprocess.clean_news.
    similar_finder : fungsi (teks bersih) -> {"title","url","score"} atau None,
                     misalnya SimilarIndex.query. Ambang tau_mirip diterapkan di sini.
    language_rule  : LanguageRule dari data LATIH.
    """

    def __init__(
        self,
        model: Any = None,
        preprocess: Optional[Callable[[Optional[str], Optional[str]], str]] = None,
        similar_finder: Optional[Callable[[str], Optional[dict]]] = None,
        language_rule: Optional[LanguageRule] = None,
        config: AgentConfig = DEFAULT_CONFIG,
        warnings: Optional[list[str]] = None,
    ) -> None:
        self.model = model
        self.preprocess = preprocess
        self.similar_finder = similar_finder
        self.language_rule = language_rule
        self.config = config
        self.warnings = list(warnings or [])
        thresholds_from_costs(config.c_fp, config.c_fn, config.c_r)  # validasi awal

    def status(self) -> dict:
        """Lapisan mana yang aktif, untuk ditampilkan jujur di aplikasi."""
        return {
            "preprocess": self.preprocess is not None,
            "preprocess_version": getattr(self, "preprocess_version", None),
            "scope_guard_panjang": True,
            "scope_guard_bahasa": self.language_rule is not None,
            "find_similar": self.similar_finder is not None,
            "model": self.model is not None,
            "tau_similar": self.config.tau_similar,
        }

    def clean(self, title: Optional[str], body: Optional[str]) -> str:
        if self.preprocess is None:
            raise RuntimeError("Fungsi preprocessing belum dipasang (src/preprocess.py).")
        return self.preprocess(title, body)

    def hoax_probability(self, clean: str) -> float:
        """P(HOAKS) dari model untuk satu teks yang sudah dibersihkan."""
        if self.model is None:
            raise RuntimeError(
                f"Model belum dipasang. Letakkan {MODEL_FILE} dari Anggota B di folder models/."
            )
        proba = self.model.predict_proba([clean])[0]
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 1 not in classes:
            raise RuntimeError(f"Kelas HOAKS (1) tidak ada di model.classes_ = {classes}.")
        return float(proba[classes.index(1)])

    def predict(self, title: Optional[str], body: Optional[str], include_timing: bool = False) -> dict:
        timing: dict[str, float] = {}

        def out(action, layer, p, reason, similar=None, clean=None):
            result = {
                "action": action,
                "layer": layer,
                "p_hoaks": p,
                "reason": reason,
                "similar": similar,
                "model": self.config.model_name,
            }
            if include_timing:
                result["timing_ms"] = timing
                result["clean_text"] = clean
            return result

        t_start = time.perf_counter()

        # Input benar-benar kosong: tidak perlu dibersihkan.
        if count_words(combine_text(title, body)) == 0:
            guard = scope_guard("", self.config)
            timing["total"] = (time.perf_counter() - t_start) * 1000
            return out(guard["action"], LAYER_SCOPE_GUARD, None, guard["reason"])

        # [0] Persepsi
        t0 = time.perf_counter()
        clean = self.clean(title, body)
        timing["preprocess"] = (time.perf_counter() - t0) * 1000

        # [1] scope_guard pada teks bersih
        t0 = time.perf_counter()
        guard = scope_guard(clean, self.config, self.language_rule)
        timing["scope_guard"] = (time.perf_counter() - t0) * 1000
        if guard is not None:
            timing["total"] = (time.perf_counter() - t_start) * 1000
            return out(guard["action"], LAYER_SCOPE_GUARD, None, guard["reason"], clean=clean)

        # [2] find_similar
        nearest = None
        if self.similar_finder is not None:
            t0 = time.perf_counter()
            hit = self.similar_finder(clean)
            timing["similar"] = (time.perf_counter() - t0) * 1000
            if hit is not None:
                nearest = {"title": hit.get("title"), "url": hit.get("url"), "score": float(hit["score"])}
                if nearest["score"] >= self.config.tau_similar:
                    timing["total"] = (time.perf_counter() - t_start) * 1000
                    return out(
                        HOAKS, LAYER_SIMILAR, None,
                        f"Mirip hoaks yang pernah diperiksa (cosine {_fmt(nearest['score'])} "
                        f"mencapai ambang {_fmt(self.config.tau_similar)})",
                        nearest, clean,
                    )

        # [3] model + decide
        t0 = time.perf_counter()
        p = self.hoax_probability(clean)
        action, reason = decide(p, self.config)
        timing["model"] = (time.perf_counter() - t0) * 1000
        timing["total"] = (time.perf_counter() - t_start) * 1000
        return out(action, LAYER_MODEL, p, reason, nearest, clean)


# ---------------------------------------------------------------------------
# Merakit agen dari file asli di models/
# ---------------------------------------------------------------------------
def read_model_card(models_dir: Path = MODELS_DIR) -> tuple[dict, list[str]]:
    """Baca model card B dan bandingkan versi scikit-learn dengan yang terpasang."""
    warnings: list[str] = []
    card_path = next((Path(models_dir) / f for f in MODEL_CARD_FILES if (Path(models_dir) / f).exists()), None)
    if card_path is None:
        return {}, [f"Model card ({' / '.join(MODEL_CARD_FILES)}) tidak ditemukan; "
                    f"versi data dan scikit-learn model tidak bisa dicek."]
    import sklearn

    card = json.loads(card_path.read_text(encoding="utf-8"))
    want = card.get("environment", {}).get("scikit_learn_version")
    if want and want != sklearn.__version__:
        warnings.append(
            f"Model dibuat dengan scikit-learn {want}, terpasang {sklearn.__version__}. "
            f"Pasang versi yang sama (lihat requirements.txt); hasil di versi lain tidak boleh dilaporkan."
        )
    return card, warnings


def preprocess_version_from_card(card: dict) -> str:
    """'V3.1' jika model dilatih pada text_clean_v31, selain itu 'V3'."""
    return "V3.1" if "V3.1" in str(card.get("data_version", "")) else "V3"


def load_default_agent(models_dir: Path = MODELS_DIR, config: AgentConfig = DEFAULT_CONFIG) -> Agent:
    """Rakit agen dari file asli kelompok.

    Wajib  : models/LinearSVM_calibrated_v31.joblib (atau LinearSVM_calibrated.joblib) dari Anggota B
    Disarankan: models/model_card.json (B3_MODEL_CARD_BASELINE.json) -- menentukan versi preprocessing
    Opsional (lapisan dinonaktifkan jika tidak ada, terlihat di Agent.status()):
             models/language_rule.json   (scripts/build_language_rule.py)
             models/similar_index.joblib (scripts/build_similar_index.py)
    """
    from .preprocess import clean_news

    models_dir = Path(models_dir)
    model_path = next((models_dir / f for f in MODEL_FILES if (models_dir / f).exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"{' / '.join(MODEL_FILES)} tidak ada di {models_dir}. File ini diserahkan Anggota B.")
    import joblib

    card, warnings = read_model_card(models_dir)
    version = preprocess_version_from_card(card) if card else "V3.1"
    if not card:
        warnings.append("Versi preprocessing diasumsikan V3.1 karena model card tidak ada.")
    model = joblib.load(model_path)

    rule_path = models_dir / LANGUAGE_RULE_FILE
    rule = LanguageRule.load(rule_path) if rule_path.exists() else None
    if rule is None:
        warnings.append("Aturan bahasa belum dibangun; teks non-Indonesia tidak disaring.")

    finder = None
    index_path = models_dir / SIMILAR_INDEX_FILE
    if index_path.exists():
        index = SimilarIndex.load(index_path)
        finder = index.query
        tuned = index.meta.get("tau_similar")
        if tuned is not None:
            config = dataclasses.replace(config, tau_similar=float(tuned))
        else:
            warnings.append("tau_mirip belum disetel di VALIDASI; memakai nilai awal 0,90.")
    else:
        warnings.append("Indeks hoaks serupa belum dibangun; lapisan find_similar nonaktif.")

    agent = Agent(model=model, preprocess=functools.partial(clean_news, v31=(version == "V3.1")),
                  similar_finder=finder, language_rule=rule, config=config, warnings=warnings)
    agent.preprocess_version = version
    agent.model_path = model_path
    agent.model_card = card
    return agent


_DEFAULT_AGENT: Optional[Agent] = None


def predict(title: Optional[str], body: Optional[str]) -> dict:
    """Format keluaran yang disepakati (Rencana Kerja Tahap 3 bagian 4):

    {
      "action": "HOAKS" | "VALID" | "PERLU VERIFIKASI",
      "layer":  "scope_guard" | "similar" | "model",
      "p_hoaks": float | None,   # None jika diputuskan lapisan reaktif
      "reason": str,
      "similar": {"title", "url", "score"} | None,
      "model": "LinearSVM-platt"
    }
    """
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = load_default_agent()
    return _DEFAULT_AGENT.predict(title, body)
