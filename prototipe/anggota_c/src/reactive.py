"""Lapisan reaktif agen (pemilik: Anggota C).

Isi (Desain Metode subbab 2.5):
  - scope_guard : input kosong, teks < 8 kata setelah dibersihkan, aturan bahasa.
  - LanguageRule / build_language_rule : aturan bahasa dari data LATIH.
  - SimilarIndex / choose_tau_similar : pencarian hoaks serupa (TF-IDF +
    cosine similarity, satu tetangga terdekat) dan penyetelan tau_mirip.

Definisi "kata" memakai regex yang sama dengan pipeline V3 Anggota A
(EDA_Preprocessing_V3_1.ipynb, WORD_RE, MIN_WORDS = 8), supaya batas 8 kata
di aplikasi bermakna sama dengan batas 8 kata saat dataset dibangun.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .config import DEFAULT_CONFIG, PERLU_VERIFIKASI, AgentConfig

# Sama persis dengan WORD_RE di pipeline V3 Anggota A.
WORD_RE = re.compile(r"(?u)\b[\w]+(?:[-’'][\w]+)*\b")


def _fmt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def tokenize(text: Optional[str]) -> list[str]:
    """Daftar kata (huruf kecil) menurut definisi kata V3."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


def count_words(text: Optional[str]) -> int:
    return len(WORD_RE.findall(text or ""))


# ---------------------------------------------------------------------------
# Aturan bahasa
# ---------------------------------------------------------------------------
def percentile_linear(values: list[float], q: float) -> float:
    """Persentil dengan interpolasi linear (sama dengan numpy.percentile bawaan)."""
    if not values:
        raise ValueError("Daftar nilai kosong.")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q harus di rentang [0, 100].")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class LanguageRule:
    """Skor bahasa = proporsi kata teks yang termasuk top-N kata LATIH.

    Teks dengan skor < tau dianggap di luar cakupan bahasa (misalnya bahasa
    Inggris) dan diserahkan ke manusia.
    """
    vocab: frozenset
    tau: float
    top_n: int
    percentile: float
    n_texts: int
    source: str = ""          # keterangan data asal (file, kolom, split)
    meta: dict = field(default_factory=dict)

    def score(self, text: Optional[str]) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        return sum(t in self.vocab for t in toks) / len(toks)

    def save(self, path: Path | str) -> None:
        data = {
            "vocab": sorted(self.vocab),
            "tau": self.tau,
            "top_n": self.top_n,
            "percentile": self.percentile,
            "n_texts": self.n_texts,
            "source": self.source,
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "LanguageRule":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab=frozenset(d["vocab"]), tau=float(d["tau"]), top_n=int(d["top_n"]),
            percentile=float(d["percentile"]), n_texts=int(d["n_texts"]),
            source=d.get("source", ""), meta=d.get("meta", {}),
        )


def build_language_rule(
    train_texts: Iterable[str],
    top_n: int = DEFAULT_CONFIG.lang_top_n,
    percentile: float = DEFAULT_CONFIG.lang_percentile,
    source: str = "",
) -> LanguageRule:
    """Bangun aturan bahasa dari teks LATIH saja.

    1. Hitung frekuensi kata di seluruh teks LATIH, ambil top_n kata.
       Jika frekuensi sama, urutan alfabet dipakai agar hasilnya deterministik.
    2. Hitung skor setiap teks LATIH terhadap kosakata itu.
    3. tau = persentil ke-`percentile` dari skor tersebut.
    Jangan memasukkan teks VALIDASI atau UJI ke fungsi ini.
    """
    texts = [t for t in train_texts if t and tokenize(t)]
    if not texts:
        raise ValueError("Tidak ada teks LATIH yang berisi kata.")
    freq: Counter = Counter()
    for t in texts:
        freq.update(tokenize(t))
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    vocab = frozenset(w for w, _ in top)

    tmp = LanguageRule(vocab=vocab, tau=0.0, top_n=top_n, percentile=percentile, n_texts=len(texts))
    scores = [tmp.score(t) for t in texts]
    tau = percentile_linear(scores, percentile)
    return LanguageRule(
        vocab=vocab, tau=tau, top_n=top_n, percentile=percentile, n_texts=len(texts),
        source=source,
        meta={"score_min": min(scores), "score_median": percentile_linear(scores, 50.0)},
    )


# ---------------------------------------------------------------------------
# scope_guard
# ---------------------------------------------------------------------------
def scope_guard(
    text: Optional[str],
    config: AgentConfig = DEFAULT_CONFIG,
    language_rule: Optional[LanguageRule] = None,
) -> Optional[dict]:
    """Periksa apakah teks layak dinilai model (Desain Metode 2.5.1).

    `text` adalah teks SETELAH dibersihkan clean_text, sesuai Desain Metode:
    "Jika isi teks setelah dibersihkan kurang dari 8 kata ...".
    Mengembalikan None jika lolos, atau {"action": "PERLU VERIFIKASI",
    "reason": ...}. Urutan: kosong -> panjang < min_words -> bahasa.
    """
    n = count_words(text)
    if n == 0:
        return {"action": PERLU_VERIFIKASI,
                "reason": "Di luar cakupan: input kosong, tidak ada teks untuk dinilai"}
    if n < config.min_words:
        return {
            "action": PERLU_VERIFIKASI,
            "reason": f"Di luar cakupan: teks terlalu pendek ({n} kata, minimal {config.min_words} kata)",
        }
    if language_rule is not None:
        s = language_rule.score(text)
        if s < language_rule.tau:
            return {
                "action": PERLU_VERIFIKASI,
                "reason": (
                    f"Di luar cakupan: kemungkinan bukan bahasa Indonesia (skor bahasa "
                    f"{_fmt(s)} di bawah ambang {_fmt(language_rule.tau)})"
                ),
            }
    return None


# ---------------------------------------------------------------------------
# Pencarian hoaks serupa (Desain Metode 2.5.2)
# ---------------------------------------------------------------------------
# Pengaturan TF-IDF indeks. ngram dan sublinear_tf disamakan dengan TF-IDF
# model Anggota B; min_df=1 karena indeks harus memuat setiap hoaks,
# termasuk kata yang hanya muncul di satu hoaks.
SIMILAR_TFIDF_PARAMS = {"ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1}


class SimilarIndex:
    """Indeks TF-IDF hoaks LATIH dan pencarian satu tetangga terdekat.

    Vektor TF-IDF ternormalisasi L2, sehingga cosine similarity sama dengan
    hasil kali titik (dot product).
    """

    def __init__(self, vectorizer, matrix, titles, urls, raw_ids, meta=None):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.titles = list(titles)
        self.urls = list(urls)
        self.raw_ids = list(raw_ids)
        self.meta = dict(meta or {})

    @classmethod
    def build(cls, texts, titles, urls, raw_ids, meta=None, **tfidf_params):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = list(texts)
        if not texts:
            raise ValueError("Tidak ada teks hoaks untuk diindeks.")
        params = {**SIMILAR_TFIDF_PARAMS, **tfidf_params}
        vec = TfidfVectorizer(**params)
        matrix = vec.fit_transform(texts)
        meta = {**(meta or {}), "n_hoaks": len(texts), "tfidf_params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}}
        return cls(vec, matrix, titles, urls, raw_ids, meta)

    def scores_many(self, clean_texts, chunk: int = 512):
        """(indeks_terdekat, skor_terdekat) untuk banyak teks sekaligus."""
        import numpy as np

        best_idx, best_score = [], []
        texts = list(clean_texts)
        for i in range(0, len(texts), chunk):
            q = self.vectorizer.transform(texts[i:i + chunk])
            sims = (q @ self.matrix.T).toarray()
            best_idx.extend(sims.argmax(axis=1).tolist())
            best_score.extend(sims.max(axis=1).tolist())
        return np.array(best_idx), np.array(best_score)

    def query(self, clean_text: str) -> Optional[dict]:
        """Hoaks LATIH paling mirip, atau None jika tidak ada kata yang sama."""
        q = self.vectorizer.transform([clean_text or ""])
        if q.nnz == 0:
            return None
        sims = (q @ self.matrix.T).toarray().ravel()
        i = int(sims.argmax())
        return {
            "title": self.titles[i],
            "url": self.urls[i],
            "score": float(sims[i]),
            "raw_id": self.raw_ids[i],
        }

    __call__ = query

    def save(self, path) -> None:
        """Simpan dalam format portabel: hanya kosakata, bobot IDF, matriks scipy,
        dan pengaturan TF-IDF. Tidak ada objek scikit-learn yang di-pickle, sehingga
        indeks yang dibangun di satu versi scikit-learn bisa dimuat di versi lain."""
        import joblib
        import numpy as np

        v = self.vectorizer
        joblib.dump(
            {"format": "similar_index_portabel_v1",
             "vocabulary": {str(k): int(i) for k, i in v.vocabulary_.items()},
             "idf": np.asarray(v.idf_, dtype=np.float64),
             "tfidf_params": {k: v.get_params()[k] for k in
                              ("ngram_range", "sublinear_tf", "min_df", "lowercase", "norm",
                               "use_idf", "smooth_idf", "token_pattern", "analyzer", "strip_accents")},
             "matrix": self.matrix.tocsr(), "titles": self.titles, "urls": self.urls,
             "raw_ids": self.raw_ids, "meta": self.meta},
            path, compress=3,
        )

    @classmethod
    def load(cls, path) -> "SimilarIndex":
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer

        d = joblib.load(path)
        if d.get("format") != "similar_index_portabel_v1":
            raise ValueError(f"Format indeks tidak dikenal di {path}. Bangun ulang dengan scripts/build_similar_index.py.")
        params = dict(d["tfidf_params"])
        params["ngram_range"] = tuple(params["ngram_range"])
        params.pop("min_df", None)  # tidak dipakai saat kosakata sudah tetap
        vec = TfidfVectorizer(vocabulary=d["vocabulary"], **params)
        vec.idf_ = d["idf"]
        return cls(vec, d["matrix"], d["titles"], d["urls"], d["raw_ids"], d.get("meta"))


def tau_table(scores, labels, grid) -> list[dict]:
    """Untuk setiap ambang t: berapa berita VALIDASI yang cocok (skor >= t),
    berapa yang benar HOAKS, dan precision aturan hoaks serupa."""
    rows = []
    n_hoaks_total = int(sum(1 for y in labels if int(y) == 1))
    for t in grid:
        t = round(float(t), 2)
        hit = [int(y) for s, y in zip(scores, labels) if s >= t]
        n, tp = len(hit), sum(hit)
        rows.append({
            "tau": t,
            "n_cocok": n,
            "n_cocok_hoaks": tp,
            "n_cocok_valid": n - tp,
            "precision": (tp / n) if n else None,
            "coverage_hoaks": (tp / n_hoaks_total) if n_hoaks_total else None,
        })
    return rows


def choose_tau_similar(table: list[dict], floor: float = 0.90, min_precision: float = 0.99) -> Optional[float]:
    """Ambang terkecil >= floor dengan precision >= min_precision.

    Mengembalikan None jika tidak ada ambang yang memenuhi (misalnya tidak ada
    berita VALIDASI yang cocok sama sekali), sehingga precision tidak bisa
    dihitung. Keputusan dalam kasus itu diserahkan ke kelompok.
    """
    ok = [r["tau"] for r in table
          if r["tau"] >= floor - 1e-9 and r["n_cocok"] > 0 and r["precision"] >= min_precision]
    return min(ok) if ok else None
