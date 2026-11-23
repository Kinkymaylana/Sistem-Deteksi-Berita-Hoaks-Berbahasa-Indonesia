"""Penjelasan keputusan model (pemilik: Anggota C).

Dua jenis penjelasan untuk tombol "Jelaskan" (Rencana Kerja, Anggota C butir 4;
Desain Metode 2.8.1 butir 5 dan Tabel 2.1):

1. Kontribusi kata SVM (cepat, eksak untuk model linear)
   kontribusi(kata) = nilai TF-IDF kata di teks x bobot SVM kata itu.
   Jumlah seluruh kontribusi + intercept = skor keputusan SVM (decision_function).
   Platt scaling mengubah skor itu menjadi peluang secara monoton, sehingga
   kata berkontribusi positif mendorong p(HOAKS) naik bila koefisien Platt a < 0.

2. LIME (Ribeiro et al., 2016), dijalankan hanya atas permintaan karena perlu
   ratusan sampai ribuan prediksi tambahan. Memakai pustaka `lime` resmi.

Struktur model yang didukung (dari notebooks/anggota_b/scripts/B3_calibrate.py):
    CalibratedClassifierCV(estimator=Pipeline([('tfidf', TfidfVectorizer),
                                               ('classifier', LinearSVC)]),
                           method='sigmoid', cv='prefit')
Pada scikit-learn >= 1.6, kalibrasi prefit juga bisa dibungkus FrozenEstimator;
keduanya ditangani _unwrap().
"""
from __future__ import annotations

from typing import Any


def _unwrap(model: Any):
    """Kembalikan (pipeline TF-IDF+SVM, kalibrator Platt atau None)."""
    calibrator = None
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        cc = model.calibrated_classifiers_[0]
        base = cc.estimator
        if not hasattr(base, "named_steps") and hasattr(base, "estimator"):
            base = base.estimator  # FrozenEstimator
        cals = getattr(cc, "calibrators", None) or getattr(cc, "calibrators_", None)
        calibrator = cals[0] if cals else None
    if not hasattr(base, "named_steps"):
        raise TypeError("Model bukan Pipeline TF-IDF + SVM yang dikenal; kontribusi kata tidak bisa dihitung.")
    steps = list(base.named_steps.values())
    return base, steps[0], steps[-1], calibrator


def svm_word_contributions(model: Any, clean_text: str, top_k: int = 10) -> dict:
    """Kata dengan kontribusi terbesar ke arah HOAKS dan ke arah VALID."""
    pipe, tfidf, clf, calibrator = _unwrap(model)
    if not hasattr(clf, "coef_"):
        raise TypeError("Classifier tidak memiliki coef_ (bukan model linear).")
    x = tfidf.transform([clean_text])
    coef = clf.coef_.ravel()
    terms = tfidf.get_feature_names_out()
    idx = x.indices
    contrib = x.data * coef[idx]
    intercept = float(clf.intercept_.ravel()[0])
    decision = float(contrib.sum() + intercept)

    # Arah Platt: p = 1 / (1 + exp(a*f + b)); a < 0 berarti f naik -> p naik.
    a = getattr(calibrator, "a_", None)
    sign = -1.0 if (a is not None and a > 0) else 1.0
    signed = contrib * sign

    pairs = sorted(zip((terms[i] for i in idx), signed), key=lambda t: t[1])
    to_hoaks = [(w, float(c)) for w, c in reversed(pairs) if c > 0][:top_k]
    to_valid = [(w, float(c)) for w, c in pairs if c < 0][:top_k]
    return {
        "ke_hoaks": to_hoaks,
        "ke_valid": to_valid,
        "skor_svm": decision,
        "intercept": intercept,
        "platt_a": None if a is None else float(a),
        "n_kata_dikenal": int(len(idx)),
    }


def lime_explanation(model: Any, clean_text: str, num_features: int = 10,
                     num_samples: int = 1000, random_state: int = 42) -> list[tuple[str, float]]:
    """Bobot LIME untuk kelas HOAKS. Positif = mendorong ke HOAKS."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as e:
        raise ImportError("Pustaka `lime` belum terpasang. Jalankan: pip install lime") from e

    classes = list(getattr(model, "classes_", [0, 1]))
    hoaks_col = classes.index(1)
    explainer = LimeTextExplainer(class_names=["VALID", "HOAKS"] if hoaks_col == 1 else ["HOAKS", "VALID"],
                                  random_state=random_state)
    exp = explainer.explain_instance(clean_text, model.predict_proba, num_features=num_features,
                                     labels=(hoaks_col,), num_samples=num_samples)
    return [(w, float(v)) for w, v in exp.as_list(label=hoaks_col)]
