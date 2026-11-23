"""Modul evaluasi bersama (pemilik: Anggota D).

Dipakai oleh skrip B (B1_tune.py, B2_run_experiments.py, B3_calibrate.py),
skrip evaluasi agen D, dan siapa pun yang perlu menulis ke log eksperimen.

Fungsi publik (nama dan argumen mengikuti pemanggilan di skrip B):
  calculate_metrics(y_true, y_pred, y_prob=None, ...)  -> dict metrik
  compute_ece(y_true, y_prob, n_bins=10)               -> float
  compute_reliability_bins(y_true, y_prob, n_bins=10)  -> list[dict]
  precision_at_k(y_true, y_prob, k)                    -> float
  bootstrap_macro_f1_ci(y_true, y_pred, ...)           -> (bawah, atas)
  paired_bootstrap_test(y_true, pred_a, pred_b, ...)   -> dict
  evaluate_selective_prediction(y_true, y_prob, ...)   -> dict
  log_experiment(results_csv_path, run_id, ...)        -> dict baris log

Konvensi (Rencana Kerja Tahap 3, bagian Anggota D):
  - Label: 1 = HOAKS (kelas positif), 0 = VALID.
  - y_prob = peluang HOAKS.
  - ECE: 10 bin selebar 0,1 pada peluang HOAKS, |rerata peluang - proporsi
    HOAKS| ditimbang jumlah anggota bin. Bin terakhir memuat p = 1,0.
  - CI 95% macro-F1: bootstrap persentil 1000 kali, seed 42.
  - Aturan penolakan (Chow): VALID jika p <= tau_low, HOAKS jika p >= tau_high,
    selain itu PERLU VERIFIKASI. Bawaan tau_low = C_R/C_FN = 0,20 dan
    tau_high = 1 - C_R/C_FP = 0,50 (C_FP = 2, C_FN = 5, C_R = 1), sama dengan
    src/config.py milik C.
  - Kerugian harapan = (C_FP*FP + C_FN*FN + C_R*jumlah_ditolak) / N.
  - Semua angka dibulatkan 6 desimal, CI ditulis "[0.1234, 0.5678]".

Log eksperimen (results/experiments.csv) memakai 25 kolom yang sama dengan
results/anggota_b/B2_EXPERIMENTS_BASELINE.csv, lihat EXPERIMENT_COLUMNS.
Baris berstatus UJI hanya bisa ditulis dengan allow_uji=True. Skrip UJI
tidak memanggil ini langsung, tetapi lewat UjiSession.log_experiment di
src/uji_lock.py (flag --final, manifest pembekuan, uji_log.csv).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
POS_LABEL = 1  # HOAKS
NEG_LABEL = 0  # VALID

DEFAULT_N_BINS = 10
DEFAULT_N_BOOTSTRAPS = 1000
DEFAULT_SEED = 42
DEFAULT_K = (50, 100)

DEFAULT_C_FP = 2.0
DEFAULT_C_FN = 5.0
DEFAULT_C_R = 1.0

ROUND = 6

STATUS_VALIDASI = "VALIDASI"
STATUS_UJI = "UJI"

EXPERIMENT_COLUMNS = [
    "run_id", "tanggal", "versi_data", "kondisi", "split", "model",
    "hyperparameter", "seed", "status_evaluasi",
    "macro_f1", "macro_f1_ci_95", "accuracy",
    "f1_hoaks", "precision_hoaks", "recall_hoaks",
    "f1_valid", "precision_valid", "recall_valid",
    "ece", "brier_score", "precision_at_50", "precision_at_100",
    "coverage_agen", "selective_macro_f1", "expected_loss",
]

# Kolom metrik di log yang diambil langsung dari hasil calculate_metrics().
_METRIC_COLUMNS = EXPERIMENT_COLUMNS[9:]


# ---------------------------------------------------------------------------
# Bantuan
# ---------------------------------------------------------------------------
def _r(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    if np.isnan(x):
        return None
    return round(x, ROUND)


def _as_int_array(y: Iterable) -> np.ndarray:
    arr = np.asarray(list(y) if not hasattr(y, "__len__") else y)
    if arr.dtype.kind in "OUS":  # label teks, misalnya "HOAKS"/"VALID"
        mapping = {"HOAKS": POS_LABEL, "VALID": NEG_LABEL, "1": 1, "0": 0}
        arr = np.array([mapping[str(v).strip().upper()] for v in arr])
    arr = arr.astype(int)
    bad = set(np.unique(arr)) - {0, 1}
    if bad:
        raise ValueError(f"Label harus 0 (VALID) atau 1 (HOAKS), ditemukan: {sorted(bad)}")
    return arr


def _as_prob_array(p: Iterable, n: int) -> np.ndarray:
    arr = np.asarray(p, dtype=float).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"Panjang y_prob ({arr.shape[0]}) tidak sama dengan y_true ({n})")
    if np.any(np.isnan(arr)):
        raise ValueError("y_prob berisi NaN")
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("y_prob harus di rentang [0, 1]; ubah skor SVM dengan sigmoid atau kalibrasi dulu")
    return arr


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)


def format_ci(low: float, high: float) -> str:
    return f"[{low:.4f}, {high:.4f}]"


# ---------------------------------------------------------------------------
# Metrik klasifikasi
# ---------------------------------------------------------------------------
def bootstrap_macro_f1_ci(
    y_true, y_pred, n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_SEED, alpha: float = 0.05,
) -> tuple[float, float]:
    """CI persentil macro-F1 dengan resampling baris (dengan pengembalian)."""
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    scores = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        scores[i] = _macro_f1(y_true[idx], y_pred[idx])
    low = float(np.percentile(scores, 100 * alpha / 2))
    high = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return low, high


def precision_at_k(y_true, y_prob, k: int) -> float | None:
    """Proporsi HOAKS di antara k berita dengan peluang HOAKS tertinggi.

    Urutan stabil (mergesort), sehingga peluang yang sama tetap urut sesuai
    baris asli. None jika data kurang dari k.
    """
    y_true = _as_int_array(y_true)
    y_prob = _as_prob_array(y_prob, len(y_true))
    if len(y_true) < k:
        return None
    order = np.argsort(-y_prob, kind="mergesort")[:k]
    return float(y_true[order].mean())


def compute_reliability_bins(y_true, y_prob, n_bins: int = DEFAULT_N_BINS) -> list[dict]:
    """Isi diagram reliabilitas: per bin [low, high), bin terakhir [0,9, 1,0].

    Bin kosong tetap dilaporkan dengan count 0 dan nilai rerata None.
    """
    y_true = _as_int_array(y_true)
    y_prob = _as_prob_array(y_prob, len(y_true))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.floor(y_prob * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        rows.append({
            "bin_index": b,
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "count": cnt,
            "mean_predicted_prob": float(y_prob[mask].mean()) if cnt else None,
            "empirical_positive_rate": float(y_true[mask].mean()) if cnt else None,
        })
    return rows


def compute_ece(y_true, y_prob, n_bins: int = DEFAULT_N_BINS) -> float:
    y_true = _as_int_array(y_true)
    n = len(y_true)
    ece = 0.0
    for b in compute_reliability_bins(y_true, y_prob, n_bins):
        if b["count"]:
            ece += b["count"] / n * abs(b["mean_predicted_prob"] - b["empirical_positive_rate"])
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = _as_int_array(y_true)
    y_prob = _as_prob_array(y_prob, len(y_true))
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Prediksi selektif (aturan penolakan berbasis biaya)
# ---------------------------------------------------------------------------
def thresholds_from_costs(c_fp: float = DEFAULT_C_FP, c_fn: float = DEFAULT_C_FN,
                          c_r: float = DEFAULT_C_R) -> tuple[float, float]:
    """tau_low = C_R/C_FN, tau_high = 1 - C_R/C_FP (sama dengan config.py C)."""
    return c_r / c_fn, 1.0 - c_r / c_fp


def selective_decisions(y_prob, tau_low: float, tau_high: float) -> np.ndarray:
    """1 = HOAKS, 0 = VALID, -1 = PERLU VERIFIKASI (ditolak)."""
    p = np.asarray(y_prob, dtype=float)
    out = np.full(p.shape, -1, dtype=int)
    out[p <= tau_low] = NEG_LABEL
    out[p >= tau_high] = POS_LABEL
    return out


def evaluate_selective_prediction(
    y_true, y_prob, tau_low: float | None = None, tau_high: float | None = None,
    c_fp: float = DEFAULT_C_FP, c_fn: float = DEFAULT_C_FN, c_r: float = DEFAULT_C_R,
    y_pred_standard=None,
) -> dict:
    """Bandingkan keputusan selektif dengan keputusan standar (p >= 0,5).

    y_pred_standard boleh diberikan bila prediksi standar bukan p >= 0,5.
    """
    y_true = _as_int_array(y_true)
    y_prob = _as_prob_array(y_prob, len(y_true))
    n = len(y_true)
    d_low, d_high = thresholds_from_costs(c_fp, c_fn, c_r)
    tau_low = d_low if tau_low is None else tau_low
    tau_high = d_high if tau_high is None else tau_high
    if tau_low > tau_high:
        raise ValueError(f"tau_low ({tau_low}) lebih besar dari tau_high ({tau_high})")

    dec = selective_decisions(y_prob, tau_low, tau_high)
    decided = dec != -1
    n_dec = int(decided.sum())
    reject = n - n_dec
    yt, yd = y_true[decided], dec[decided]
    fp_sel = int(((yd == 1) & (yt == 0)).sum())
    fn_sel = int(((yd == 0) & (yt == 1)).sum())

    std = (y_prob >= 0.5).astype(int) if y_pred_standard is None else _as_int_array(y_pred_standard)
    fp_std = int(((std == 1) & (y_true == 0)).sum())
    fn_std = int(((std == 0) & (y_true == 1)).sum())

    sel_acc = float((yt == yd).mean()) if n_dec else None
    return {
        "coverage": _r(n_dec / n),
        "abstain_rate": _r(reject / n),
        "selective_accuracy": _r(sel_acc),
        "selective_error_rate": _r(1 - sel_acc) if sel_acc is not None else None,
        "selective_macro_f1": _r(_macro_f1(yt, yd)) if n_dec else None,
        "fp_count_selective": fp_sel,
        "fn_count_selective": fn_sel,
        "fn_count_standard": fn_std,
        "reject_count": reject,
        "expected_loss_selective": _r((c_fp * fp_sel + c_fn * fn_sel + c_r * reject) / n),
        "expected_loss_standard": _r((c_fp * fp_std + c_fn * fn_std) / n),
    }


# ---------------------------------------------------------------------------
# Metrik lengkap satu kondisi
# ---------------------------------------------------------------------------
def calculate_metrics(
    y_true, y_pred, y_prob=None, n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_SEED, n_bins: int = DEFAULT_N_BINS, ks: Iterable[int] = DEFAULT_K,
    c_fp: float = DEFAULT_C_FP, c_fn: float = DEFAULT_C_FN, c_r: float = DEFAULT_C_R,
    tau_low: float | None = None, tau_high: float | None = None,
) -> dict:
    """Semua metrik Tahap 2 untuk satu kondisi.

    Tanpa y_prob, metrik berbasis peluang (ECE, Brier, Precision@k,
    coverage, kerugian) bernilai None.

    Catatan tafsir: untuk model tanpa kalibrasi (sigmoid dari skor SVM),
    ECE dan coverage memang dihitung tetapi tidak bermakna sebagai ukuran
    keyakinan. Laporkan hanya untuk model terkalibrasi.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Panjang y_true dan y_pred tidak sama")
    if len(y_true) == 0:
        raise ValueError("Data kosong")

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[NEG_LABEL, POS_LABEL], zero_division=0)
    low, high = bootstrap_macro_f1_ci(y_true, y_pred, n_bootstraps, seed)
    tn, fp, fn, tp = _confusion(y_true, y_pred)

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "macro_f1": _r(_macro_f1(y_true, y_pred)),
        "macro_f1_ci_95": format_ci(low, high),
        "macro_f1_ci_low": _r(low),
        "macro_f1_ci_high": _r(high),
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "f1_hoaks": _r(f[1]), "precision_hoaks": _r(p[1]), "recall_hoaks": _r(r[1]),
        "f1_valid": _r(f[0]), "precision_valid": _r(p[0]), "recall_valid": _r(r[0]),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "ece": None, "brier_score": None,
        "coverage_agen": None, "selective_macro_f1": None, "expected_loss": None,
    }
    for k in ks:
        out[f"precision_at_{k}"] = None

    if y_prob is not None:
        prob = _as_prob_array(y_prob, len(y_true))
        out["ece"] = _r(compute_ece(y_true, prob, n_bins))
        out["brier_score"] = _r(brier_score(y_true, prob))
        for k in ks:
            out[f"precision_at_{k}"] = _r(precision_at_k(y_true, prob, k))
        sel = evaluate_selective_prediction(
            y_true, prob, tau_low=tau_low, tau_high=tau_high,
            c_fp=c_fp, c_fn=c_fn, c_r=c_r, y_pred_standard=y_pred)
        out["coverage_agen"] = sel["coverage"]
        out["selective_macro_f1"] = sel["selective_macro_f1"]
        out["expected_loss"] = sel["expected_loss_selective"]
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def confusion_matrix_dict(y_true, y_pred) -> dict:
    """Matriks kebingungan dengan nama baris/kolom yang jelas."""
    tn, fp, fn, tp = _confusion(_as_int_array(y_true), _as_int_array(y_pred))
    return {"asli_VALID_pred_VALID": tn, "asli_VALID_pred_HOAKS": fp,
            "asli_HOAKS_pred_VALID": fn, "asli_HOAKS_pred_HOAKS": tp}


# ---------------------------------------------------------------------------
# Uji signifikansi berpasangan
# ---------------------------------------------------------------------------
def paired_bootstrap_test(
    y_true, pred_a, pred_b, n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    seed: int = DEFAULT_SEED, alpha: float = 0.05,
) -> dict:
    """Uji bootstrap berpasangan untuk selisih macro-F1 (A - B).

    Baris yang sama diambil untuk A dan B di setiap ulangan. p_value satu
    arah = proporsi ulangan dengan selisih <= 0, yaitu peluang A tidak lebih
    baik dari B. Di E1, A = data mentah dan B = data bersih.
    """
    y_true = _as_int_array(y_true)
    a = _as_int_array(pred_a)
    b = _as_int_array(pred_b)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("Panjang y_true, pred_a, dan pred_b harus sama")
    n = len(y_true)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        deltas[i] = _macro_f1(y_true[idx], a[idx]) - _macro_f1(y_true[idx], b[idx])
    obs = _macro_f1(y_true, a) - _macro_f1(y_true, b)
    p_value = float(np.mean(deltas <= 0))
    low = float(np.percentile(deltas, 100 * alpha / 2))
    high = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "delta_observed": _r(obs),
        "delta_ci_95": format_ci(low, high),
        "delta_ci_low": _r(low),
        "delta_ci_high": _r(high),
        "p_value": _r(p_value),
        "significant_at_005": bool(p_value < alpha),
        "n_bootstraps": n_bootstraps,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Log eksperimen
# ---------------------------------------------------------------------------
def _fmt_cell(v: Any) -> Any:
    return "" if v is None else v


def log_experiment(
    results_csv_path, run_id: str, versi_data: str, kondisi: str, split: str,
    model_name: str, hyperparameter: Mapping | str | None, seed: int,
    metrics: Mapping, status_evaluasi: str = STATUS_VALIDASI,
    tanggal: str | None = None, allow_uji: bool = False,
) -> dict:
    """Tambahkan satu baris ke log eksperimen (format B2_EXPERIMENTS_BASELINE).

    - File dan foldernya dibuat bila belum ada, header ditulis sekali.
    - Jika file lama punya header berbeda, fungsi berhenti agar log tidak rusak.
    - status_evaluasi UJI ditolak kecuali allow_uji=True (dipakai src/uji_lock.py).
    """
    status = str(status_evaluasi).upper()
    if status not in (STATUS_VALIDASI, STATUS_UJI):
        raise ValueError(f"status_evaluasi harus VALIDASI atau UJI, bukan {status_evaluasi!r}")
    if status == STATUS_UJI and not allow_uji:
        raise PermissionError("Baris UJI hanya boleh ditulis oleh skrip UJI final D (allow_uji=True)")

    if isinstance(hyperparameter, Mapping):
        hp = json.dumps(dict(hyperparameter), ensure_ascii=False)
    elif hyperparameter is None:
        hp = "{}"
    else:
        hp = str(hyperparameter)

    row = {
        "run_id": run_id,
        "tanggal": tanggal or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "versi_data": versi_data,
        "kondisi": kondisi,
        "split": split,
        "model": model_name,
        "hyperparameter": hp,
        "seed": seed,
        "status_evaluasi": status,
    }
    for col in _METRIC_COLUMNS:
        row[col] = metrics.get(col)

    path = Path(results_csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != EXPERIMENT_COLUMNS:
            raise ValueError(
                f"Header {path} berbeda dari format log eksperimen. "
                "Pakai file baru atau samakan kolomnya dengan EXPERIMENT_COLUMNS.")
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXPERIMENT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: _fmt_cell(v) for k, v in row.items()})
    return row


def read_experiments(results_csv_path):
    """Baca log eksperimen sebagai DataFrame (hyperparameter tetap teks JSON)."""
    import pandas as pd
    return pd.read_csv(results_csv_path, encoding="utf-8-sig")
