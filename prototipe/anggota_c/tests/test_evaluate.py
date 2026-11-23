"""Uji src/evaluate.py (pemilik: Anggota D).

Semua angka di sini dihitung tangan dari data mainan, bukan dataset kelompok.

Jalankan dari folder prototipe/anggota_c:
    python -m pytest tests/test_evaluate.py -v
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.evaluate import (  # noqa: E402
    EXPERIMENT_COLUMNS, bootstrap_macro_f1_ci, brier_score, calculate_metrics,
    compute_ece, compute_reliability_bins, confusion_matrix_dict,
    evaluate_selective_prediction, log_experiment, paired_bootstrap_test,
    precision_at_k, selective_decisions, thresholds_from_costs,
)

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
PRED = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])          # FP 1, FN 1
PROB = np.array([0.05, 0.15, 0.30, 0.60, 0.95, 0.85, 0.40, 0.55, 0.10, 0.99])


def test_kolom_log_sama_dengan_b2():
    path = ROOT.parent.parent / "results" / "anggota_b" / "B2_EXPERIMENTS_BASELINE.csv"
    if not path.exists():
        pytest.skip("file hasil B tidak ada di salinan repo ini")
    with path.open(encoding="utf-8-sig") as fh:
        assert next(csv.reader(fh)) == EXPERIMENT_COLUMNS


def test_metrik_klasifikasi_dasar():
    m = calculate_metrics(Y, PRED)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (4, 1, 1, 4)
    assert m["accuracy"] == 0.8
    assert m["precision_hoaks"] == m["recall_hoaks"] == m["f1_hoaks"] == 0.8
    assert m["macro_f1"] == 0.8
    # tanpa y_prob, metrik peluang kosong
    for k in ("ece", "brier_score", "precision_at_50", "coverage_agen", "expected_loss"):
        assert m[k] is None


def test_label_teks_diterima():
    yt = ["VALID", "HOAKS", "HOAKS", "VALID"]
    m = calculate_metrics(yt, [0, 1, 0, 0])
    assert m["recall_hoaks"] == 0.5


def test_label_asing_ditolak():
    with pytest.raises(ValueError):
        calculate_metrics([0, 2], [0, 1])


def test_ece_dan_bin():
    # dua bin terisi: [0.0,0.1) p=0.05 y=0 ; [0.9,1.0] p=0.95,1.0 y=1,0
    y = [0, 1, 0]
    p = [0.05, 0.95, 1.0]
    bins = compute_reliability_bins(y, p, n_bins=10)
    assert len(bins) == 10
    assert bins[0]["count"] == 1 and bins[9]["count"] == 2   # p = 1,0 masuk bin terakhir
    assert bins[5]["count"] == 0 and bins[5]["mean_predicted_prob"] is None
    expected = (1 / 3) * 0.05 + (2 / 3) * abs(0.975 - 0.5)
    assert compute_ece(y, p) == pytest.approx(expected)


def test_ece_nol_untuk_kalibrasi_sempurna():
    y = [0] * 8 + [1] * 2       # bin 0.2: 2/10 HOAKS
    assert compute_ece(y, [0.2] * 10) == pytest.approx(0.0)


def test_brier():
    assert brier_score([0, 1], [0.2, 0.6]) == pytest.approx((0.04 + 0.16) / 2)


def test_precision_at_k():
    assert precision_at_k(Y, PROB, 3) == 1.0            # 0.99, 0.95, 0.85 semuanya HOAKS
    assert precision_at_k(Y, PROB, 5) == 0.8            # + 0.60 (VALID), 0.55 (HOAKS)
    assert precision_at_k(Y, PROB, 50) is None          # data < k
    m = calculate_metrics(Y, PRED, y_prob=PROB)
    assert m["precision_at_50"] is None and m["precision_at_100"] is None


def test_peluang_di_luar_rentang_ditolak():
    with pytest.raises(ValueError, match="sigmoid"):
        calculate_metrics([0, 1], [0, 1], y_prob=[-1.2, 3.4])


def test_ambang_sama_dengan_config_c():
    assert thresholds_from_costs(2, 5, 1) == pytest.approx(cfg.thresholds_from_costs(2, 5, 1))
    assert thresholds_from_costs() == pytest.approx((0.20, 0.50))
    assert thresholds_from_costs(2, 10, 1)[0] == pytest.approx(0.10)


def test_keputusan_selektif_batas_inklusif():
    d = selective_decisions([0.20, 0.2001, 0.4999, 0.50], 0.20, 0.50)
    assert d.tolist() == [0, -1, -1, 1]


def test_prediksi_selektif_hitung_tangan():
    s = evaluate_selective_prediction(Y, PROB, c_fp=2, c_fn=5, c_r=1)
    # ditolak: 0.30, 0.40 -> 2 ; diputuskan 8: 0.60 FP, 0.55 benar
    assert s["reject_count"] == 2
    assert s["coverage"] == 0.8
    assert s["fp_count_selective"] == 1 and s["fn_count_selective"] == 0
    assert s["selective_accuracy"] == pytest.approx(7 / 8, abs=1e-6)
    assert s["expected_loss_selective"] == pytest.approx((2 * 1 + 1 * 2) / 10)
    # standar p >= 0,5: FP 0.60, FN 0.40
    assert s["fn_count_standard"] == 1
    assert s["expected_loss_standard"] == pytest.approx((2 + 5) / 10)


def test_calculate_metrics_mengisi_kolom_agen():
    m = calculate_metrics(Y, PRED, y_prob=PROB)
    assert m["coverage_agen"] == 0.8
    assert m["expected_loss"] == pytest.approx(0.4)
    assert set(EXPERIMENT_COLUMNS[9:]) <= set(m)


def test_bootstrap_ci_dapat_diulang_dan_mengapit():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    pred = np.where(rng.rand(300) < 0.85, y, 1 - y)
    a = bootstrap_macro_f1_ci(y, pred, n_bootstraps=200, seed=42)
    b = bootstrap_macro_f1_ci(y, pred, n_bootstraps=200, seed=42)
    assert a == b
    f1 = calculate_metrics(y, pred, n_bootstraps=200)["macro_f1"]
    assert a[0] <= f1 <= a[1]
    assert calculate_metrics(y, pred, n_bootstraps=200)["macro_f1_ci_95"].startswith("[")


def test_paired_bootstrap():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 400)
    good = np.where(rng.rand(400) < 0.95, y, 1 - y)
    bad = np.where(rng.rand(400) < 0.70, y, 1 - y)
    r = paired_bootstrap_test(y, good, bad, n_bootstraps=300, seed=42)
    assert r["delta_observed"] > 0 and r["p_value"] < 0.05 and r["significant_at_005"]
    same = paired_bootstrap_test(y, good, good, n_bootstraps=100)
    assert same["p_value"] == 1.0 and not same["significant_at_005"]   # selisih 0 tidak signifikan


def test_confusion_matrix_dict():
    cm = confusion_matrix_dict(Y, PRED)
    assert cm == {"asli_VALID_pred_VALID": 4, "asli_VALID_pred_HOAKS": 1,
                  "asli_HOAKS_pred_VALID": 1, "asli_HOAKS_pred_HOAKS": 4}


def _log(path, **kw):
    m = calculate_metrics(Y, PRED, y_prob=PROB, n_bootstraps=50)
    args = dict(results_csv_path=path, run_id="r1", versi_data="V3.1_A2_PROVISIONAL",
                kondisi="uji_unit", split="random_split", model_name="LinearSVM",
                hyperparameter={"C": 1.0}, seed=42, metrics=m)
    args.update(kw)
    return log_experiment(**args)


def test_log_experiment_menulis_header_sekali(tmp_path):
    path = tmp_path / "sub" / "experiments.csv"
    _log(path)
    _log(path, run_id="r2")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == EXPERIMENT_COLUMNS
    assert len(rows) == 3
    rec = dict(zip(rows[0], rows[1]))
    assert json.loads(rec["hyperparameter"]) == {"C": 1.0}
    assert rec["status_evaluasi"] == "VALIDASI"
    assert rec["precision_at_50"] == ""        # None ditulis kosong


def test_log_experiment_menolak_uji_tanpa_izin(tmp_path):
    path = tmp_path / "experiments.csv"
    with pytest.raises(PermissionError):
        _log(path, status_evaluasi="UJI")
    assert not path.exists()
    _log(path, status_evaluasi="UJI", allow_uji=True)
    assert path.exists()


def test_log_experiment_menolak_header_berbeda(tmp_path):
    path = tmp_path / "experiments.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Header"):
        _log(path)
