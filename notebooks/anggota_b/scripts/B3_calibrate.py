"""scripts/calibrate.py
Skrip Kalibrasi LinearSVM menggunakan Platt Scaling (Sigmoid) — Standar S2 (Anggota B).
Menghasilkan:
1. models/LinearSVM_calibrated.joblib
2. models/model_card.json (metadata versi data, hyperparameter, ECE, Brier Score, Chow's Rule, Reliability Bins)
3. results/reliability_bins_svm.csv (data kurva kalibrasi sebelum & sesudah Platt Scaling)
4. results/selective_risk_coverage_svm.csv (analisis Expected Loss untuk C_FN in {3, 5, 10})

Penggunaan:
    python scripts/calibrate.py --data path/to/dataset.csv
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.evaluate import (
    calculate_metrics,
    compute_reliability_bins,
    evaluate_selective_prediction,
    log_experiment
)


def load_dataset(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        zip_path = BASE_DIR.parent.parent / 'Dataset' / 'HASIL' / 'HASIL_EDA_PREPROCESSING.zip'
        if zip_path.exists():
            print(f"[INFO] Membaca dataset dari arsip ZIP: {zip_path.name}")
            with zipfile.ZipFile(zip_path) as z:
                with z.open('DATASET_V3_KANDIDAT_OPERASIONAL.csv') as f:
                    return pd.read_csv(f)
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path} maupun arsip zip.")
    return pd.read_csv(data_path)


def main():
    parser = argparse.ArgumentParser(description="Kalibrasi Probabilitas SVM & Evaluasi Agen S2 (Anggota B)")
    parser.add_argument('--data', default=str(BASE_DIR / 'data' / 'DATASET_V3_KANDIDAT_OPERASIONAL.csv'))
    parser.add_argument('--model-in', default=str(BASE_DIR / 'models' / 'LinearSVM.joblib'))
    parser.add_argument('--model-out', default=str(BASE_DIR / 'models' / 'LinearSVM_calibrated.joblib'))
    parser.add_argument('--model-card', default=str(BASE_DIR / 'models' / 'model_card.json'))
    parser.add_argument('--experiments-log', default=str(BASE_DIR / 'results' / 'experiments.csv'))
    args = parser.parse_args()

    models_dir = BASE_DIR / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = BASE_DIR / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    df_active = df.loc[df['random_split'].isin(['LATIH', 'VALIDASI'])].copy()
    train_df = df_active.loc[df_active['random_split'] == 'LATIH']
    valid_df = df_active.loc[df_active['random_split'] == 'VALIDASI']

    X_train = train_df['text_clean'].fillna('').astype(str)
    y_train = train_df['label']
    X_valid = valid_df['text_clean'].fillna('').astype(str)
    y_valid = valid_df['label']

    model_in_path = Path(args.model_in)
    if model_in_path.exists():
        print(f"[INFO] Memuat model LinearSVM dasar dari: {model_in_path.name}")
        base_pipeline = joblib.load(model_in_path)
    else:
        print("[INFO] Model dasar tidak ditemukan, melatih pipeline LinearSVM baru...")
        base_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)),
            ('classifier', LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=3000))
        ])
        base_pipeline.fit(X_train, y_train)
        joblib.dump(base_pipeline, model_in_path, compress=3)

    # 1. Evaluasi Sebelum Kalibrasi (uncalibrated sigmoid margin)
    print("\n[1/4] Mengevaluasi skor mentah sebelum kalibrasi...")
    dfunc_valid = base_pipeline.decision_function(X_valid)
    raw_prob = 1.0 / (1.0 + np.exp(-dfunc_valid))
    raw_pred = (raw_prob >= 0.5).astype(int)
    metrics_before = calculate_metrics(y_valid, raw_pred, y_prob=raw_prob)
    ece_before = metrics_before['ece']
    brier_before = metrics_before['brier_score']
    print(f"  -> ECE Sebelum Kalibrasi   : {ece_before:.6f}")
    print(f"  -> Brier Score Sebelum     : {brier_before:.6f}")
    print(f"  -> Macro-F1                : {metrics_before['macro_f1']:.4f}")

    # 2. Kalibrasi Platt Scaling (Sigmoid) pada data VALIDASI
    print("\n[2/4] Menerapkan CalibratedClassifierCV(method='sigmoid', cv='prefit') pada VALIDASI...")
    tic = time.time()
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv="prefit"
    )
    calibrated_model.fit(X_valid, y_valid)
    calib_duration = round(time.time() - tic, 2)

    calib_prob = calibrated_model.predict_proba(X_valid)[:, 1]
    calib_pred = calibrated_model.predict(X_valid)
    metrics_after = calculate_metrics(y_valid, calib_pred, y_prob=calib_prob)
    ece_after = metrics_after['ece']
    brier_after = metrics_after['brier_score']
    ece_reduction_pct = round(((ece_before - ece_after) / ece_before) * 100, 2) if ece_before > 0 else 0.0
    brier_reduction_pct = round(((brier_before - brier_after) / brier_before) * 100, 2) if brier_before > 0 else 0.0

    print(f"  -> ECE Sesudah Kalibrasi   : {ece_after:.6f} (Penurunan ECE: {ece_reduction_pct}%)")
    print(f"  -> Brier Score Sesudah     : {brier_after:.6f} (Penurunan Brier: {brier_reduction_pct}%)")
    print(f"  -> Macro-F1 Pasca Kalibrasi: {metrics_after['macro_f1']:.4f} (95% CI: {metrics_after['macro_f1_ci_95']})")

    # 3. Analisis Reliability Diagram & Chow's Rejection Rule (Minimum Expected Loss)
    print("\n[3/4] Menghitung statistik Reliability Diagram & Chow's Rejection Rule (C_FN in {3, 5, 10})...")
    bins_before = compute_reliability_bins(y_valid, raw_prob, n_bins=10)
    bins_after = compute_reliability_bins(y_valid, calib_prob, n_bins=10)
    rel_rows = []
    for b_pre, b_post in zip(bins_before, bins_after):
        rel_rows.append({
            'bin_index': b_pre['bin_index'],
            'bin_range': f"[{b_pre['bin_low']:.1f}, {b_pre['bin_high']:.1f})",
            'count_pre': b_pre['count'],
            'mean_prob_pre': b_pre['mean_predicted_prob'],
            'emp_pos_pre': b_pre['empirical_positive_rate'],
            'count_post': b_post['count'],
            'mean_prob_post': b_post['mean_predicted_prob'],
            'emp_pos_post': b_post['empirical_positive_rate']
        })
    rel_path = results_dir / 'reliability_bins_svm.csv'
    pd.DataFrame(rel_rows).to_csv(rel_path, index=False)

    # Evaluasi Risk-Coverage / Expected Loss untuk C_FN in {3, 5, 10}
    risk_rows = []
    for c_fn in [3.0, 5.0, 10.0]:
        # Ambang Bayes-optimal teoritis dengan C_FP=2, C_R=1:
        # tau_low = C_R / C_FN, tau_high = 1 - (C_R / C_FP) = 0.50
        tau_low = round(1.0 / c_fn, 4)
        tau_high = 0.50
        sel_res = evaluate_selective_prediction(
            y_valid, calib_prob, tau_low=tau_low, tau_high=tau_high,
            c_fp=2.0, c_fn=c_fn, c_r=1.0
        )
        sel_res['c_fn'] = c_fn
        sel_res['c_fp'] = 2.0
        sel_res['c_r'] = 1.0
        sel_res['tau_low'] = tau_low
        sel_res['tau_high'] = tau_high
        risk_rows.append(sel_res)
        print(f"  -> [C_FN={int(c_fn):2d} | tau=({tau_low:.2f}, {tau_high:.2f})] Coverage: {sel_res['coverage']*100:.2f}% | "
              f"Selective F1: {sel_res['selective_macro_f1']:.4f} | FN turun: {sel_res['fn_count_standard']} -> {sel_res['fn_count_selective']} | "
              f"Expected Loss: {sel_res['expected_loss_standard']:.4f} -> {sel_res['expected_loss_selective']:.4f}")

    risk_path = results_dir / 'selective_risk_coverage_svm.csv'
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False)

    # 4. Simpan Artefak Model & Model Card S2
    print(f"\n[4/4] Menyimpan model terkalibrasi ke: {args.model_out}")
    joblib.dump(calibrated_model, args.model_out, compress=3)

    model_card_data = {
        "model_name": "LinearSVM_calibrated",
        "description": "Model Linear Support Vector Classifier dengan TF-IDF (1,2) dikalibrasi menggunakan Platt Scaling (Sigmoid) & dievaluasi dengan Chow's Rejection Rule",
        "data_version": "Korpus V3.1 (atau V3 pre-V3.1)",
        "n_samples_train": len(X_train),
        "n_samples_valid": len(X_valid),
        "hyperparameters": {
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 2,
            "tfidf_sublinear_tf": True,
            "tfidf_max_features": 200000,
            "svm_C": 1.0,
            "svm_class_weight": "balanced",
            "svm_max_iter": 3000,
            "svm_random_state": 42,
            "calibration_method": "sigmoid (Platt Scaling)",
            "calibration_cv": "prefit"
        },
        "metrics_validasi": {
            "macro_f1": metrics_after['macro_f1'],
            "macro_f1_ci_95": metrics_after['macro_f1_ci_95'],
            "accuracy": metrics_after['accuracy'],
            "precision_hoaks": metrics_after['precision_hoaks'],
            "recall_hoaks": metrics_after['recall_hoaks'],
            "f1_hoaks": metrics_after['f1_hoaks'],
            "precision_at_50": metrics_after['precision_at_50'],
            "precision_at_100": metrics_after['precision_at_100'],
            "ece_before_calibration": ece_before,
            "ece_after_calibration": ece_after,
            "ece_improvement_percent": ece_reduction_pct,
            "brier_before_calibration": brier_before,
            "brier_after_calibration": brier_after,
            "brier_improvement_percent": brier_reduction_pct,
            "selective_prediction_default_cfn5": risk_rows[1]
        },
        "environment": {
            "scikit_learn_version": sklearn.__version__,
            "python_version": platform.python_version()
        },
        "serah_terima": {
            "pemilik": "Anggota B",
            "penerima": ["Anggota C (Integrasi Agen dan Aplikasi Streamlit)", "Anggota D (Evaluasi Metrik)"],
            "tenggat": "15 November 2026",
            "status": "SELESAI_TERUJI_STANDAR_S2"
        }
    }

    with open(args.model_card, 'w', encoding='utf-8') as f:
        json.dump(model_card_data, f, indent=2, ensure_ascii=False)
    print(f"Model card berhasil disimpan di: {args.model_card}")

    log_experiment(
        results_csv_path=args.experiments_log,
        run_id="calibration_LinearSVM_platt_s2",
        versi_data="V3.1 (atau V3 pre-V3.1)",
        kondisi="platt_scaling_sigmoid_s2",
        split="random_split",
        model_name="LinearSVM_calibrated",
        hyperparameter={"method": "sigmoid", "cv": "prefit", "base_C": 1.0},
        seed=42,
        metrics=metrics_after,
        status_evaluasi="VALIDASI"
    )

    print("\n" + "="*70)
    print("KALIBRASI & EVALUASI SELECTIVE PREDICTION S2 SELESAI:")
    print(f"1. Model terkalibrasi : {Path(args.model_out).name}")
    print(f"2. Model Card JSON    : {Path(args.model_card).name}")
    print(f"3. Reliability Bins   : {rel_path.name}")
    print(f"4. Risk-Coverage CSV  : {risk_path.name}")
    print("="*70)


if __name__ == '__main__':
    main()
