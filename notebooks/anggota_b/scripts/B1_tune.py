"""scripts/tune.py
Skrip Tuning Hyperparameter Model Baseline (Anggota B).
Mengeksplorasi ComplementNB (alpha in {0.1, 0.5, 1.0}) dan
LinearSVM (C in {0.1, 1.0, 10.0}) pada partisi VALIDASI.

Penggunaan:
    python scripts/tune.py --data path/to/dataset.csv --split-col random_split
"""
import argparse
import json
from pathlib import Path
import sys
import time
import zipfile
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Tambahkan direktori root prototipe ke sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.evaluate import calculate_metrics, log_experiment


def get_pipeline(model_type, hyperparams):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    if model_type == 'ComplementNB':
        clf = ComplementNB(alpha=hyperparams['alpha'])
    elif model_type == 'LinearSVM':
        clf = LinearSVC(
            C=hyperparams['C'],
            class_weight='balanced',
            random_state=42,
            max_iter=3000
        )
    else:
        raise ValueError(f"Tipe model tidak dikenal: {model_type}")
    return Pipeline([('tfidf', tfidf), ('classifier', clf)])


def load_dataset(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        # Coba fallback ke file zip jika file CSV langsung belum diekstrak
        zip_path = BASE_DIR.parent.parent / 'Dataset' / 'HASIL' / 'HASIL_EDA_PREPROCESSING.zip'
        if zip_path.exists():
            print(f"[INFO] Membaca dataset langsung dari arsip ZIP: {zip_path.name}")
            with zipfile.ZipFile(zip_path) as z:
                with z.open('DATASET_V3_KANDIDAT_OPERASIONAL.csv') as f:
                    return pd.read_csv(f)
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path} maupun arsip zip.")
    
    print(f"[INFO] Membaca dataset dari: {data_path}")
    return pd.read_csv(data_path)


def main():
    parser = argparse.ArgumentParser(description="Tuning Hyperparameter Baseline (Anggota B)")
    parser.add_argument('--data', default=str(BASE_DIR / 'data' / 'DATASET_V3_KANDIDAT_OPERASIONAL.csv'),
                        help="Path ke berkas dataset CSV")
    parser.add_argument('--split-col', default='random_split',
                        help="Nama kolom split (default: random_split)")
    parser.add_argument('--output', default=str(BASE_DIR / 'results' / 'tuning_results.csv'),
                        help="Path berkas output perbandingan tuning")
    parser.add_argument('--experiments-log', default=str(BASE_DIR / 'results' / 'experiments.csv'),
                        help="Path berkas log experiments.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models_dir = BASE_DIR / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    
    # Filter hanya data LATIH dan VALIDASI
    df_active = df.loc[df[args.split_col].isin(['LATIH', 'VALIDASI'])].copy()
    train_df = df_active.loc[df_active[args.split_col] == 'LATIH']
    valid_df = df_active.loc[df_active[args.split_col] == 'VALIDASI']

    print(f"\n[INFO] Split: {args.split_col} | Latih: {len(train_df):,} | Validasi: {len(valid_df):,}")

    grid = [
        # ComplementNB Grid
        ('ComplementNB', {'alpha': 0.1}),
        ('ComplementNB', {'alpha': 0.5}),
        ('ComplementNB', {'alpha': 1.0}),
        # LinearSVM Grid
        ('LinearSVM', {'C': 0.1}),
        ('LinearSVM', {'C': 1.0}),
        ('LinearSVM', {'C': 10.0}),
    ]

    feature_cols = ['text_clean', 'text_raw']
    results = []
    best_models = {}

    for feature_col in feature_cols:
        print(f"\n{'='*60}\nMenjalankan Tuning pada fitur: {feature_col}\n{'='*60}")
        X_train = train_df[feature_col].fillna('').astype(str)
        y_train = train_df['label']
        X_valid = valid_df[feature_col].fillna('').astype(str)
        y_valid = valid_df['label']

        for model_type, params in grid:
            run_id = f"tune_{model_type}_{list(params.keys())[0]}_{list(params.values())[0]}_{feature_col}"
            print(f"--> Melatih {model_type} ({params}) pada {feature_col}...", end=' ', flush=True)
            tic = time.time()
            
            pipe = get_pipeline(model_type, params)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_valid)
            
            # Decision function / probabilitas
            y_prob = None
            if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
                y_prob = pipe.predict_proba(X_valid)[:, 1]
            elif hasattr(pipe.named_steps['classifier'], 'decision_function'):
                # Normalisasi min-max sederhana untuk kalkulasi precision@k indikatif sebelum kalibrasi
                dfunc = pipe.decision_function(X_valid)
                y_prob = 1.0 / (1.0 + np.exp(-dfunc))

            elapsed = round(time.time() - tic, 2)
            metrics = calculate_metrics(y_valid, pred, y_prob=y_prob)
            print(f"Selesai ({elapsed}s) | Macro-F1 Validasi: {metrics['macro_f1']:.4f}")

            res_row = {
                'run_id': run_id,
                'model': model_type,
                'fit_text': feature_col,
                'param_name': list(params.keys())[0],
                'param_value': list(params.values())[0],
                'durasi_detik': elapsed,
                'macro_f1_validasi': metrics['macro_f1'],
                'macro_f1_ci_95': metrics['macro_f1_ci_95'],
                'accuracy_validasi': metrics['accuracy'],
                'f1_hoaks': metrics['f1_hoaks'],
                'precision_hoaks': metrics['precision_hoaks'],
                'recall_hoaks': metrics['recall_hoaks'],
                'f1_valid': metrics['f1_valid'],
                'precision_valid': metrics['precision_valid'],
                'recall_valid': metrics['recall_valid'],
                'ece_mentah': metrics['ece']
            }
            results.append(res_row)

            # Catat ke log resmi experiments.csv (selalu berlabel VALIDASI)
            log_experiment(
                results_csv_path=args.experiments_log,
                run_id=run_id,
                versi_data='V3.1 (atau V3 pre-V3.1)',
                kondisi='tuning_hyperparameter',
                split=args.split_col,
                model_name=model_type,
                hyperparameter=params,
                seed=42,
                metrics=metrics,
                status_evaluasi='VALIDASI'
            )

            # Lacak konfigurasi terbaik per model pada text_clean
            if feature_col == 'text_clean':
                if model_type not in best_models or metrics['macro_f1'] > best_models[model_type]['macro_f1']:
                    best_models[model_type] = {
                        'params': params,
                        'macro_f1': metrics['macro_f1'],
                        'pipeline': pipe
                    }

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"\n[SUKSES] Seluruh hasil tuning tersimpan di: {output_path}")

    print("\n" + "="*70)
    print("RINGKASAN KONFIGURASI TERBAIK (VALIDASI - TEXT_CLEAN):")
    print("="*70)
    for m_type, b_info in best_models.items():
        print(f"Model: {m_type:15} | Params: {str(b_info['params']):20} | Macro-F1: {b_info['macro_f1']:.6f}")
        # Simpan model terbaik ke models/
        save_path = models_dir / f"{m_type}.joblib"
        joblib.dump(b_info['pipeline'], save_path, compress=3)
        print(f"  -> Model terbaik disimpan ke: {save_path.name}")
    print("="*70)


if __name__ == '__main__':
    import numpy as np
    main()
