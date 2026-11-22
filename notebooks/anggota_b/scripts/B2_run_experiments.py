"""scripts/run_experiments.py
Skrip Eksekusi Eksperimen Resmi Anggota B (Standar S2 & Anti-Shortcut):
1. Baseline Evaluasi (Random Split, Clean Text) + Paired Bootstrap Significance Test (SVM vs NB)
2. Eksperimen E1: Analisis Shortcut (Data Mentah vs Bersih) -> Delta F1 + p-value signifikansi
3. Eksperimen E2: Generalisasi Temporal (temporal_main vs temporal_split) -> Validasi 2021
4. Eksperimen E-Title: Evaluasi Berbasis Judul Saja (Title-Only) -> Bebas Bias Panjang & Dateline
5. Eksperimen E4: Kontrol Panjang Teks (text_clean_e4) -> Menyamakan Distribusi Panjang Valid & Hoaks

Penggunaan:
    python scripts/run_experiments.py --data path/to/dataset.csv
"""
import argparse
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.evaluate import calculate_metrics, log_experiment, paired_bootstrap_test


def build_pipeline(model_type, hyperparams):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    if model_type == 'ComplementNB':
        alpha = hyperparams.get('alpha', 1.0)
        clf = ComplementNB(alpha=alpha)
    elif model_type == 'LinearSVM':
        C = hyperparams.get('C', 1.0)
        clf = LinearSVC(C=C, class_weight='balanced', random_state=42, max_iter=3000)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return Pipeline([('tfidf', tfidf), ('classifier', clf)])


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


def create_length_controlled_column(df, text_col='text_clean', seed=42):
    """E4: Menyamakan distribusi panjang berita VALID dengan distribusi berita HOAKS."""
    rng = np.random.RandomState(seed)
    hoax_texts = df.loc[df['label'] == 1, text_col].fillna('').astype(str)
    hoax_lengths = [max(1, len(t.split())) for t in hoax_texts]

    controlled_texts = []
    for _, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ''
        if row['label'] == 1:
            controlled_texts.append(text)
        else:
            target_len = rng.choice(hoax_lengths)
            words = text.split()
            controlled_texts.append(' '.join(words[:target_len]) if len(words) > target_len else text)
    return controlled_texts


def run_single_eval(df, split_col, feature_col, model_type, hyperparams, run_id, kondisi, exp_log_path, versi_data='V3.1 (atau V3 pre-V3.1)'):
    df_active = df.loc[df[split_col].isin(['LATIH', 'VALIDASI'])].copy()
    train_df = df_active.loc[df_active[split_col] == 'LATIH']
    valid_df = df_active.loc[df_active[split_col] == 'VALIDASI']

    X_train = train_df[feature_col].fillna('').astype(str)
    y_train = train_df['label'].values
    X_valid = valid_df[feature_col].fillna('').astype(str)
    y_valid = valid_df['label'].values

    tic = time.time()
    pipe = build_pipeline(model_type, hyperparams)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_valid)

    y_prob = None
    if hasattr(pipe.named_steps['classifier'], 'predict_proba'):
        y_prob = pipe.predict_proba(X_valid)[:, 1]
    elif hasattr(pipe.named_steps['classifier'], 'decision_function'):
        dfunc = pipe.decision_function(X_valid)
        y_prob = 1.0 / (1.0 + np.exp(-dfunc))

    elapsed = round(time.time() - tic, 2)
    metrics = calculate_metrics(y_valid, pred, y_prob=y_prob)

    log_experiment(
        results_csv_path=exp_log_path,
        run_id=run_id,
        versi_data=versi_data,
        kondisi=kondisi,
        split=split_col,
        model_name=model_type,
        hyperparameter=hyperparams,
        seed=42,
        metrics=metrics,
        status_evaluasi='VALIDASI'
    )

    return {
        'run_id': run_id,
        'kondisi': kondisi,
        'split': split_col,
        'model': model_type,
        'fit_text': feature_col,
        'hyperparams': hyperparams,
        'durasi': elapsed,
        'macro_f1': metrics['macro_f1'],
        'macro_f1_ci': metrics['macro_f1_ci_95'],
        'accuracy': metrics['accuracy'],
        'f1_hoaks': metrics['f1_hoaks'],
        'precision_hoaks': metrics['precision_hoaks'],
        'recall_hoaks': metrics['recall_hoaks'],
        'ece': metrics['ece'],
        'brier_score': metrics['brier_score'],
        'y_true': y_valid,
        'y_pred': pred
    }


def main():
    parser = argparse.ArgumentParser(description="Jalankan Eksperimen Resmi Anggota B (Standar S2)")
    parser.add_argument('--data', default=str(BASE_DIR / 'data' / 'DATASET_V3_KANDIDAT_OPERASIONAL.csv'))
    parser.add_argument('--text-col', default='text_clean')
    parser.add_argument('--data-version', default='V3.1 (atau V3 pre-V3.1)')
    parser.add_argument('--experiments-log', default=str(BASE_DIR / 'results' / 'experiments.csv'))
    parser.add_argument('--e1-out', default=str(BASE_DIR / 'results' / 'e1_shortcut_analysis.csv'))
    parser.add_argument('--e2-out', default=str(BASE_DIR / 'results' / 'e2_temporal_analysis.csv'))
    parser.add_argument('--title-out', default=str(BASE_DIR / 'results' / 'title_only_analysis.csv'))
    parser.add_argument('--e4-out', default=str(BASE_DIR / 'results' / 'e4_length_control_analysis.csv'))
    args = parser.parse_args()

    df = load_dataset(args.data)
    models = [
        ('ComplementNB', {'alpha': 1.0}),
        ('LinearSVM', {'C': 1.0})
    ]

    print("\n" + "="*75)
    print("BAGIAN 1: EKSPERIMEN E1 — ANALISIS SHORTCUT & UJI SIGNIFIKANSI (PAIRED BOOTSTRAP)")
    print("="*75)
    e1_records = []
    clean_preds_by_model = {}
    y_valid_ref = None

    for model_name, hparams in models:
        raw_res = run_single_eval(
            df=df, split_col='random_split', feature_col='text_raw',
            model_type=model_name, hyperparams=hparams,
            run_id=f"E1_{model_name}_raw", kondisi="E1_data_mentah",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        clean_res = run_single_eval(
            df=df, split_col='random_split', feature_col=args.text_col,
            model_type=model_name, hyperparams=hparams,
            run_id=f"E1_{model_name}_clean", kondisi="E1_data_bersih",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        clean_preds_by_model[model_name] = clean_res['y_pred']
        y_valid_ref = clean_res['y_true']

        sig_test = paired_bootstrap_test(
            clean_res['y_true'], raw_res['y_pred'], clean_res['y_pred'], n_bootstraps=1000, seed=42
        )
        delta_f1 = round(raw_res['macro_f1'] - clean_res['macro_f1'], 6)
        print(f"[{model_name}] Macro-F1 Mentah: {raw_res['macro_f1']:.4f} | Bersih: {clean_res['macro_f1']:.4f} | "
              f"Delta F1: {delta_f1:+.4f} (p-value={sig_test['p_value']:.4f}, sig@0.05={sig_test['significant_at_005']})")

        e1_records.append({
            'model': model_name,
            'macro_f1_mentah': raw_res['macro_f1'],
            'macro_f1_bersih': clean_res['macro_f1'],
            'macro_f1_bersih_ci_95': clean_res['macro_f1_ci'],
            'delta_f1': delta_f1,
            'p_value_raw_vs_clean': sig_test['p_value'],
            'signifikan_alpha_005': sig_test['significant_at_005'],
            'f1_hoaks_mentah': raw_res['f1_hoaks'],
            'f1_hoaks_bersih': clean_res['f1_hoaks']
        })

    e1_df = pd.DataFrame(e1_records)
    e1_df.to_csv(args.e1_out, index=False)
    print(f"Hasil E1 disimpan di: {args.e1_out}")

    print("\n" + "="*75)
    print("BAGIAN 2: EKSPERIMEN E2 — GENERALISASI TEMPORAL (VALIDASI 2021)")
    print("="*75)
    e2_records = []
    for model_name, hparams in models:
        main_res = run_single_eval(
            df=df, split_col='temporal_main', feature_col=args.text_col,
            model_type=model_name, hyperparams=hparams,
            run_id=f"E2_{model_name}_temporal_main", kondisi="E2_temporal_main_2021",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        split_res = run_single_eval(
            df=df, split_col='temporal_split', feature_col=args.text_col,
            model_type=model_name, hyperparams=hparams,
            run_id=f"E2_{model_name}_temporal_split", kondisi="E2_temporal_split_2021",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        print(f"[{model_name}] Validasi 2021 temporal_main Macro-F1: {main_res['macro_f1']:.4f} | temporal_split: {split_res['macro_f1']:.4f}")

        e2_records.append({
            'model': model_name,
            'skema_split': 'temporal_main (Kompas & TBH)',
            'macro_f1_val_2021': main_res['macro_f1'],
            'macro_f1_ci_95': main_res['macro_f1_ci'],
            'accuracy': main_res['accuracy'],
            'f1_hoaks': main_res['f1_hoaks']
        })
        e2_records.append({
            'model': model_name,
            'skema_split': 'temporal_split (Semua Sumber)',
            'macro_f1_val_2021': split_res['macro_f1'],
            'macro_f1_ci_95': split_res['macro_f1_ci'],
            'accuracy': split_res['accuracy'],
            'f1_hoaks': split_res['f1_hoaks']
        })

    e2_df = pd.DataFrame(e2_records)
    e2_df.to_csv(args.e2_out, index=False)
    print(f"Hasil E2 disimpan di: {args.e2_out}")

    print("\n" + "="*75)
    print("BAGIAN 3: EKSPERIMEN E-TITLE — EVALUASI BERBASIS JUDUL SAJA (TITLE-ONLY)")
    print("="*75)
    title_records = []
    for model_name, hparams in models:
        t_res = run_single_eval(
            df=df, split_col='random_split', feature_col='title',
            model_type=model_name, hyperparams=hparams,
            run_id=f"E_Title_{model_name}", kondisi="Title_Only_Anti_Shortcut",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        print(f"[{model_name} (Title-Only)] Macro-F1: {t_res['macro_f1']:.4f} | F1-Hoaks: {t_res['f1_hoaks']:.4f} | Acc: {t_res['accuracy']:.4f}")
        title_records.append({
            'model': model_name,
            'feature': 'title (Judul Saja)',
            'macro_f1': t_res['macro_f1'],
            'macro_f1_ci_95': t_res['macro_f1_ci'],
            'accuracy': t_res['accuracy'],
            'f1_hoaks': t_res['f1_hoaks'],
            'precision_hoaks': t_res['precision_hoaks'],
            'recall_hoaks': t_res['recall_hoaks']
        })
    title_df = pd.DataFrame(title_records)
    title_df.to_csv(args.title_out, index=False)
    print(f"Hasil Title-Only disimpan di: {args.title_out}")

    print("\n" + "="*75)
    print("BAGIAN 4: EKSPERIMEN E4 — KONTROL PANJANG TEKS (LENGTH-MATCHED SUBSAMPLING)")
    print("="*75)
    print("Membuat kolom text_clean_e4 (menyamakan distribusi panjang berita VALID dengan HOAKS)...")
    df['text_clean_e4'] = create_length_controlled_column(df, text_col=args.text_col, seed=42)
    e4_records = []
    for model_name, hparams in models:
        e4_res = run_single_eval(
            df=df, split_col='random_split', feature_col='text_clean_e4',
            model_type=model_name, hyperparams=hparams,
            run_id=f"E4_{model_name}_length_control", kondisi="E4_Length_Controlled",
            exp_log_path=args.experiments_log, versi_data=args.data_version
        )
        print(f"[{model_name} (E4 Kontrol Panjang)] Macro-F1: {e4_res['macro_f1']:.4f} | F1-Hoaks: {e4_res['f1_hoaks']:.4f} | Acc: {e4_res['accuracy']:.4f}")
        e4_records.append({
            'model': model_name,
            'feature': 'text_clean_e4 (Kontrol Panjang)',
            'macro_f1': e4_res['macro_f1'],
            'macro_f1_ci_95': e4_res['macro_f1_ci'],
            'accuracy': e4_res['accuracy'],
            'f1_hoaks': e4_res['f1_hoaks'],
            'precision_hoaks': e4_res['precision_hoaks'],
            'recall_hoaks': e4_res['recall_hoaks']
        })
    e4_df = pd.DataFrame(e4_records)
    e4_df.to_csv(args.e4_out, index=False)
    print(f"Hasil E4 disimpan di: {args.e4_out}")

    print("\n[SELESAI] Seluruh rangkaian eksperimen (E1, E2, Title-Only, E4) tuntas dan tercatat!")


if __name__ == '__main__':
    main()
