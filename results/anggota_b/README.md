# Hasil Pengujian Anggota B

Folder ini menyimpan hasil pemodelan baseline, kalibrasi probabilitas, statistik token, dan ringkasan eksperimen IndoBERT oleh **Anggota B**.

## B2 — Eksperimen Baseline (Korpus V3, `VALIDASI`)
- `B2_BASELINE_METADATA.json`
- `B2_EXPERIMENTS_BASELINE.csv`
- `B2_E1_SHORTCUT_ANALYSIS.csv`
- `B2_E2_TEMPORAL_ANALYSIS.csv`
- `B2_E4_LENGTH_CONTROL_ANALYSIS.csv`
- `B2_TITLE_ONLY_ANALYSIS.csv`

## B3 — Kalibrasi Platt Scaling & Model Card (`LinearSVM_calibrated`)
- `B3_MODEL_CARD_BASELINE.json`
- `B3_RELIABILITY_BINS_SVM.csv`
- `B3_SELECTIVE_RISK_COVERAGE_SVM.csv`

## B4 — Statistik Panjang Token IndoBERT
- `TOKEN_INDOBERT.json`

## B5 — Fine-Tuning & Kalibrasi Suhu IndoBERT (`random_split`, 3-Seed)
- `B5_INDOBERT_RANDOM_METADATA.json`
- `experiments_indobert_random_split.csv`
- `experiments_indobert_random_split_text_clean_head_tail_ls10_mask10.csv`
- `reliability_diagram_indobert_random_split.png`
*(Checkpoint `.pt` dan dataset besar disimpan di luar Git pada Google Drive `/content/drive/MyDrive/PSP_IndoBERT_Checkpoints_S2/`.)*
