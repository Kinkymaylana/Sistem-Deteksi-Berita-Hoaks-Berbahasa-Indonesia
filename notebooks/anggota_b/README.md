# Notebook & Skrip Anggota B

Folder ini berisi skrip dan notebook pengerjaan **Anggota B** untuk proyek Sistem Deteksi Berita Hoaks Berbahasa Indonesia (pemodelan baseline, eksperimen anti-shortcut, kalibrasi probabilitas, dan fine-tuning IndoBERT).

## Cakupan Pekerjaan
- **B1**: Tuning hyperparameter baseline (`ComplementNB` & `LinearSVM`) pada partisi `VALIDASI` (`scripts/B1_tune.py`)
- **B2**: Eksekusi eksperimen baseline (`E1` shortcut mentah vs bersih, `E2` generalisasi temporal 2021, `Title-Only`, dan `E4` kontrol panjang teks pada korpus V3) (`scripts/B2_run_experiments.py`)
- **B3**: Kalibrasi probabilitas *Platt Scaling* pada `LinearSVM` (`LinearSVM_calibrated`), evaluasi Chow's Rejection Rule, dan pembuatan Model Card (`scripts/B3_calibrate.py`)
- **B4**: Finalisasi statistik panjang token IndoBERT (`indobenchmark/indobert-base-p1`, threshold 256 token) (`B4_TOKEN_INDOBERT.ipynb`)
- **B5**: Fine-tuning multi-seed IndoBERT (`seeds=[42, 123, 2026]`, LLRD, class weights, `head_tail` 256 token) & *Temperature Scaling* pada `random_split` korpus V3 (`B5_INDOBERT_FINETUNE_COLAB.ipynb`)

## Catatan Penyimpanan & Versi Data
- Dataset berukuran besar, cache tokenisasi (`.pt`), bobot model (`.joblib` / checkpoint `.pt`), dan artefak Drive Colab disimpan di Google Drive kelompok (`/content/drive/MyDrive/PSP_IndoBERT_Checkpoints_S2/`).
- Eksperimen **B2** dan **B5** pada folder ini dijalankan pada korpus **V3** (`DATASET_V3_KANDIDAT_OPERASIONAL.csv`, kolom `text_clean`). Evaluasi lanjutan pada korpus **V3.1** (`text_clean_v31`), `temporal_main`/`temporal_split`, dan `E4` untuk IndoBERT akan menjadi tahap berikutnya.
- Seluruh evaluasi hanya menggunakan partisi `VALIDASI`; partisi `UJI` tidak diakses.
