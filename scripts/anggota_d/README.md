# Skrip Anggota D

## D1. Modul evaluasi `src/evaluate.py`

**Lokasi modul:** `prototipe/anggota_c/src/evaluate.py`.

Modul ini berada di folder prototipe, bukan di folder `anggota_d`, karena ada dua hal yang sudah menunjuk ke lokasi itu:
- Skrip B mengimpornya dengan `from src.evaluate import ...` dari "root prototipe".
- README C (bagian 6) sudah mencadangkan `src/evaluate.py` untuk D.

**Tes:** `prototipe/anggota_c/tests/test_evaluate.py` berisi 19 tes dengan angka yang dihitung tangan.

### Fungsi

| Fungsi | Dipakai oleh | Isi |
|---|---|---|
| `calculate_metrics(y_true, y_pred, y_prob=None)` | B1, B2, B3, D | P, R, dan F1 per kelas, macro-F1 dengan CI 95%, akurasi, CM (`tn/fp/fn/tp`), ECE, Brier, Precision@50/100, *coverage*, macro-F1 selektif, dan kerugian harapan |
| `log_experiment(...)` | B1, B2, B3, D | Menambah satu baris ke log eksperimen. Formatnya 25 kolom, sama dengan `B2_EXPERIMENTS_BASELINE.csv` |
| `paired_bootstrap_test(y_true, pred_a, pred_b)` | B2 (E1), D | Selisih macro-F1 A − B, CI 95%, dan *p-value* satu arah |
| `compute_reliability_bins(y_true, y_prob)` | B3, D | Isi diagram reliabilitas dengan 10 bin |
| `evaluate_selective_prediction(y_true, y_prob, tau_low, tau_high, c_fp, c_fn, c_r)` | B3, D | *Coverage*, galat pada berita yang diputuskan, dan kerugian harapan selektif dibanding standar |
| `compute_ece`, `brier_score`, `precision_at_k`, `bootstrap_macro_f1_ci`, `thresholds_from_costs`, `confusion_matrix_dict` | D | Fungsi pembantu yang juga bisa dipanggil langsung |

### Konvensi
- Label 1 = HOAKS dan 0 = VALID. `y_prob` adalah peluang HOAKS.
- **ECE** dihitung dengan 10 bin selebar 0,1 pada peluang HOAKS. Selisih |rerata peluang − proporsi HOAKS| ditimbang dengan jumlah anggota bin. Rumus ini cocok dengan `B3_RELIABILITY_BINS_SVM.csv`.
- **CI 95% macro-F1** memakai bootstrap persentil 1000 kali dengan seed 42 (`np.random.RandomState`).
- **Aturan penolakan** memutuskan VALID jika p ≤ τ_low, HOAKS jika p ≥ τ_high, dan PERLU VERIFIKASI di antaranya. Bawaannya τ_low = 0,20 dan τ_high = 0,50, diturunkan dari C_FP = 2, C_FN = 5, dan C_R = 1. Nilai ini sama dengan `config.py` milik C.
- **Kerugian harapan** = (2·FP + C_FN·FN + 1·ditolak) / N.
- **Kunci UJI:** baris berstatus `UJI` ditolak oleh `log_experiment` kecuali dipanggil dengan `allow_uji=True`. Flag `--final` dan `uji_log.csv` akan dibuat di tugas D2.
- **Peringatan tafsir:** untuk SVM tanpa kalibrasi (sigmoid dari skor SVM), ECE dan *coverage* tetap dihitung tetapi **tidak bermakna**. Di laporan, pakai angka dari model terkalibrasi saja.

### Menjalankan skrip B dari repo
Skrip B tidak perlu diubah. Cukup arahkan `PYTHONPATH` ke folder prototipe sebelum menjalankannya.

Dari root repo, di Linux atau Colab:
```bash
PYTHONPATH=prototipe/anggota_c python notebooks/anggota_b/scripts/B2_run_experiments.py --data <CSV> --text-col text_clean_v31 --data-version V3.1_A2_PROVISIONAL
```

Di Windows PowerShell:
```powershell
$env:PYTHONPATH="prototipe/anggota_c"; python notebooks/anggota_b/scripts/B2_run_experiments.py --data <CSV> --text-col text_clean_v31 --data-version V3.1_A2_PROVISIONAL
```

### Verifikasi terhadap hasil resmi B
Skrip `d1_verifikasi_evaluate.py` menjalankan ulang B2 dan B3 apa adanya, lalu membandingkan setiap angka dengan `results/anggota_b/` pada toleransi 1e-6. Versi Colab-nya ada di `notebooks/anggota_d/D1_Verifikasi_Evaluate.ipynb`.

Uji pendahuluan memakai salinan V3.1 buatan ulang, bukan file A (SHA-256 berbeda). Python yang dipakai 3.13 dengan scikit-learn 1.6.1.

| Hasil B | Kesimpulan |
|---|---|
| 10 baris `text_clean_v31` dan `title` di `B2_EXPERIMENTS_BASELINE.csv` (E1 bersih, E2, judul saja, E4) | Semua kolom cocok persis, termasuk CI, Precision@k, Brier, *coverage*, dan kerugian, kecuali satu baris: SVM bersih selisih 1 berita di ambang 0,20 dan ECE berbeda 0,000001. Penyebabnya peluang SVM hasil pelatihan, bukan rumus |
| `B3_SELECTIVE_RISK_COVERAGE_SVM.csv` (C_FN 3, 5, 10) | 48/48 angka cocok persis |
| Baris kalibrasi Platt | Semua cocok kecuali ECE (0,003931 vs 0,003933) |
| Baris `text_raw` di E1 | Berbeda, karena kolom `text_raw` di salinan lokal tidak sama dengan file A |

Verifikasi akhir harus dijalankan dengan file A (SHA-256 `a790c0a6…`) lewat notebook D1. Hasilnya `HASIL_D1_VERIFIKASI.zip`. Simpan `D1_RINGKASAN.json` dan `D1_VERIFIKASI_EVALUATE.csv` ke `results/anggota_d/`.
