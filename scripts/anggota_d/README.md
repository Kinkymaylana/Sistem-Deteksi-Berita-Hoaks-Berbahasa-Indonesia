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

**Hasil akhir (23 Nov, Colab, file V3.1 milik A, SHA-256 `a790c0a6…` cocok):** 500 dari 501 angka sama persis dengan hasil B. Rinciannya ada di `results/anggota_d/D1_RINGKASAN.json` dan `D1_VERIFIKASI_EVALUATE.csv`.

| File B | Dibandingkan | Beda |
|---|---|---|
| `B2_EXPERIMENTS_BASELINE.csv` (12 baris E1, E2, judul saja, E4) | 276 | 0 |
| Baris kalibrasi Platt (B3) | 23 | 0 |
| `B3_RELIABILITY_BINS_SVM.csv` | 80 | 0 |
| `B3_SELECTIVE_RISK_COVERAGE_SVM.csv` | 48 | 0 |
| `B2_E2`, `B2_E4`, `B2_TITLE_ONLY` | 56 | 0 |
| `B2_E1_SHORTCUT_ANALYSIS.csv` | 18 | 1 |

**Satu angka yang berbeda:** `p_value_raw_vs_clean` untuk LinearSVM. Nilai B 0,002, sedangkan `evaluate.py` menghasilkan 0,000.
- Prediksi kedua model identik, karena macro-F1, CI, dan selisihnya sama persis. Artinya, bedanya hanya di cara menghitung *p-value* bootstrap.
- Di antara 1000 ulangan, B mendapat 2 ulangan dengan selisih ≤ 0, sedangkan `evaluate.py` mendapat 0.
- Rumus B tidak bisa dipastikan dari satu angka ini. Ada 26 varian yang sama-sama cocok, misalnya generator `default_rng`, uji terpusat (*Berg-Kirkpatrick*), atau *p* dua arah.
- **Kesimpulan E1 tidak berubah:** kedua *p-value* jauh di bawah 0,05, sehingga NB dan SVM sama-sama signifikan.
- `evaluate.py` memakai uji satu arah yang umum dipakai (Koehn, 2004): *p* = proporsi ulangan dengan selisih ≤ 0, dengan `RandomState(42)`. Di laporan, tulis *p* < 0,001, karena 1000 ulangan tidak bisa menghasilkan *p* lebih kecil dari 0,001.

Catatan lingkungan: Colab menjalankan Python 3.13.18 dengan scikit-learn 1.6.1, sedangkan B memakai Python 3.11.9 dengan scikit-learn 1.6.1. Hasilnya tetap sama.

## D2. Kunci UJI

**Modul:** `prototipe/anggota_c/src/uji_lock.py`. **Tes:** `prototipe/anggota_c/tests/test_uji_lock.py` (16 tes).

### Gambaran singkat
Data UJI hanya dibuka **satu kali per run**, setelah keputusan kelompok dibekukan. Setiap upaya membuka UJI, baik berhasil maupun ditolak, dicatat di `results/anggota_d/uji_log.csv`.

| File | Isi |
|---|---|
| `results/anggota_d/PEMBEKUAN_UJI_DRAF.json` | Draf yang diisi kelompok: 5 keputusan, 7 run UJI yang direncanakan, dan daftar berkas yang dibekukan |
| `results/anggota_d/PEMBEKUAN_UJI.json` | Manifest resmi, dibuat `d2_bekukan.py`. Berisi SHA-256 dataset, SHA-256 setiap berkas, nama yang menyetujui, dan tanggal |
| `results/anggota_d/uji_log.csv` | Jejak akses: `waktu, run_key, status, operator, skrip, split, n_baris_uji, dataset_sha256, manifest_sha256, git_commit, alasan, ringkasan` |

### Syarat agar UJI terbuka
Semua syarat ini harus terpenuhi:
1. Skrip dipanggil dengan `--final`.
2. `PEMBEKUAN_UJI.json` ada dan berstatus `DIBEKUKAN`, dan semua keputusannya `DISEPAKATI`.
3. SHA-256 dataset sama dengan yang tercatat di manifest.
4. SHA-256 setiap berkas yang dibekukan belum berubah. Ini mencegah kode atau model diubah setelah pembekuan.
5. `run_key` terdaftar di manifest, dengan split yang sama.
6. Run itu belum pernah `SELESAI`. Mengulang hanya bisa dengan `--alasan-ulang "..."`, tercatat sebagai `ULANG`, dan wajib dilaporkan.

### Status di `uji_log.csv`
| Status | Arti |
|---|---|
| `DITOLAK` | Ada syarat yang tidak terpenuhi. Data tidak dibaca |
| `DIBUKA` | Run dimulai |
| `SELESAI` | Run selesai, dengan ringkasan macro-F1, CI, ECE, dan *coverage* |
| `GAGAL` | Error di tengah jalan. Run ini boleh dicoba lagi tanpa alasan, karena hasilnya belum ada |
| `ULANG` | Run yang sudah selesai dijalankan lagi dengan alasan tertulis |
| `BATAL_BEKU` | Pembekuan dibatalkan. Ini hanya bisa dilakukan selama UJI belum pernah dibuka |

### Urutan pemakaian
1. Kelompok membahas 5 keputusan di `PEMBEKUAN_UJI_DRAF.json`. Untuk setiap keputusan, ubah `status` menjadi `DISEPAKATI` dan isi `keputusan_akhir`.
   - K1: data V3.1
   - K2: hyperparameter
   - K3: catatan C nomor 1, 3, dan 4
   - K4: E4 resmi
   - K5: konfigurasi agen
2. Tambahkan skrip UJI final D4 ke `berkas_dibekukan` begitu skrip itu selesai dibuat.
3. Periksa draf tanpa menulis apa pun:
   ```
   python scripts/anggota_d/d2_bekukan.py --dataset <CSV V3.1> --cek
   ```
4. Bekukan:
   ```
   python scripts/anggota_d/d2_bekukan.py --dataset <CSV V3.1> --disetujui "Anggota A,Anggota B,Anggota C,Anggota D"
   ```
5. **Commit `PEMBEKUAN_UJI.json` sebelum UJI dijalankan.** Dengan begitu, riwayat git membuktikan pembekuan terjadi lebih dulu.
6. Jalankan skrip UJI D4 dengan `--final`.
7. Cek status kapan saja (tidak membuka UJI):
   ```
   python scripts/anggota_d/d2_status_uji.py
   ```
8. Commit `uji_log.csv` bersama hasil UJI.

### Contoh pemakaian di skrip UJI
```python
from src.evaluate import calculate_metrics
from src.uji_lock import add_final_args, open_uji

ap = argparse.ArgumentParser(); ap.add_argument("--dataset"); add_final_args(ap); a = ap.parse_args()
with open_uji(a.dataset, run_key="UJI_LinearSVM_random_clean", split_col="random_split",
              final=a.final, manifest_path=a.manifest, log_path=a.uji_log,
              operator=a.operator, alasan_ulang=a.alasan_ulang) as sesi:
    latih, uji = sesi.load_latih(), sesi.load()
    ...  # latih model dengan konfigurasi dari sesi.run, prediksi uji
    sesi.log_experiment("results/anggota_d/experiments_uji.csv", calculate_metrics(y, pred, y_prob=prob))
```

### Batasan
Ini **kunci prosedural**, bukan enkripsi. Orang tetap bisa membuka CSV secara manual. Fungsinya adalah membuat setiap akses lewat kode kelompok tercatat dan bisa diperiksa. Hasilnya, laporan dapat menyatakan bahwa UJI dibuka sekali setelah pembekuan, lengkap dengan bukti tanggal, SHA-256, dan commit.
