# Notebook Anggota D

| Notebook | Isi | Hasil |
|---|---|---|
| `D1_Verifikasi_Evaluate.ipynb` | Menjalankan ulang B2 dan B3 dengan `src/evaluate.py`, lalu membandingkan angkanya dengan `results/anggota_b/` | `HASIL_D1_VERIFIKASI.zip`: simpan `D1_RINGKASAN.json` dan `D1_VERIFIKASI_EVALUATE.csv` ke `results/anggota_d/` |
| `D3_Evaluasi_Agen.ipynb` | Menjalankan agen utuh milik C pada VALIDASI dan membandingkannya dengan model saja (C_FN 3/5/10) | `HASIL_D3_EVALUASI_AGEN.zip`: salin semua file `D3_*` ke `results/anggota_d/` |
| `D4_UJI_Final.ipynb` | Tahap 1 membekukan konfigurasi. Tahap 2 menjalankan 7 run UJI satu kali lewat kunci D2 | `PEMBEKUAN_UJI.json` (commit sebelum tahap 2) dan `HASIL_D4_UJI.zip` (salin ke `results/anggota_d/`) |

Cara menjalankan:
1. Buka notebook di Colab.
2. Isi `REPO_URL`, atau kosongkan lalu unggah zip repo.
3. Sesuaikan `DATASET_PATH` dengan letak file V3.1 di Drive.
4. Pilih Runtime → Run all.

Hanya LATIH dan VALIDASI yang dipakai. Penjelasan modul ada di `scripts/anggota_d/README.md`.
