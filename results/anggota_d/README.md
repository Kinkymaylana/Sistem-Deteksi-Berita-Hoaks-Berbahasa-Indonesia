# Hasil Anggota D

## D1 — Verifikasi `src/evaluate.py`
- `D1_RINGKASAN.json`: SHA-256 dataset, versi Python dan scikit-learn, serta jumlah angka yang cocok dan berbeda per file
- `D1_VERIFIKASI_EVALUATE.csv`: satu baris per angka yang dibandingkan (file, baris, kolom, nilai B, nilai ulang, cocok)

Kedua file berasal dari notebook `notebooks/anggota_d/D1_Verifikasi_Evaluate.ipynb` yang dijalankan dengan file V3.1 milik A pada 23 Nov.

**Hasil:** 500 dari 501 angka sama dengan hasil B. Satu-satunya beda adalah *p-value* E1 SVM (0,002 vs 0,000), dan kesimpulannya tetap sama. Penjelasan lengkap ada di `scripts/anggota_d/README.md`.

## D2 — Kunci UJI
- `PEMBEKUAN_UJI_DRAF.json`: draf keputusan yang diisi kelompok. Statusnya masih DRAF, dan kelima keputusan BELUM_DISEPAKATI
- `PEMBEKUAN_UJI.json`: dibuat `scripts/anggota_d/d2_bekukan.py` setelah kelima keputusan disepakati. **Belum ada**
- `uji_log.csv`: dibuat otomatis saat UJI pertama kali dicoba dibuka. **Belum ada**

Cara pakai dijelaskan di `scripts/anggota_d/README.md` bagian D2.
