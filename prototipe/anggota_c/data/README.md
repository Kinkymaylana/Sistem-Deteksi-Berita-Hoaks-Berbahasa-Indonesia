# data/

File dataset tidak dimasukkan ke zip karena ukurannya sekitar 170 MB.

| File | Pemilik | Lokasi |
|---|---|---|
| `DATASET_V3_1_KANDIDAT_OPERASIONAL.csv` (175.489.858 byte, Drive milik Bintang) | Anggota A | Tautan Google Drive: **BELUM DIISI** |

SHA-256 yang benar: `a790c0a6175455f385529f98a48dbd5ee8d939805494168ad698bd56343bb295`. Nilai ini sama dengan `B2_BASELINE_METADATA.json` milik B. Cek di Windows dengan `certutil -hashfile data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv SHA256`.

Kolom yang dibaca skrip Anggota C: `raw_id`, `source`, `label`, `title`, `url`, `text_raw`, kolom teks bersih, dan `random_split`.

Kolom teks bersih dipilih lewat `--text-col`. Pakai **`text_clean_v31`**, karena model B saat ini dilatih pada kolom itu (`B2_BASELINE_METADATA.json`: preprocess "Kolom text_clean_v31").

Hanya baris `LATIH` dan `VALIDASI` yang dibaca. Baris `UJI` ditolak oleh `scripts/_common.py`.
