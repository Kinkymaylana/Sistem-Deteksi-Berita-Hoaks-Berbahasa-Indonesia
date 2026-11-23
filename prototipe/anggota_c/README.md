# PROTOTIPE_KEL5 — Agen Hybrid Deteksi Berita Hoaks Berbahasa Indonesia

Kelompok 5, Pengantar Sistem Pintar, Tahap 3. Bagian Anggota C: agen, integrasi model, aplikasi, dan pengujian fungsional.

Agen menerima judul dan isi berita, lalu menghasilkan salah satu tindakan: **HOAKS**, **VALID**, atau **PERLU VERIFIKASI**.

> **Status (23 Nov 2026).**
> - Tahap V3.1 preprocessing sekarang memakai **file asli Anggota A**, `src/preprocess_v31.py`, tanpa diubah.
> - Model Anggota B **terpasang**: `models/LinearSVM_calibrated_v31.joblib` (dari repo, commit `af01067`, SHA-256 `7d8da944…51cbd6`) beserta `models/model_card.json`.
> - Dataset yang dipakai: `DATASET_V3_1_KANDIDAT_OPERASIONAL.csv`. SHA-256-nya `a790c0a6…bb295`, sama persis dengan `B2_BASELINE_METADATA.json`.
> - **Semua lapisan agen sudah aktif.** Artefak dibangun dari LATIH, ambang disetel di VALIDASI, UJI tidak dibaca:
>   - `models/language_rule.json`: τ_bahasa = 0,100; 0,86% teks VALIDASI tertolak.
>   - `models/similar_index.joblib`: 6.658 hoaks LATIH; τ_mirip = 0,90 dengan precision VALIDASI 11/11.
>   - `results/paritas_preprocess.csv`: CNN 100%, Kompas 99,4%, Tempo 99,2% identik, sehingga **lulus kriteria ≥ 99%**.
> - 87 tes otomatis lulus.
> - **Uji resmi sudah dijalankan** dengan Python 3.11.18 dan scikit-learn 1.6.1: `results/uji_fungsional.csv` lulus 10/10, dan `results/latency.csv` mencatat waktu total p50 15,4 ms serta p95 21,8 ms. Jika laptop demo berbeda, ulangi pengukuran latensi di laptop tersebut.

## 1. Apa yang sudah teruji dan yang belum

| Komponen | File | Teruji? | Keterangan |
|---|---|---|---|
| `decide(p)`, ambang dari biaya | `src/agent.py`, `src/config.py` | Ya | Termasuk nilai p tepat di batas 0,20 dan 0,50 |
| Preprocessing V3 + V3.1 | `src/preprocess.py`, `src/preprocess_v31.py` (A) | Ya | Tahap V3: salinan identik notebook EDA sel 6. Tahap V3.1: file asli A. Kesetaraan dengan `text_clean_v31` (500 berita per sumber, LATIH+VALIDASI): CNN 100%, Kompas 99,4%, Tempo 99,2%, TurnBackHoax 99,8–100%. Selisihnya karena label di awal judul ("Isu", "Berita", "Fakta") ikut dihapus |
| Integrasi model B | `src/agent.py` | Ya | Model dimuat, p dari agen sama dengan `predict_proba`, versi preprocessing dibaca dari model card |
| `scope_guard` (kosong, < 8 kata, bahasa) | `src/reactive.py` | Ya | τ_bahasa = 0,100 dari LATIH. Teks bahasa Inggris umumnya tersaring, tetapi **belum semuanya**, lihat bagian 5 butir 1 |
| `find_similar` + penyetelan τ_mirip | `src/reactive.py`, `scripts/build_similar_index.py` | Ya | τ_mirip = 0,90 (`results/tau_similar_validasi.csv`). Indeks disimpan dalam format portabel, tidak bergantung versi scikit-learn |
| Kontribusi kata SVM | `src/explain.py` | Ya, dengan model asli | Jumlah kontribusi + intercept sama dengan `decision_function` model B |
| LIME | `src/explain.py` | Ya | Berjalan pada skenario 10 uji fungsional dengan `lime` dari `requirements.txt` |
| Antrean CSV | `src/batch.py` | Ya | Sesuai Desain Metode 2.8.2 |
| Aplikasi Streamlit | `app/app.py` | Logika ya, tampilan **belum** | Diuji dengan modul streamlit tiruan, juga dengan model asli |
| Uji 10 skenario, latensi | `scripts/uji_fungsional.py`, `scripts/ukur_latensi.py` | Ya | Lulus 10/10. Latensi total p50 15,4 ms dan p95 21,8 ms dari 100 berita VALIDASI |

Tes dan uji resmi dijalankan dengan Python 3.11.18 dan scikit-learn 1.6.1, sama dengan versi scikit-learn model B. **Peluang yang dihitung di luar 1.6.1 tidak boleh dilaporkan.** Semua angka resmi harus dihasilkan di laptop dengan `requirements.txt`.

## 2. Instalasi (Windows, Python 3.11)

```bat
cd PROTOTIPE_KEL5
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

Aplikasi sudah bisa dibuka dengan semua lapisan aktif:

```bat
streamlit run app/app.py
```

## 3. Urutan menjalankan

Langkah 2–4 sudah dijalankan oleh Anggota C, dan hasilnya ada di `models/` dan `results/`. Ulangi hanya jika dataset atau aturan berubah. Langkah 5 wajib dijalankan di laptop demo.

1. Letakkan `DATASET_V3_1_KANDIDAT_OPERASIONAL.csv` di folder `data/`.
2. **Cek kesetaraan preprocessing.** Target minimal 99% identik untuk CNN, Kompas, dan Tempo.
   ```bat
   python scripts/uji_paritas_preprocess.py --data data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv --text-col text_clean_v31
   ```
3. **Bangun aturan bahasa** (hanya dari LATIH):
   ```bat
   python scripts/build_language_rule.py --data data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv --text-col text_clean_v31
   ```
4. **Bangun indeks hoaks serupa dan setel τ_mirip** (hanya di VALIDASI):
   ```bat
   python scripts/build_similar_index.py --data data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv --text-col text_clean_v31
   ```
5. **Jalankan uji fungsional dan latensi**, lalu buka aplikasi dan ambil tangkapan layar setiap skenario:
   ```bat
   python scripts/uji_fungsional.py --data data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv
   python scripts/ukur_latensi.py --data data\DATASET_V3_1_KANDIDAT_OPERASIONAL.csv
   streamlit run app/app.py
   ```

Tidak ada skrip yang membaca baris UJI. `scripts/_common.py` menolaknya secara eksplisit.

## 4. Alur agen yang diimplementasikan

```
title, body
  │
  ├─ [0] Persepsi     clean_news(title, body): aturan V3 lalu V3.1   src/preprocess.py
  ├─ [1] Reaktif      scope_guard pada teks bersih                    src/reactive.py
  │                    kosong / < 8 kata / bukan bahasa Indonesia → PERLU VERIFIKASI
  ├─ [1] Reaktif      find_similar (TF-IDF, cosine)                   src/reactive.py
  │                    skor ≥ τ_mirip → HOAKS + hoaks lama yang cocok
  ├─ [2] Deliberatif  p = LinearSVM_calibrated_v31 (Platt, Anggota B)
  │                    decide(p): VALID ≤ 0,20 < PERLU VERIFIKASI < 0,50 ≤ HOAKS
  └─ [3] Aktuator     app/app.py: label, p dalam persen, alasan, hoaks mirip,
                       tombol Jelaskan, antrean CSV
```

Ambang 0,20 dan 0,50 dihitung dari biaya C_FP = 2, C_FN = 5, C_R = 1 dengan aturan biaya terkecil:
- VALID jika p·C_FN ≤ C_R, yaitu p ≤ 1/5.
- HOAKS jika (1−p)·C_FP ≤ C_R, yaitu p ≥ 1/2.

Hasil ini sama dengan `tau_low` dan `tau_high` di model card B.

Format keluaran `predict(title, body)`:

```python
{
  "action": "HOAKS" | "VALID" | "PERLU VERIFIKASI",
  "layer": "scope_guard" | "similar" | "model",
  "p_hoaks": 0.83,           # None jika diputuskan lapisan reaktif
  "reason": "p = 0,83 mencapai ambang HOAKS 0,50",
  "similar": {"title": "...", "url": "...", "score": 0.94},  # atau None
  "model": "LinearSVM-platt"
}
```

Untuk berita yang diputuskan model, `similar` berisi hoaks lama terdekat sebagai pembanding. Hoaks ini tidak memengaruhi keputusan (Desain Metode 2.8.1 butir 4).

## 5. Catatan yang perlu dikonfirmasi kelompok

1. **Aturan bahasa (Desain Metode 2.5.1) diperketat karena sistem khusus berita berbahasa Indonesia.**
   - Kata "the" dan "i" sebelumnya termasuk 500 kata tersering LATIH. Keduanya sekarang dikeluarkan lewat `lang_exclude` di `src/config.py`, dan posisinya terisi "adapun" dan "kemungkinan".
   - τ_bahasa = persentil ke-1 LATIH sekarang 0,100 (sebelumnya 0,111), dan 0,86% teks VALIDASI tertolak (sebelumnya 1,03%). Ambang tetap rendah karena teks TurnBackHoax pendek dan informal.
   - Dari 12 contoh kalimat Inggris buatan, 9 ditolak dan 4 contoh kalimat Indonesia tetap lolos. Tiga kalimat Inggris masih lolos karena memuat nama atau kata serapan seperti "Jakarta", "Joko Widodo", "video", dan "media".
   - Menaikkan τ akan ikut menolak sekitar separuh hoaks berbahasa Indonesia, sehingga bukan solusi.
   - Pengecualian kata ini menyimpang dari Desain Metode, jadi perlu disetujui kelompok dan dicatat di laporan.
2. **Preprocessing.**
   - Tahap V3.1 memakai `src/preprocess_v31.py` dari A (diterima 23 Nov). File ini identik dengan salinan C, dan mereproduksi `text_clean_v31` pada 26.270 dari 26.270 baris LATIH+VALIDASI.
   - Tahap V3 (teks mentah → `text_clean`) tidak ada di file A. Tahap ini tetap salinan persis notebook EDA sel 6.
3. **Sumber berita tidak diketahui di aplikasi.** A2 menghapus kredit foto dan dateline "KOMPAS.com -" hanya untuk berita Kompas.
   - File A menyediakan `clean_v31_for_unknown_source()`, yang tidak menerapkan aturan Kompas.
   - Aplikasi C tetap menerapkan aturan Kompas.
   - Bukti dari teks mentah ke `text_clean_v31` pada seluruh LATIH+VALIDASI:

     | Sumber | Mode A | Mode C |
     |---|---|---|
     | CNN Indonesia | 99,78% | 99,78% |
     | Kompas | 94,98% | 99,09% |
     | Tempo | 99,29% | 99,29% |
     | TurnBackHoax | 99,99% | 99,88% |

   - Mode A tidak memenuhi kriteria ≥ 99% untuk Kompas. Biaya mode C adalah 8 teks hoaks yang mengutip "kompas.com -". Perlu disepakati dengan A.
4. **Regex kredit foto milik A** (`PHOTO`, dengan flag `(?i)`) juga menghapus kata biasa setelah nama fotografer, sampai 60 karakter. Contoh: "KOMPAS.com/ANDI PRATAMA hujan" ikut menghapus "hujan". Aplikasi meniru perilaku ini agar sama dengan data latih. Perbaikannya keputusan A, dan model harus dilatih ulang jika aturannya diubah.
5. **Dataset V3.1 masih berstatus `V3.1_A2_PROVISIONAL`.** Ini tercatat di model card B dan ringkasan A2.
6. **`src/evaluate.py` milik D belum ada di repo**, padahal skrip B mengimpornya.
7. **Diagram "Page-1" di `Flowchart PSP.json` belum memuat lapisan hoaks serupa.** Diagram di laporan perlu mengikuti bagian 4 README ini.
8. **τ_mirip = 0,90 didukung hanya 11 berita VALIDASI yang cocok** (semuanya HOAKS). Precision 11/11 belum cukup untuk membuktikan precision ≥ 0,99 secara statistik. Ini perlu ditulis D di bagian 4.3 laporan.

## 6. Struktur folder

| Path | Pemilik | Status |
|---|---|---|
| `src/config.py`, `src/agent.py`, `src/reactive.py`, `src/explain.py`, `src/batch.py` | C | Ada |
| `src/preprocess_v31.py` | A | **Ada** (file asli A, tidak diubah) |
| `src/preprocess.py` | A (tahap V3) dan C (pembungkus) | Ada; tahap V3 salinan notebook EDA A |
| `app/app.py` | C | Ada |
| `scripts/build_language_rule.py`, `build_similar_index.py`, `uji_paritas_preprocess.py`, `uji_fungsional.py`, `ukur_latensi.py` | C | Ada |
| `tests/` | C | 87 tes |
| `models/LinearSVM_calibrated_v31.joblib`, `models/model_card.json` | B | **Terpasang** (disalin dari repo B tanpa diubah) |
| `models/language_rule.json`, `similar_index.joblib` | C | **Ada** (dari dataset V3.1; indeks 22 Nov, aturan bahasa dibangun ulang 23 Nov) |
| `results/paritas_preprocess.csv`, `tau_similar_validasi.csv` | C | **Ada** (VALIDASI dan LATIH) |
| `results/uji_fungsional.csv`, `latency.csv`, `latency_detail.csv` | C | **Ada** (23 Nov, Python 3.11.18, scikit-learn 1.6.1) |
| `src/evaluate.py` | D | Belum diserahkan |
| `scripts/tune.py`, `run_experiments.py`, `calibrate.py` | B | Ada di repo sebagai `notebooks/anggota_b/scripts/B1–B3`, belum disalin ke sini |
