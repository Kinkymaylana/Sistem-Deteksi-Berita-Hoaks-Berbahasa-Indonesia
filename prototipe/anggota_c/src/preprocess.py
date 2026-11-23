"""Preprocessing inferensi -- VERSI SEMENTARA, disusun Anggota C.

STATUS: menunggu verifikasi dan pengganti resmi dari Anggota A.

Asal kode
---------
Dua tahap, keduanya DISALIN APA ADANYA dari kode Anggota A
(repo GitHub Kinkymaylana/Sistem-Deteksi-Berita-Hoaks-Berbahasa-Indonesia,
commit af01067, 22 Nov 2026):

  Tahap V3   : regex, clean_piece(), tag_info()
               notebooks/anggota_a/EDA_Preprocessing_V3_1.ipynb, sel 6 (baris 12-56)
  Tahap V3.1 : NAV, DATELINE, PHOTO, clean_text_v31()
               DIIMPOR dari src/preprocess_v31.py = file asli Anggota A
               (diterima 23 Nov 2026, isi sama dengan A2_PEMBERSIHAN_V3_1_COLAB.ipynb sel 4)

Tidak ada aturan yang diubah. Yang ditambahkan Anggota C hanya pembungkus
clean_news(title, body) dan clean_text(text) untuk input pengguna.

Versi mana yang dipakai aplikasi
--------------------------------
Model LinearSVM_calibrated_v31 dari Anggota B dilatih pada text_clean_v31
(B3_MODEL_CARD_BASELINE.json: data_version "V3.1_A2_PROVISIONAL";
B2_BASELINE_METADATA.json: preprocess "Kolom text_clean_v31"). Karena itu
default v31=True. load_default_agent() memilih versi dari model card.

Cara dataset membentuk teks bersih (dari kode A)
------------------------------------------------
  V3 VALID : clean_piece(title) + " " + clean_piece(body, body=True)
  V3 HOAKS : clean_piece(judul tanpa label [SALAH] dsb.) + " " + clean_piece(NARASI)
  V3.1     : clean_text_v31(text_clean V3, source)
Saat inferensi label dan sumber tidak diketahui, sehingga clean_news():
  1. clean_piece(judul tanpa label) + " " + clean_piece(body, body=True)
  2. clean_text_v31(hasil, source)  -- source=None berarti sumber tidak diketahui
Keputusan C untuk sumber tidak diketahui: aturan khusus Kompas (kredit foto
dan dateline "KOMPAS.com -") TETAP diterapkan, karena di data latih semua
berita Kompas sudah bersih dari penanda itu. Ini berbeda dari
clean_v31_for_unknown_source() di file A, yang tidak menerapkan aturan Kompas.
Bukti (teks mentah -> text_clean_v31, seluruh LATIH+VALIDASI, 23 Nov 2026):
  sumber         mode A (UNKNOWN)   mode C (aturan Kompas)
  CNN Indonesia      99,78%             99,78%
  Kompas             94,98%             99,09%
  Tempo              99,29%             99,29%
  TurnBackHoax       99,99%             99,88%
Mode A tidak memenuhi kriteria >= 99% untuk Kompas; mode C memenuhi untuk
ketiga sumber berita, dengan biaya 8 teks hoaks yang mengutip "kompas.com -".
Mode A tetap bisa dipakai dengan clean_news(..., source="UNKNOWN").
Penghapusan label di judul diperlukan agar label seperti "[SALAH]" yang
diketik pengguna tidak menjadi petunjuk bagi model.
Ekstraksi NARASI tidak dipakai karena format itu khusus artikel TurnBackHoax.

Kesetaraan dengan text_clean dataset belum terbukti. Jalankan
scripts/uji_paritas_preprocess.py setelah dataset tersedia.
"""
from __future__ import annotations

import collections
import re
import unicodedata

# ----------------------------------------------------------------------------
# Disalin apa adanya dari pipeline V3 Anggota A (sel 6, baris 12-28)
# ----------------------------------------------------------------------------
WORD_RE = re.compile(r"(?u)\b[\w]+(?:[-’'][\w]+)*\b")
POS = {'SALAH', 'HOAX', 'HOAKS', 'FALSE', 'HASUT', 'FITNAH', 'DISINFORMASI', 'MISINFORMASI', 'SCAM'}
NEG = {'BENAR', 'TRUE', 'KLARIFIKASI', 'EDUKASI', 'KOREKSI', 'BERITA', 'CEK FAKTA', 'CEKFAKTA', 'CLARIFICATION', 'ACARA', 'FAKTA', 'ISU', 'OPINI', 'ARTIKEL', 'PENGUMUMAN'}
TAGALT = '|'.join(re.escape(x) for x in sorted(POS | NEG, key=len, reverse=True))
RE_PREFIX = re.compile(r'^\s*(?:[\[(]\s*)?(?P<tags>' + TAGALT + r')\b(?:\s*(?:[/+,;]|\s+dan\s+)\s*(?:' + TAGALT + r')\b){0,5}\s*(?:[\])]\s*)?\s*(?:[:：\-–—]\s*)?', re.I)
RE_PARTS = re.compile(r'\s*(?:/|\+|,|;|\bdan\b)\s*', re.I)
RE_URL = re.compile(r'(?i)(?:https?://|www\.)\S+')
RE_HTML = re.compile(r'<[^<>]{0,300}>')
RE_CNN = re.compile(r'(?i)^\s*(?:[A-ZÀ-ÿ][\w\s.-]{1,40},\s*)?CNN\s+Indonesia\s*(?:--|[-–—])\s*')
RE_KOMPAS = re.compile(r'(?i)^\s*(?:[A-ZÀ-ÿ][\w\s.-]{1,40},\s*)?KOMPAS\s*\.\s*com\s*(?:--|[-–—])\s*')
RE_TEMPO = re.compile(r'(?i)^\s*(?:TEMPO\s*\.\s*CO|TEMPO)\s*,?\s*(?:[A-ZÀ-ÿ][\w\s.-]{1,35},?\s*)?(?:--|[-–—])\s*')
RE_BOILER = re.compile(r'(?i)\b(?:ADVERTISEMENT|SCROLL TO RESUME CONTENT|LEBIH BANYAK DARI KOLUMNIS)\b')
RE_GAMBAS = re.compile(r'(?i)\[Gambas:[^\]]{1,150}\]')


def clean_piece(s, body=False):
    """Disalin apa adanya dari pipeline V3 Anggota A (sel 6, baris 37-47)."""
    s = unicodedata.normalize('NFKC', str(s or '')).replace(' ', ' ').replace('­', '')
    if body:
        for rex in [RE_CNN, RE_KOMPAS, RE_TEMPO]:
            s = rex.sub('', s, count=1)
    s = RE_GAMBAS.sub(' ', s)
    s = re.sub(r'(?:\s*=\s*){3,}', ' ', s)
    s = RE_BOILER.sub(' ', s)
    s = RE_URL.sub(' ', s)
    s = RE_HTML.sub(' ', s)
    s = re.sub(r'(?im)^\s*(?:LIHAT JUGA|BACA JUGA)\s*:\s*[^\n]{0,250}', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def tag_info(title):
    """Disalin apa adanya dari pipeline V3 Anggota A (sel 6, baris 49-56)."""
    s = str(title or '').strip()
    m = RE_PREFIX.match(s)
    if not m:
        return [], s, 'NO_TAG'
    raw = m.group('tags')
    tags = [k.strip().upper() for k in RE_PARTS.split(raw) if k.strip()]
    tags = [t for t in tags if t in POS | NEG]
    tail = s[m.end():].lstrip(' :\t-–—')
    if not tags:
        return [], s, 'UNKNOWN_TAG'
    return tags, tail, 'RECOGNIZED'


# ----------------------------------------------------------------------------
# Tahap V3.1: diimpor dari file asli Anggota A (src/preprocess_v31.py, tidak diubah).
# File diterima 23 Nov 2026; identik perilakunya dengan A2 sel 4 dan mereproduksi
# text_clean_v31 pada 26.270/26.270 baris LATIH+VALIDASI.
# ----------------------------------------------------------------------------
from .preprocess_v31 import DATELINE, NAV, PHOTO, clean_text_v31  # noqa: E402,F401


# ----------------------------------------------------------------------------
# Pembungkus untuk input pengguna (ditambahkan Anggota C)
# ----------------------------------------------------------------------------
UNKNOWN_SOURCE_AS = 'Kompas'  # lihat docstring modul: aturan Kompas diterapkan bila sumber tak diketahui


def clean_news(title: str | None, body: str | None, v31: bool = True, source: str | None = None) -> str:
    """Bersihkan judul + isi berita dari pengguna, judul selalu di depan.

    v31=True  : bentuk text_clean_v31 (model B saat ini).
    v31=False : bentuk text_clean V3.
    source    : nama sumber jika diketahui (dipakai uji paritas); None untuk input pengguna.
    """
    _, title_wo_tag, _ = tag_info(title)
    out = (clean_piece(title_wo_tag) + ' ' + clean_piece(body, body=True)).strip()
    if not v31:
        return out
    return clean_text_v31(out, source if source is not None else UNKNOWN_SOURCE_AS)[0]


def clean_text(text: str | None, v31: bool = True) -> str:
    """Bersihkan satu teks tanpa pemisahan judul (antarmuka sesuai rencana kerja)."""
    return clean_news(None, text, v31=v31)
