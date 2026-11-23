"""Konfigurasi agen (pemilik: Anggota C).

Semua angka yang menentukan perilaku agen dikumpulkan di sini, supaya
Anggota D bisa menjalankan analisis risk-coverage (C_FN = 3, 5, 10) tanpa
mengubah kode agen.

Sumber angka: Rencana Kerja Tahap 3 Kelompok 5, bagian 5 (Anggota C, butir 2).
  - decide : VALID jika p <= 0,20, HOAKS jika p >= 0,50, PERLU VERIFIKASI di antaranya.
  - Biaya  : C_FP = 2, C_FN = 5, C_R = 1.
  - scope_guard : teks kurang dari 8 kata -> PERLU VERIFIKASI.
  - Aturan bahasa : 500 kata paling sering di data LATIH, tau_bahasa persentil ke-1.

Hubungan biaya dan ambang (aturan penolakan berbasis biaya, Chow):
  biaya harapan memutuskan VALID = p * C_FN
  biaya harapan memutuskan HOAKS = (1 - p) * C_FP
  biaya menyerahkan ke manusia   = C_R
  Pilih tindakan dengan biaya terkecil, sehingga
    VALID  jika p <= C_R / C_FN        = 1/5 = 0,20
    HOAKS  jika p >= 1 - C_R / C_FP    = 1 - 1/2 = 0,50
  Jadi ambang 0,20 dan 0,50 di rencana adalah turunan langsung dari biaya
  2/5/1. Fungsi thresholds_from_costs() menghitungnya, dan tes memastikan
  hasilnya sama dengan ambang di rencana.
"""
from __future__ import annotations

from dataclasses import dataclass

# Label tindakan. Ditulis persis seperti format predict() yang disepakati.
HOAKS = "HOAKS"
VALID = "VALID"
PERLU_VERIFIKASI = "PERLU VERIFIKASI"
ACTIONS = (HOAKS, VALID, PERLU_VERIFIKASI)

# Nama lapisan yang memutuskan.
LAYER_SCOPE_GUARD = "scope_guard"
LAYER_SIMILAR = "similar"
LAYER_MODEL = "model"


@dataclass(frozen=True)
class AgentConfig:
    # Biaya keputusan
    c_fp: float = 2.0   # biaya berita VALID dituduh HOAKS
    c_fn: float = 5.0   # biaya berita HOAKS diloloskan sebagai VALID
    c_r: float = 1.0    # biaya menyerahkan ke pemeriksa manusia

    # scope_guard
    min_words: int = 8              # teks < 8 kata -> PERLU VERIFIKASI
    lang_top_n: int = 500           # jumlah kata paling sering dari LATIH
    lang_percentile: float = 1.0    # tau_bahasa = persentil ke-1 skor LATIH
    lang_exclude: tuple[str, ...] = ("the", "i")  # kata Inggris yang tidak boleh masuk kosakata bahasa

    # find_similar (diisi setelah penyetelan di VALIDASI; titik awal 0,90)
    tau_similar: float = 0.90

    # Nama model yang ditulis di keluaran predict()
    model_name: str = "LinearSVM-platt"


DEFAULT_CONFIG = AgentConfig()


def thresholds_from_costs(c_fp: float, c_fn: float, c_r: float) -> tuple[float, float]:
    """Kembalikan (ambang_valid, ambang_hoaks) dari biaya keputusan.

    VALID jika p <= ambang_valid, HOAKS jika p >= ambang_hoaks.
    Penolakan (PERLU VERIFIKASI) hanya masuk akal jika C_R lebih kecil dari
    biaya salah, sehingga ambang_valid < ambang_hoaks.
    """
    if min(c_fp, c_fn, c_r) <= 0:
        raise ValueError("Semua biaya harus positif.")
    t_valid = c_r / c_fn
    t_hoaks = 1.0 - c_r / c_fp
    if not t_valid < t_hoaks:
        raise ValueError(
            f"Biaya tidak menghasilkan zona PERLU VERIFIKASI "
            f"(ambang VALID {t_valid:.3f} >= ambang HOAKS {t_hoaks:.3f})."
        )
    return t_valid, t_hoaks
