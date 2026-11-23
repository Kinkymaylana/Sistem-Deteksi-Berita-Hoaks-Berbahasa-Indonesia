
"""Antarmuka Streamlit Sistem Deteksi Berita Hoaks Berbahasa Indonesia.

File ini hanya mengatur tampilan. Logika model dan agen tetap
menggunakan modul asli di folder src/.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import MODEL_FILE, MODELS_DIR, load_default_agent
from src.batch import process, queue
from src.config import HOAKS, VALID, PERLU_VERIFIKASI, thresholds_from_costs
from src.explain import svm_word_contributions, lime_explanation


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Deteksi Hoaks ",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 14px;
    }

    div.stButton > button[kind="primary"] {
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


LAYER_NAMES = {
    "scope_guard": "Pemeriksaan kelayakan input",
    "similar": "Pencarian hoaks serupa",
    "model": "Model klasifikasi SVM",
}


# =========================================================
# MEMUAT AGEN
# =========================================================

@st.cache_resource(
    show_spinner="Memuat model dan komponen analisis..."
)
def get_agent():
    return load_default_agent()


def format_angka(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


try:
    agent = get_agent()

except FileNotFoundError:
    st.error(
        f"Model belum tersedia. Periksa file `{MODEL_FILE}` "
        f"di folder `{MODELS_DIR}`."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Tentang Sistem")

    st.write(
        "Aplikasi ini membantu melakukan analisis awal "
        "berita berbahasa Indonesia menggunakan "
        "pemrosesan teks, pencarian hoaks serupa, "
        "dan model machine learning."
    )

    st.divider()

    st.subheader("Kesiapan Komponen")

    status = agent.status()

    komponen = [
        ("preprocess", "Pemrosesan teks"),
        ("scope_guard_panjang", "Pemeriksaan panjang"),
        ("scope_guard_bahasa", "Pemeriksaan bahasa"),
        ("find_similar", "Pencarian berita serupa"),
        ("model", "Model klasifikasi"),
    ]

    for key, nama in komponen:
        indikator = "✅" if status.get(key) else "⚪"
        st.write(f"{indikator} {nama}")

    st.caption(
        "Indikator menunjukkan komponen telah dimuat, "
        "bukan jaminan seluruh pengujian berhasil."
    )

    st.divider()

    with st.expander("Informasi Teknis"):

        t_valid, t_hoaks = thresholds_from_costs(
            agent.config.c_fp,
            agent.config.c_fn,
            agent.config.c_r,
        )

        st.write("**Model:** Linear SVM Terkalibrasi")

        st.write(
            "**Preprocessing:**",
            status.get("preprocess_version") or "-",
        )

        st.write(
            f"**Ambang VALID:** ≤ {format_angka(t_valid)}"
        )

        st.write(
            f"**Ambang HOAKS:** ≥ {format_angka(t_hoaks)}"
        )

        st.write(
            "**Ambang kemiripan:**",
            format_angka(agent.config.tau_similar),
        )

        if getattr(agent, "model_card", None):
            st.write(
                "**Versi dataset:**",
                agent.model_card.get("data_version", "-"),
            )

    for warning in agent.warnings:
        with st.expander("Catatan Sistem"):
            st.warning(warning)

    st.divider()

    st.caption(
        "Note: Sistem Deteksi Hoaks. "
        "Keputusan akhir memerlukan verifikasi manusia."
    )


# =========================================================
# HEADER UTAMA
# =========================================================

st.title("Sistem Deteksi Berita Hoaks Berbahasa Indonesia")

st.subheader(
    "Berbasis Machine Learning dengan Evaluasi Shortcut Learning"
)

st.write(
    "Periksa teks berita menggunakan sistem berbasis "
    "agen hybrid yang menggabungkan klasifikasi "
    "machine learning dan pencarian hoaks serupa."
)

st.caption(
    "Sistem memberikan analisis awal dan tidak "
    "melakukan verifikasi langsung ke sumber eksternal."
)

st.divider()


# =========================================================
# FUNGSI MENAMPILKAN HASIL
# =========================================================

def tampilkan_hasil(res):

    st.subheader("Hasil Analisis")

    tindakan = res["action"]

    if tindakan == HOAKS:

        st.error(
            "**HOAKS** — Sistem mengindikasikan "
            "berita perlu diwaspadai.",
            icon="🚩",
        )

    elif tindakan == VALID:

        st.success(
            "**VALID** — Sistem mengklasifikasikan "
            "berita sebagai valid.",
            icon="✅",
        )

    else:

        st.warning(
            "**PERLU VERIFIKASI** — Pemeriksaan "
            "lebih lanjut diperlukan.",
            icon="⚠️",
        )

    kolom1, kolom2 = st.columns(2)

    with kolom1:

        st.markdown("**Metode Penentuan Hasil**")

        st.write(
            LAYER_NAMES.get(
                res["layer"],
                res["layer"],
            )
        )

    with kolom2:

        if res["p_hoaks"] is not None:

            peluang = 100 * res["p_hoaks"]

            st.metric(
                "Estimasi Probabilitas HOAKS",
                f"{format_angka(peluang, 1)}%",
            )

        else:

            st.markdown("**Probabilitas Model**")

            st.write(
                "Tidak dihitung karena keputusan "
                "ditentukan sebelum tahap model."
            )

    st.markdown("**Alasan Hasil**")

    st.write(res["reason"])

    # Berita pembanding
    similar = res.get("similar")

    if similar:

        st.markdown(
            "**Berita Pembanding dari Indeks Hoaks**"
        )

        judul = similar.get("title") or "Berita pembanding"
        url = similar.get("url")

        if url:
            st.markdown(f"[{judul}]({url})")
        else:
            st.write(judul)

        st.caption(
            "Skor kemiripan: "
            f"{format_angka(similar['score'], 3)}"
        )

        if res["layer"] != "similar":

            st.caption(
                "Berita pembanding hanya sebagai "
                "referensi dan tidak menentukan "
                "keputusan model."
            )

    st.info(
        "Hasil analisis merupakan prediksi sistem, "
        "bukan bukti kebenaran atau pemeriksaan fakta "
        "independen. Periksa sumber tepercaya sebelum "
        "menyebarkan informasi.",
        icon="ℹ️",
    )


# =========================================================
# FUNGSI PENJELASAN MODEL
# =========================================================

def tampilkan_penjelasan(agent, clean):

    st.subheader("Penjelasan Model")

    st.caption(
        "Penjelasan memperlihatkan kontribusi fitur "
        "terhadap keputusan model, bukan bukti "
        "kebenaran suatu klaim."
    )

    ex = svm_word_contributions(
        agent.model,
        clean,
        top_k=10,
    )

    kiri, kanan = st.columns(2)

    with kiri:

        st.markdown(
            "**Fitur yang Mendukung Skor HOAKS**"
        )

        df_hoaks = pd.DataFrame(
            ex["ke_hoaks"],
            columns=["Kata / N-gram", "Kontribusi"],
        )

        st.dataframe(
            df_hoaks,
            hide_index=True,
            use_container_width=True,
        )

    with kanan:

        st.markdown(
            "**Fitur yang Mendukung Skor VALID**"
        )

        df_valid = pd.DataFrame(
            ex["ke_valid"],
            columns=["Kata / N-gram", "Kontribusi"],
        )

        st.dataframe(
            df_valid,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Detail Teknis Penjelasan"):

        st.write(
            "Skor SVM:",
            format_angka(ex["skor_svm"], 3),
        )

        st.write(
            "Jumlah N-gram dikenali:",
            ex["n_kata_dikenal"],
        )

        st.caption(
            "Kontribusi SVM dihitung dari "
            "TF-IDF × bobot model."
        )

    st.markdown("**Penjelasan LIME**")

    with st.spinner("Menghitung penjelasan LIME..."):

        try:

            lime = lime_explanation(
                agent.model,
                clean,
                num_features=10,
                num_samples=1000,
            )

            df_lime = pd.DataFrame(
                lime,
                columns=["Kata / N-gram", "Bobot LIME"],
            )

            st.dataframe(
                df_lime,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Bobot positif meningkatkan "
                "kecenderungan model ke kelas HOAKS."
            )

        except ImportError as e:

            st.warning(
                f"Penjelasan LIME belum tersedia: {e}"
            )


# =========================================================
# FUNGSI MEMBACA CSV
# =========================================================

def baca_csv(raw):

    if not raw or not raw.strip():

        raise ValueError(
            "File CSV kosong. Silakan unggah file "
            "yang memiliki kolom judul atau isi berita."
        )

    for encoding in ["utf-8-sig", "latin-1"]:

        try:

            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
            )

        except UnicodeDecodeError:
            continue

        except pd.errors.EmptyDataError:

            raise ValueError(
                "File CSV kosong atau tidak "
                "memiliki kolom yang dapat dibaca."
            )

        except pd.errors.ParserError:

            raise ValueError(
                "Format CSV tidak valid. "
                "Periksa pemisah kolom dan tanda kutip."
            )

    raise ValueError(
        "CSV tidak dapat dibaca. "
        "Coba simpan ulang sebagai UTF-8."
    )


# =========================================================
# TAB APLIKASI
# =========================================================

tab1, tab2 = st.tabs(
    [
        "Periksa Satu Berita",
        "Analisis Banyak Berita (CSV)",
    ]
)


# =========================================================
# TAB 1 — ANALISIS SATU BERITA
# =========================================================

with tab1:

    st.subheader("Masukkan Berita")

    st.write(
        "Isi judul dan teks berita yang ingin "
        "dianalisis oleh sistem."
    )

    with st.form(
        "form_analisis",
        clear_on_submit=False,
    ):

        title = st.text_input(
            "Judul Berita",
            placeholder="Masukkan judul berita...",
        )

        body = st.text_area(
            "Isi Berita",
            height=200,
            placeholder=(
                "Tempelkan isi berita yang "
                "ingin dianalisis di sini..."
            ),
        )

        tombol_analisis = st.form_submit_button(
            "Analisis Berita",
            type="primary",
            use_container_width=True,
        )

    if tombol_analisis:

        try:

            with st.spinner(
                "Sistem sedang menganalisis berita..."
            ):

                res = agent.predict(
                    title,
                    body,
                    include_timing=True,
                )

            st.session_state["hasil_analisis"] = res
            st.session_state["tampilkan_lime"] = False

        except Exception as e:

            st.session_state.pop(
                "hasil_analisis",
                None,
            )

            st.error(
                f"Analisis gagal dijalankan: {e}"
            )

    res = st.session_state.get(
        "hasil_analisis"
    )

    if res is not None:

        st.divider()

        tampilkan_hasil(res)

        waktu = res.get(
            "timing_ms",
            {},
        ).get("total")

        if waktu is not None:

            st.caption(
                "Waktu pemrosesan agen: "
                f"{format_angka(waktu, 0)} ms"
            )

        if (
            res["layer"] != "scope_guard"
            and res.get("clean_text")
        ):

            if st.button(
                "Lihat Penjelasan Model",
                key="tombol_jelaskan",
            ):

                st.session_state["tampilkan_lime"] = (
                    not st.session_state.get(
                        "tampilkan_lime",
                        False,
                    )
                )

            if st.session_state.get(
                "tampilkan_lime",
                False,
            ):

                tampilkan_penjelasan(
                    agent,
                    res["clean_text"],
                )


# =========================================================
# TAB 2 — ANALISIS BANYAK BERITA
# =========================================================

with tab2:

    st.subheader(
        "Analisis Banyak Berita"
    )

    st.write(
        "Unggah file CSV untuk menganalisis "
        "beberapa berita secara bersamaan."
    )

    st.info(
        "CSV harus memiliki kolom **title/judul** "
        "dan/atau **body/isi/text**. "
        "Maksimal 1.000 berita per unggahan.",
        icon="ℹ️",
    )

    with st.expander(
        "Lihat Contoh Format CSV"
    ):

        st.code(
            'title,body\n'
            '"Contoh judul berita",'
            '"Contoh isi berita yang ingin dianalisis"',
            language="csv",
        )

    uploaded = st.file_uploader(
        "Pilih File CSV",
        type=["csv"],
    )

    if uploaded is not None:

        try:

            raw = uploaded.getvalue()

            df = baca_csv(raw)

            if df.empty:

                raise ValueError(
                    "File CSV belum memiliki baris "
                    "berita. Tambahkan data dahulu."
                )

            with st.spinner(
                f"Menganalisis {len(df)} berita..."
            ):

                results = process(agent, df)

            q = queue(results)

        except (
            ValueError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as e:

            st.error(str(e))

        except Exception as e:

            st.error(
                f"File CSV gagal diproses: {e}"
            )

        else:

            st.success(
                f"Berhasil menganalisis "
                f"{len(results)} berita."
            )

            # RINGKASAN HASIL
            kol1, kol2, kol3 = st.columns(3)

            with kol1:

                st.metric(
                    "Antrean Pemeriksaan",
                    len(q["antrean"]),
                )

            with kol2:

                st.metric(
                    "Cocok Hoaks Lama",
                    len(q["cocok_hoaks_lama"]),
                )

            with kol3:

                st.metric(
                    "Di Luar Cakupan",
                    len(q["di_luar_cakupan"]),
                )

            st.divider()

            # ANTREAN PEMERIKSAAN
            st.subheader(
                "Antrean Pemeriksaan"
            )

            st.caption(
                "Berita yang diproses model "
                "diurutkan berdasarkan estimasi "
                "probabilitas HOAKS tertinggi."
            )

            st.dataframe(
                q["antrean"],
                hide_index=True,
                use_container_width=True,
            )

            # HOAKS LAMA
            st.subheader(
                "Berita yang Cocok "
                "dengan Hoaks Lama"
            )

            st.dataframe(
                q["cocok_hoaks_lama"],
                hide_index=True,
                use_container_width=True,
            )

            # DI LUAR CAKUPAN
            st.subheader(
                "Berita di Luar Cakupan"
            )

            st.dataframe(
                q["di_luar_cakupan"],
                hide_index=True,
                use_container_width=True,
            )

            st.divider()

            # UNDUH HASIL
            csv_output = (
                results.to_csv(
                    index=False,
                ).encode("utf-8-sig")
            )

            st.download_button(
                label="Unduh Hasil Analisis (CSV)",
                data=csv_output,
                file_name="hasil_deteksi.csv",
                mime="text/csv",
                type="primary",
            )
