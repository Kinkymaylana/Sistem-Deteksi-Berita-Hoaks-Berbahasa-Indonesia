"""Aplikasi prototipe deteksi berita hoaks (Lapisan Aktuator, pemilik: Anggota C).

Jalankan dari folder PROTOTIPE_KEL5:
    streamlit run app/app.py

Seluruh logika keputusan ada di src/ (agent.py, reactive.py, explain.py,
batch.py) dan diuji di tests/. File ini hanya menyusun tampilan sesuai
Desain Metode 2.8 (isi keluaran dan urutan antrean).
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

from src.agent import MODEL_FILE, MODELS_DIR, load_default_agent  # noqa: E402
from src.batch import process, queue  # noqa: E402
from src.config import HOAKS, PERLU_VERIFIKASI, VALID, thresholds_from_costs  # noqa: E402
from src.explain import lime_explanation, svm_word_contributions  # noqa: E402

COLORS = {HOAKS: "#C62828", VALID: "#2E7D32", PERLU_VERIFIKASI: "#EF8F00"}
LAYER_NAMES = {"scope_guard": "Penjaga cakupan (reaktif)",
               "similar": "Pencarian hoaks serupa (reaktif)",
               "model": "Linear SVM terkalibrasi + minimum expected loss (deliberatif)"}

st.set_page_config(page_title="Deteksi Hoaks - Kelompok 5", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_agent():
    return load_default_agent()


def badge(action: str) -> str:
    return (f"<div style='display:inline-block;padding:0.35rem 0.9rem;border-radius:0.4rem;"
            f"background:{COLORS[action]};color:white;font-weight:700;font-size:1.3rem'>{action}</div>")


def show_result(res: dict) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(res["action"]), unsafe_allow_html=True)
        if res["p_hoaks"] is not None:
            st.metric("Peluang HOAKS (terkalibrasi)", f"{100 * res['p_hoaks']:.1f}%")
        else:
            st.caption("Peluang tidak dihitung karena keputusan diambil lapisan reaktif.")
    with c2:
        st.markdown(f"**Alasan:** {res['reason']}")
        st.markdown(f"**Lapisan penentu:** {LAYER_NAMES.get(res['layer'], res['layer'])}")
        sim = res.get("similar")
        if sim:
            label = ("Hoaks lama yang cocok" if res["layer"] == "similar"
                     else "Hoaks lama terdekat (pembanding, tidak memengaruhi keputusan)")
            skor = f"{sim['score']:.3f}".replace(".", ",")
            st.markdown(f"**{label}:** [{sim['title']}]({sim['url']}) — skor kemiripan {skor}")


def show_explanation(agent, clean: str) -> None:
    st.subheader("Penjelasan")
    ex = svm_word_contributions(agent.model, clean, top_k=10)
    a, b = st.columns(2)
    with a:
        st.markdown("**Kata yang mendorong ke HOAKS** (kontribusi SVM)")
        st.dataframe(pd.DataFrame(ex["ke_hoaks"], columns=["kata", "kontribusi"]), hide_index=True)
    with b:
        st.markdown("**Kata yang mendorong ke VALID** (kontribusi SVM)")
        st.dataframe(pd.DataFrame(ex["ke_valid"], columns=["kata", "kontribusi"]), hide_index=True)
    st.caption(f"Kontribusi = nilai TF-IDF × bobot SVM. Skor SVM = {ex['skor_svm']:.3f}; "
               f"{ex['n_kata_dikenal']} n-gram dikenali model.")
    try:
        with st.spinner("Menghitung LIME (beberapa detik)..."):
            lime = lime_explanation(agent.model, clean, num_features=10, num_samples=1000)
        st.markdown("**LIME** (positif = mendorong ke HOAKS)")
        st.dataframe(pd.DataFrame(lime, columns=["kata", "bobot LIME"]), hide_index=True)
    except ImportError as e:
        st.warning(str(e))


# ---------------------------------------------------------------------------
st.title("Deteksi Berita Hoaks Berbahasa Indonesia")
st.caption("Prototipe Tahap 3 Kelompok 5. Penyaring awal untuk pemeriksa fakta; "
           "keputusan akhir tetap di tangan manusia. Sistem tidak memverifikasi ke sumber eksternal.")

try:
    agent = get_agent()
except FileNotFoundError:
    st.error(f"Model belum tersedia. Letakkan `{MODEL_FILE}` dan model card dari Anggota B "
             f"di folder `{MODELS_DIR}`, lalu muat ulang halaman.")
    st.stop()

with st.sidebar:
    st.header("Status lapisan agen")
    stt = agent.status()
    for key, name in [("preprocess", "Persepsi: clean_text"), ("scope_guard_panjang", "Penjaga cakupan: panjang"),
                      ("scope_guard_bahasa", "Penjaga cakupan: bahasa"), ("find_similar", "Hoaks serupa"),
                      ("model", "Model SVM terkalibrasi")]:
        st.write(("✅ " if stt[key] else "⛔ ") + name)
    t_valid, t_hoaks = thresholds_from_costs(agent.config.c_fp, agent.config.c_fn, agent.config.c_r)
    st.write(f"Ambang: VALID ≤ {t_valid:.2f}, HOAKS ≥ {t_hoaks:.2f}".replace(".", ","))
    st.write(f"τ_mirip = {agent.config.tau_similar:.2f}".replace(".", ","))
    st.write(f"Preprocessing: {stt.get('preprocess_version') or '-'}")
    if getattr(agent, "model_path", None):
        st.caption(f"Model: {agent.model_path.name} ({agent.model_card.get('data_version', 'versi data tidak diketahui')})")
    for w in agent.warnings:
        st.warning(w)

tab1, tab2 = st.tabs(["Periksa satu berita", "Unggah CSV (banyak berita)"])

with tab1:
    title = st.text_input("Judul berita")
    body = st.text_area("Isi berita", height=220)
    if st.button("Periksa", type="primary"):
        st.session_state["hasil"] = agent.predict(title, body, include_timing=True)
        st.session_state["jelaskan"] = False
    res = st.session_state.get("hasil")
    if res:
        show_result(res)
        st.caption(f"Waktu respons: {res['timing_ms'].get('total', 0):.0f} ms")
        if res["layer"] != "scope_guard":
            if st.button("Jelaskan"):
                st.session_state["jelaskan"] = True
            if st.session_state.get("jelaskan"):
                show_explanation(agent, res["clean_text"])

with tab2:
    st.markdown("CSV berisi kolom **title/judul** dan **body/isi/text**. Berita yang dinilai model "
                "diurutkan dari peluang hoaks tertinggi (Desain Metode 2.8.2).")
    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        raw = up.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        try:
            with st.spinner(f"Memproses {len(df)} berita..."):
                results = process(agent, df)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        q = queue(results)
        st.subheader(f"Antrean pemeriksaan ({len(q['antrean'])} berita)")
        st.dataframe(q["antrean"], hide_index=True)
        st.subheader(f"Cocok dengan hoaks lama ({len(q['cocok_hoaks_lama'])} berita)")
        st.dataframe(q["cocok_hoaks_lama"], hide_index=True)
        st.subheader(f"Di luar cakupan, tidak masuk antrean ({len(q['di_luar_cakupan'])} berita)")
        st.dataframe(q["di_luar_cakupan"], hide_index=True)
        st.download_button("Unduh hasil (CSV)", results.to_csv(index=False).encode("utf-8-sig"),
                           file_name="hasil_deteksi.csv", mime="text/csv")
