"""Uji mekanis semua skrip Anggota C pada dataset dan model SINTETIS.

Tujuannya memastikan skrip berjalan dan keluarannya berformat benar. Keluaran
ditulis ke folder sementara pytest, bukan ke results/, dan angkanya tidak
boleh dilaporkan.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(*args):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


@pytest.fixture()
def models_dir(tmp_path, standin_model, dataset_csv):
    d = tmp_path / "models"
    d.mkdir()
    joblib.dump(standin_model, d / "LinearSVM_calibrated.joblib")
    import sklearn
    (d / "model_card.json").write_text(json.dumps({"data_version": "V3.1_SINTETIS", "environment": {"scikit_learn_version": sklearn.__version__}}))
    _run("scripts/build_language_rule.py", "--data", str(dataset_csv), "--text-col", "text_clean",
         "--top-n", "40", "--out", str(d / "language_rule.json"))
    out = _run("scripts/build_similar_index.py", "--data", str(dataset_csv), "--text-col", "text_clean",
               "--out-index", str(d / "similar_index.joblib"), "--out-table", str(tmp_path / "tau.csv"))
    assert "tau_mirip_terpilih" in out
    return d


def test_tabel_tau_hanya_validasi(models_dir, tmp_path):
    with open(tmp_path / "tau.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and {r["split"] for r in rows} == {"VALIDASI"}
    assert rows[0]["tau"] == "0.5" and rows[-1]["tau"] == "0.99"


def test_load_default_agent_merakit_semua_lapisan(models_dir):
    from src.agent import load_default_agent
    ag = load_default_agent(models_dir)
    st = ag.status()
    assert st["model"] and st["find_similar"] and st["scope_guard_bahasa"] and st["preprocess"]
    res = ag.predict("Pemerintah kota Malang meresmikan sekolah", "baru untuk warga pada Senin pagi")
    assert res["action"] in ("VALID", "HOAKS", "PERLU VERIFIKASI")


def test_uji_fungsional_menulis_10_skenario(models_dir, dataset_csv, tmp_path):
    out = tmp_path / "uji_fungsional.csv"
    _run("scripts/uji_fungsional.py", "--data", str(dataset_csv), "--models-dir", str(models_dir),
         "--out", str(out), "--lime-samples", "50")
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["no"]) for r in rows] == list(range(1, 11))
    by = {r["skenario"]: r for r in rows}
    assert by["Teks pendek"]["lulus"] == "True"
    assert by["Input kosong"]["lulus"] == "True"
    assert by["CSV 20 berita"]["lulus"] == "True"
    assert by["Hoaks beredar ulang"]["hasil_lapisan"] == "similar"


def test_ukur_latensi(models_dir, dataset_csv, tmp_path):
    out = tmp_path / "latency.csv"
    _run("scripts/ukur_latensi.py", "--data", str(dataset_csv), "--models-dir", str(models_dir),
         "--out", str(out), "--n", "10")
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["lapisan"] for r in rows] == ["preprocess", "scope_guard", "similar", "model", "total"]
    assert all(r["split"] == "VALIDASI" for r in rows)
    total = next(r for r in rows if r["lapisan"] == "total")
    assert int(total["n"]) == 10 and float(total["p95_ms"]) >= float(total["p50_ms"])


def test_uji_paritas_preprocess(dataset_csv, tmp_path):
    out = _run("scripts/uji_paritas_preprocess.py", "--data", str(dataset_csv), "--text-col", "text_clean",
               "--out", str(tmp_path / "paritas.csv"))
    data = json.loads(out)
    assert data["ringkasan"]["Kompas"]["sumber_diketahui"]["persen"] == 100.0
    assert data["ringkasan"]["Kompas"]["aplikasi"]["persen"] == 100.0


def test_skrip_menolak_membaca_uji(dataset_csv):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _common import read_rows
    with pytest.raises(ValueError, match="UJI"):
        read_rows(dataset_csv, ["raw_id"], splits=("UJI",))
    rows, counts = read_rows(dataset_csv, ["raw_id"])
    assert counts.get("UJI", 0) > 0 and all(r["random_split"] != "UJI" for r in rows)


# ---------------------------------------------------------------------------
# Uji asap app/app.py dengan modul streamlit TIRUAN.
# Ini hanya memeriksa logika Python di app.py (Periksa, Jelaskan, CSV),
# bukan tampilan. Tampilan harus dicek dengan `streamlit run app/app.py`.
# ---------------------------------------------------------------------------
def test_app_logika_dengan_streamlit_tiruan(models_dir, rows, monkeypatch):
    import functools
    import io
    import runpy
    import types

    import src.agent as agent_mod

    monkeypatch.setattr(agent_mod, "load_default_agent",
                        functools.partial(agent_mod.load_default_agent, models_dir))
    calls = []

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Stop(Exception):
        pass

    def rec(name, ret=None):
        def f(*a, **k):
            calls.append((name, a[:1]))
            return ret
        return f

    st = types.ModuleType("streamlit")
    for n in ["set_page_config", "title", "caption", "header", "subheader", "write", "markdown",
              "metric", "warning", "error", "info", "dataframe", "download_button"]:
        setattr(st, n, rec(n))
    st.stop = lambda: (_ for _ in ()).throw(Stop())
    st.cache_resource = lambda **k: (lambda fn: fn)
    st.spinner = lambda *a, **k: Ctx()
    st.sidebar = Ctx()
    st.columns = lambda spec: [Ctx() for _ in (spec if isinstance(spec, list) else range(spec))]
    st.tabs = lambda names: [Ctx() for _ in names]
    st.session_state = {}
    st.text_input = rec("text_input", "Viral pesan berantai sebarkan vaksin")
    st.text_area = rec("text_area", "jembatan mengandung chip magnet segera bagikan kepada warga Malang sekarang")
    st.button = rec("button", True)
    buf = io.StringIO()
    buf.write("judul,isi\n")
    for r in rows[:20]:
        buf.write(f"\"{r['title']}\",\"{r['text_raw']}\"\n")

    class Up:
        def getvalue(self):
            return buf.getvalue().encode("utf-8-sig")

    st.file_uploader = rec("file_uploader", Up())
    monkeypatch.setitem(sys.modules, "streamlit", st)

    runpy.run_path(str(ROOT / "app" / "app.py"), run_name="__main__")
    assert st.session_state["hasil"]["action"] in ("VALID", "HOAKS", "PERLU VERIFIKASI")
    assert st.session_state["jelaskan"] is True
    subs = [c[1][0] for c in calls if c[0] == "subheader"]
    assert "Penjelasan" in subs and any(s.startswith("Antrean pemeriksaan") for s in subs)
    assert any(c[0] == "download_button" for c in calls)
