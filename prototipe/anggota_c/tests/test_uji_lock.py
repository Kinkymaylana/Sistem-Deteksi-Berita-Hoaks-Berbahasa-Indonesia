"""Uji src/uji_lock.py (pemilik: Anggota D, tugas D2).

Semua data di sini sintetis. Setiap tes membuat "repo mini" di folder
sementara: results/anggota_d/ untuk manifest dan log, satu berkas kode yang
dibekukan, dan satu CSV kecil berisi LATIH, VALIDASI, dan UJI.

    python -m pytest tests/test_uji_lock.py -v
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import calculate_metrics  # noqa: E402
from src.uji_lock import (  # noqa: E402
    UJI_LOG_COLUMNS, UjiTerkunci, add_final_args, completed_runs, manifest_sha256,
    open_uji, read_uji_log, sha256_file,
)

REPO = ROOT.parent.parent


@pytest.fixture
def mini(tmp_path):
    d = tmp_path / "results" / "anggota_d"
    d.mkdir(parents=True)
    code = tmp_path / "kode.py"
    code.write_text("X = 1\n", encoding="utf-8")
    data = tmp_path / "data.csv"
    with data.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_id", "label", "text", "random_split"])
        for i in range(30):
            split = "LATIH" if i < 18 else ("VALIDASI" if i < 24 else "UJI")
            w.writerow([i, i % 2, f"teks {i}", split])
    manifest = {
        "status": "DIBEKUKAN", "versi_data": "SINTETIS", "dataset_sha256": sha256_file(data),
        "seed": 42,
        "keputusan": [{"id": "K1", "status": "DISEPAKATI", "keputusan_akhir": "ok"}],
        "runs": [{"run_key": "UJI_A", "split": "random_split", "model": "M", "hyperparameter": {"C": 1}}],
        "berkas_dibekukan": [{"path": "kode.py", "sha256": sha256_file(code)}],
        "disetujui_oleh": ["A", "B"],
    }
    mpath = d / "PEMBEKUAN_UJI.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return {"root": tmp_path, "data": data, "manifest": mpath, "log": d / "uji_log.csv",
            "code": code, "exp": d / "experiments_uji.csv"}


def _open(m, **kw):
    args = dict(dataset_path=m["data"], run_key="UJI_A", split_col="random_split", final=True,
                manifest_path=m["manifest"], log_path=m["log"], operator="tester", skrip="tes")
    args.update(kw)
    return open_uji(**args)


def _statuses(m):
    return [r["status"] for r in read_uji_log(m["log"])]


def test_alur_normal_membuka_uji_sekali(mini):
    with _open(mini) as s:
        uji = s.load()
        assert len(uji) == 6 and set(uji["random_split"]) == {"UJI"}
        assert len(s.load_latih()) == 18
        assert len(s.load_latih(include_validasi=True)) == 24
        met = calculate_metrics(uji["label"], uji["label"], n_bootstraps=20)
        s.log_experiment(mini["exp"], met)
    assert _statuses(mini) == ["DIBUKA", "SELESAI"]
    log = read_uji_log(mini["log"])
    assert log[1]["n_baris_uji"] == "6"
    assert log[1]["dataset_sha256"] == sha256_file(mini["data"])
    assert json.loads(log[1]["ringkasan"])["macro_f1"] == 1.0
    with mini["exp"].open(encoding="utf-8") as fh:
        row = list(csv.DictReader(fh))[0]
    assert row["status_evaluasi"] == "UJI" and row["run_id"] == "UJI_A" and row["versi_data"] == "SINTETIS"
    assert completed_runs(mini["log"]) == {"UJI_A"}


def test_tanpa_final_ditolak(mini):
    with pytest.raises(UjiTerkunci, match="--final"):
        _open(mini, final=False)
    assert _statuses(mini) == ["DITOLAK"]


def test_tanpa_manifest_ditolak(mini):
    mini["manifest"].unlink()
    with pytest.raises(UjiTerkunci, match="tidak ditemukan"):
        _open(mini)


def test_manifest_draf_ditolak(mini):
    m = json.loads(mini["manifest"].read_text())
    m["status"] = "DRAF"
    mini["manifest"].write_text(json.dumps(m))
    with pytest.raises(UjiTerkunci, match="DRAF"):
        _open(mini)


def test_keputusan_belum_disepakati_ditolak(mini):
    m = json.loads(mini["manifest"].read_text())
    m["keputusan"][0]["status"] = "BELUM_DISEPAKATI"
    mini["manifest"].write_text(json.dumps(m))
    with pytest.raises(UjiTerkunci, match="K1"):
        _open(mini)


def test_run_tidak_terdaftar_ditolak(mini):
    with pytest.raises(UjiTerkunci, match="tidak terdaftar"):
        _open(mini, run_key="UJI_COBA_COBA")


def test_split_salah_ditolak(mini):
    with pytest.raises(UjiTerkunci, match="split"):
        _open(mini, split_col="temporal_main")


def test_dataset_berubah_ditolak(mini):
    with mini["data"].open("a", encoding="utf-8") as fh:
        fh.write("99,1,teks tambahan,UJI\n")
    with pytest.raises(UjiTerkunci, match="SHA-256 dataset"):
        _open(mini)


def test_kode_berubah_setelah_beku_ditolak(mini):
    mini["code"].write_text("X = 2  # diubah setelah melihat hasil\n", encoding="utf-8")
    with pytest.raises(UjiTerkunci, match="kode.py"):
        _open(mini)
    assert "kode.py" in read_uji_log(mini["log"])[-1]["alasan"]


def test_run_kedua_ditolak_kecuali_dengan_alasan(mini):
    with _open(mini) as s:
        s.load()
    with pytest.raises(UjiTerkunci, match="sudah SELESAI"):
        _open(mini)
    with _open(mini, alasan_ulang="Colab terputus sebelum zip hasil terunduh") as s:
        s.load()
    assert _statuses(mini) == ["DIBUKA", "SELESAI", "DITOLAK", "ULANG", "SELESAI"]
    assert "Colab terputus" in read_uji_log(mini["log"])[-1]["alasan"]


def test_error_di_tengah_dicatat_gagal_dan_bisa_diulang(mini):
    with pytest.raises(RuntimeError):
        with _open(mini) as s:
            s.load()
            raise RuntimeError("model gagal dimuat")
    assert _statuses(mini) == ["DIBUKA", "GAGAL"]
    assert "model gagal dimuat" in read_uji_log(mini["log"])[-1]["alasan"]
    # GAGAL tidak dihitung SELESAI, jadi run boleh dicoba lagi tanpa alasan
    with _open(mini) as s:
        s.load()
    assert _statuses(mini)[-1] == "SELESAI"


def test_header_log(mini):
    with _open(mini) as s:
        s.load()
    with mini["log"].open(encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == UJI_LOG_COLUMNS


def test_sidik_manifest_tidak_tergantung_urutan_kunci():
    assert manifest_sha256({"a": 1, "b": 2}) == manifest_sha256({"b": 2, "a": 1})


def test_add_final_args_bawaan_tertutup():
    import argparse
    a = add_final_args(argparse.ArgumentParser()).parse_args([])
    assert a.final is False
    assert a.manifest.endswith("PEMBEKUAN_UJI.json") and a.uji_log.endswith("uji_log.csv")


def test_draf_repo_tidak_bisa_dibekukan_selama_keputusan_belum_disepakati(mini):
    draf = REPO / "results" / "anggota_d" / "PEMBEKUAN_UJI_DRAF.json"
    script = REPO / "scripts" / "anggota_d" / "d2_bekukan.py"
    if not (draf.exists() and script.exists()):
        pytest.skip("draf/skrip D2 tidak ada di salinan ini")
    out = mini["root"] / "beku.json"
    r = subprocess.run([sys.executable, str(script), "--dataset", str(mini["data"]), "--draf", str(draf),
                        "--out", str(out), "--uji-log", str(mini["log"]), "--disetujui", "A"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "belum DISEPAKATI" in r.stdout
    assert not out.exists()


def test_bekukan_lalu_tidak_bisa_ditimpa_atau_dibatalkan_setelah_uji_dibuka(mini, tmp_path):
    script = REPO / "scripts" / "anggota_d" / "d2_bekukan.py"
    if not script.exists():
        pytest.skip("skrip D2 tidak ada di salinan ini")
    rel = "prototipe/anggota_c/src/evaluate.py"
    draf = {"status": "DRAF", "versi_data": "SINTETIS", "dataset_sha256": "", "seed": 42,
            "keputusan": [{"id": "K1", "status": "DISEPAKATI", "keputusan_akhir": "ya"}],
            "runs": [{"run_key": "UJI_A", "split": "random_split"}],
            "berkas_dibekukan": [rel], "disetujui_oleh": []}
    dp = tmp_path / "draf.json"
    dp.write_text(json.dumps(draf))
    out = tmp_path / "beku" / "PEMBEKUAN_UJI.json"
    log = tmp_path / "beku" / "uji_log.csv"
    base = [sys.executable, str(script), "--dataset", str(mini["data"]), "--draf", str(dp),
            "--out", str(out), "--uji-log", str(log)]
    r = subprocess.run(base + ["--disetujui", "A,B,C,D"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    m = json.loads(out.read_text())
    assert m["status"] == "DIBEKUKAN" and m["dataset_sha256"] == sha256_file(mini["data"])
    assert m["berkas_dibekukan"][0]["sha256"] == sha256_file(REPO / rel)
    # tidak bisa ditimpa
    assert subprocess.run(base + ["--disetujui", "A"], capture_output=True, text=True).returncode == 2
    # setelah UJI dibuka, tidak bisa dibatalkan
    with log.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=UJI_LOG_COLUMNS)
        w.writeheader()
        w.writerow({"run_key": "UJI_A", "status": "DIBUKA"})
    r = subprocess.run(base + ["--batalkan", "salah ketik"], capture_output=True, text=True)
    assert r.returncode == 2 and out.exists()
