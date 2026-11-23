"""D1: Verifikasi src/evaluate.py terhadap hasil resmi Anggota B.

Menjalankan B2_run_experiments.py dan B3_calibrate.py APA ADANYA (tidak
diubah) dengan modul prototipe/anggota_c/src/evaluate.py, lalu
membandingkan setiap angka dengan file di results/anggota_b/.

Hanya memakai LATIH dan VALIDASI (skrip B tidak membaca UJI).

Contoh (dari root repo):
    python scripts/anggota_d/d1_verifikasi_evaluate.py \
        --dataset /path/DATASET_V3_1_KANDIDAT_OPERASIONAL.csv \
        --outdir HASIL_D1

Keluaran di --outdir:
    D1_VERIFIKASI_EVALUATE.csv   satu baris per (file, baris, kolom) yang dibandingkan
    D1_RINGKASAN.json            jumlah cocok/beda per file, SHA-256 dataset, versi
    b2/, b3/                     keluaran mentah skrip B hasil jalan ulang
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROTO = REPO / "prototipe" / "anggota_c"
B_SCRIPTS = REPO / "notebooks" / "anggota_b" / "scripts"
B_RESULTS = REPO / "results" / "anggota_b"

SHA_A = "a790c0a6175455f385529f98a48dbd5ee8d939805494168ad698bd56343bb295"
DATA_VERSION = "V3.1_A2_PROVISIONAL"
TEXT_COL = "text_clean_v31"
TOL = 1e-6  # angka B dibulatkan 6 desimal


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(script: str, args: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROTO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(B_SCRIPTS / script), *args]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return abs(float(a) - float(b)) <= TOL + 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)


def compare(name: str, ref: pd.DataFrame, new: pd.DataFrame, key: str | None,
            skip: set[str]) -> list[dict]:
    rows = []
    if key:
        ref = ref.set_index(key)
        new = new.set_index(key)
        keys = [k for k in ref.index if k in new.index]
    else:
        keys = list(range(min(len(ref), len(new))))
    for k in keys:
        r = ref.loc[k] if key else ref.iloc[k]
        n = new.loc[k] if key else new.iloc[k]
        for col in ref.columns:
            if col in skip or col not in new.columns:
                continue
            rows.append({"file": name, "baris": k, "kolom": col,
                         "nilai_b": r[col], "nilai_ulang": n[col], "cocok": same(r[col], n[col])})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("HASIL_D1"), type=Path)
    ap.add_argument("--skip-b2", action="store_true", help="lewati B2 (sekitar 2-4 menit)")
    a = ap.parse_args()

    out = a.outdir.resolve()
    (out / "b2").mkdir(parents=True, exist_ok=True)
    (out / "b3").mkdir(parents=True, exist_ok=True)
    sha = sha256(a.dataset)
    print(f"SHA-256 dataset: {sha}\nSama dengan file A: {sha == SHA_A}")

    common = ["--data", str(a.dataset), "--text-col", TEXT_COL, "--data-version", DATA_VERSION]
    b2, b3 = out / "b2", out / "b3"
    if not a.skip_b2:
        run("B2_run_experiments.py", common + [
            "--experiments-log", str(b2 / "experiments.csv"),
            "--e1-out", str(b2 / "e1.csv"), "--e2-out", str(b2 / "e2.csv"),
            "--title-out", str(b2 / "title.csv"), "--e4-out", str(b2 / "e4.csv")])
    run("B3_calibrate.py", common + [
        "--model-in", str(b3 / "LinearSVM.joblib"), "--model-out", str(b3 / "LinearSVM_calibrated.joblib"),
        "--model-card", str(b3 / "model_card.json"), "--experiments-log", str(b3 / "experiments.csv"),
        "--rel-out", str(b3 / "reliability.csv"), "--risk-out", str(b3 / "risk.csv")])

    rd = lambda p: pd.read_csv(p, encoding="utf-8-sig")  # noqa: E731
    rows: list[dict] = []
    ref_exp = rd(B_RESULTS / "B2_EXPERIMENTS_BASELINE.csv")
    if not a.skip_b2:
        rows += compare("B2_EXPERIMENTS_BASELINE", ref_exp, rd(b2 / "experiments.csv"), "run_id", {"tanggal"})
        for ref, new in [("B2_E1_SHORTCUT_ANALYSIS", "e1"), ("B2_E2_TEMPORAL_ANALYSIS", "e2"),
                         ("B2_TITLE_ONLY_ANALYSIS", "title"), ("B2_E4_LENGTH_CONTROL_ANALYSIS", "e4")]:
            rows += compare(ref, rd(B_RESULTS / f"{ref}.csv"), rd(b2 / f"{new}.csv"), None, set())
    rows += compare("B2_EXPERIMENTS_BASELINE(kalibrasi)", ref_exp, rd(b3 / "experiments.csv"), "run_id", {"tanggal"})
    rows += compare("B3_SELECTIVE_RISK_COVERAGE_SVM", rd(B_RESULTS / "B3_SELECTIVE_RISK_COVERAGE_SVM.csv"),
                    rd(b3 / "risk.csv"), None, set())
    rows += compare("B3_RELIABILITY_BINS_SVM", rd(B_RESULTS / "B3_RELIABILITY_BINS_SVM.csv"),
                    rd(b3 / "reliability.csv"), None, set())

    det = pd.DataFrame(rows)
    det.to_csv(out / "D1_VERIFIKASI_EVALUATE.csv", index=False)
    per_file = det.groupby("file")["cocok"].agg(dibandingkan="size", cocok="sum").reset_index()
    per_file["beda"] = per_file["dibandingkan"] - per_file["cocok"]
    import sklearn
    summary = {
        "dataset_sha256": sha, "sama_dengan_file_a": sha == SHA_A, "data_version": DATA_VERSION,
        "toleransi": TOL, "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "per_file": per_file.to_dict(orient="records"),
        "total_dibandingkan": int(len(det)), "total_beda": int((~det["cocok"]).sum()),
    }
    (out / "D1_RINGKASAN.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(per_file.to_string(index=False))
    beda = det.loc[~det["cocok"]]
    if len(beda):
        print("\nAngka yang berbeda:")
        print(beda.to_string(index=False))
    print(f"\nTotal beda: {summary['total_beda']} dari {summary['total_dibandingkan']} angka")


if __name__ == "__main__":
    main()
