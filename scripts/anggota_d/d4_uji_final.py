"""D4: Evaluasi final di UJI, satu kali per run (Anggota D).

Semua konfigurasi (kolom teks, model, hyperparameter, split) dibaca dari
manifest pembekuan results/anggota_d/PEMBEKUAN_UJI.json. Skrip ini tidak
punya pengaturan model sendiri, jadi tidak bisa "disetel" setelah melihat UJI.

Setiap run dibuka lewat src/uji_lock.py:
  - wajib --final, manifest DIBEKUKAN, SHA-256 dataset dan berkas cocok;
  - run yang sudah SELESAI ditolak kecuali --alasan-ulang;
  - setiap upaya tercatat di results/anggota_d/uji_log.csv.

Jenis run:
  baseline : TF-IDF + ComplementNB/LinearSVM dilatih ulang pada LATIH split
             yang sama (seed 42), lalu dinilai di UJI.
  agen     : agen utuh milik C (model jadi di prototipe/anggota_c/models)
             dijalankan pada teks mentah UJI, sama seperti D3 di VALIDASI.

Contoh (dari root repo):
    # lihat apa yang akan dijalankan, tanpa membuka UJI
    python scripts/anggota_d/d4_uji_final.py --dataset <CSV> --rencana

    # jalankan semua run yang belum SELESAI
    python scripts/anggota_d/d4_uji_final.py --dataset <CSV> --final --operator "Anggota D"

    # satu run saja
    python scripts/anggota_d/d4_uji_final.py --dataset <CSV> --final --run UJI_LinearSVM_random_clean

Keluaran di --outdir (bawaan results/anggota_d):
    experiments_uji.csv                 log eksperimen status UJI (format evaluate.py)
    D4_UJI_RINGKASAN.csv                satu baris per run baseline/agen
    D4_UJI_AGEN_SISTEM.csv              agen vs model saja, C_FN 3/5/10 (seperti D3)
    D4_UJI_AGEN_LAPISAN.csv             kontribusi lapisan agen di UJI
    D4_UJI_SIGNIFIKANSI.csv             E1 di UJI: mentah vs bersih (bootstrap berpasangan)
    D4_PREDIKSI_UJI_<run_key>.csv       prediksi per berita
    D4_METADATA.json                    manifest, SHA-256, versi, waktu
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROTO = REPO / "prototipe" / "anggota_c"
sys.path.insert(0, str(PROTO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import calculate_metrics, confusion_matrix_dict, paired_bootstrap_test  # noqa: E402
from src.uji_lock import (  # noqa: E402
    LOG_SELESAI, STATUS_DIBEKUKAN, UjiTerkunci, add_final_args, completed_runs,
    load_manifest, manifest_sha256, open_uji,
)

SEED = 42


# ---------------------------------------------------------------------------
def build_pipeline(run: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    tf = dict(run.get("tfidf") or {})
    if "ngram_range" in tf:
        tf["ngram_range"] = tuple(tf["ngram_range"])
    hp = dict(run.get("hyperparameter") or {})
    if run["model"] == "ComplementNB":
        clf = ComplementNB(alpha=float(hp["alpha"]))
    elif run["model"] == "LinearSVM":
        clf = LinearSVC(C=float(hp["C"]), class_weight=hp.get("class_weight", "balanced"),
                        max_iter=int(hp.get("max_iter", 3000)), random_state=int(hp.get("random_state", SEED)))
    else:
        raise ValueError(f"Model baseline tidak dikenal: {run['model']}")
    return Pipeline([("tfidf", TfidfVectorizer(**tf)), ("classifier", clf)])


def is_agent_run(run: dict) -> bool:
    return run["run_key"].startswith("UJI_Agen")


def run_baseline(sesi, run: dict, outdir: Path, exp_log: Path) -> dict:
    col = run["text_col"]
    latih = sesi.load_latih()
    uji = sesi.load()
    X_tr = latih[col].fillna("").astype(str)
    X_te = uji[col].fillna("").astype(str)
    y_tr = latih["label"].astype(int).to_numpy()
    y_te = uji["label"].astype(int).to_numpy()

    t0 = time.time()
    pipe = build_pipeline(run).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    clf = pipe.named_steps["classifier"]
    if hasattr(clf, "predict_proba"):
        prob, jenis_p = pipe.predict_proba(X_te)[:, 1], "predict_proba"
    else:
        prob, jenis_p = 1.0 / (1.0 + np.exp(-pipe.decision_function(X_te))), "sigmoid(decision_function), tidak terkalibrasi"
    m = calculate_metrics(y_te, pred, y_prob=prob)
    sesi.log_experiment(exp_log, m)

    pd.DataFrame({"raw_id": uji["raw_id"].to_numpy(), "source": uji["source"].to_numpy(), "label": y_te,
                  "pred": pred, "p_hoaks": prob}).to_csv(outdir / f"D4_PREDIKSI_UJI_{run['run_key']}.csv", index=False)
    return {"run_key": run["run_key"], "jenis": "baseline", "split": run["split"], "text_col": col,
            "model": run["model"], "hyperparameter": json.dumps(run.get("hyperparameter", {})),
            "n_latih": len(latih), "n_uji": len(uji), "n_hoaks_uji": int(y_te.sum()),
            "jenis_peluang": jenis_p, "durasi_detik": round(time.time() - t0, 1),
            **{k: m[k] for k in ("macro_f1", "macro_f1_ci_95", "accuracy", "f1_hoaks", "precision_hoaks",
                                 "recall_hoaks", "f1_valid", "ece", "brier_score", "precision_at_50",
                                 "precision_at_100", "tn", "fp", "fn", "tp")}}


def run_agent(sesi, run: dict, outdir: Path, exp_log: Path) -> dict:
    import d3_evaluasi_agen as d3
    from src.agent import load_default_agent
    from src.config import LAYER_MODEL

    uji = sesi.load().reset_index(drop=True)
    agent = load_default_agent(PROTO / "models")
    hp = run.get("hyperparameter", {})
    if abs(agent.config.tau_similar - float(hp.get("tau_similar", agent.config.tau_similar))) > 1e-12:
        raise RuntimeError("tau_similar agen berbeda dari manifest")
    t0 = time.time()
    pred = d3.run_agent(agent, uji)
    y = pred["label"].to_numpy()
    p_model = pred["p_model"].astype(float).to_numpy()
    p_model_filled = np.nan_to_num(p_model, nan=0.5)

    rows = []
    for c_fn in d3.C_FNS:
        t_cost = d3.C_FP / (d3.C_FP + c_fn)
        systems = {
            "model_ambang_0.5": (p_model_filled >= 0.5).astype(int),
            "model_ambang_biaya": (p_model_filled >= t_cost).astype(int),
            "model_selektif": d3.model_decisions(p_model_filled, c_fn),
            "agen": d3.agent_decisions(pred, c_fn),
        }
        for name, dec in systems.items():
            s = d3.summarize(y, dec, c_fn)
            s.update(d3.bootstrap_ci(y, dec, c_fn))
            rows.append({"c_fn": c_fn, "sistem": name, **s})
    sistem = pd.DataFrame(rows).round(6)
    sistem.to_csv(outdir / "D4_UJI_AGEN_SISTEM.csv", index=False)

    dec5 = d3.agent_decisions(pred, d3.MAIN_C_FN)
    lay = []
    for layer, g in pred.assign(_d=dec5).groupby("lapisan"):
        dm, yy = g["_d"].to_numpy(), g["label"].to_numpy()
        nh = int((dm == 1).sum())
        tp = int(((dm == 1) & (yy == 1)).sum())
        row = {"lapisan": layer, "n": len(g), "n_label_hoaks": int(yy.sum()), "n_diputuskan_hoaks": nh,
               "n_diputuskan_valid": int((dm == 0).sum()), "n_perlu_verifikasi": int((dm == -1).sum()),
               "precision_hoaks": tp / nh if nh else None}
        if nh:
            lo, hi = d3.clopper_pearson(tp, nh)
            row["precision_hoaks_ci95_clopper_pearson"] = f"[{lo:.4f}, {hi:.4f}]"
        lay.append(row)
    pd.DataFrame(lay).round(6).to_csv(outdir / "D4_UJI_AGEN_LAPISAN.csv", index=False)

    # Baris log: metrik klasifikasi dari model di dalam agen (p >= 0,5 pada teks
    # bersih agen), kolom agen (coverage, macro-F1 selektif, kerugian) dari agen utuh.
    m = calculate_metrics(y, (p_model_filled >= 0.5).astype(int), y_prob=p_model_filled)
    s5 = d3.summarize(y, dec5, d3.MAIN_C_FN)
    m["coverage_agen"] = round(float(s5["coverage"]), 6)
    m["selective_macro_f1"] = round(float(s5["macro_f1_selektif"]), 6)
    m["expected_loss"] = round(float(s5["kerugian_harapan"]), 6)
    sesi.log_experiment(exp_log, m)

    pred.assign(keputusan_agen_cfn5=dec5).drop(columns=["teks_bersih_agen"]) \
        .to_csv(outdir / f"D4_PREDIKSI_UJI_{run['run_key']}.csv", index=False)
    lat = pred["total_ms"].astype(float)
    return {"run_key": run["run_key"], "jenis": "agen", "split": run["split"], "text_col": "title + text_raw",
            "model": run["model"], "hyperparameter": json.dumps(hp), "n_latih": None, "n_uji": len(uji),
            "n_hoaks_uji": int(y.sum()), "jenis_peluang": "Platt (B3)", "durasi_detik": round(time.time() - t0, 1),
            **{k: m[k] for k in ("macro_f1", "macro_f1_ci_95", "accuracy", "f1_hoaks", "precision_hoaks",
                                 "recall_hoaks", "f1_valid", "ece", "brier_score", "precision_at_50",
                                 "precision_at_100", "tn", "fp", "fn", "tp")},
            "coverage_agen": m["coverage_agen"], "selective_macro_f1": m["selective_macro_f1"],
            "expected_loss_agen": m["expected_loss"],
            "n_ditolak": int(s5["ditolak"]), "fp_agen": s5["fp"], "fn_agen": s5["fn"],
            "latensi_p50_ms": round(float(lat.quantile(.5)), 2), "latensi_p95_ms": round(float(lat.quantile(.95)), 2),
            "status_agen": json.dumps(agent.status())}


def significance(outdir: Path, runs: list[dict]) -> None:
    """E1 di UJI: mentah vs bersih per model, dari prediksi yang sudah tersimpan."""
    rows = []
    for model in ("ComplementNB", "LinearSVM"):
        f_raw = outdir / f"D4_PREDIKSI_UJI_UJI_{model}_random_raw.csv"
        f_cln = outdir / f"D4_PREDIKSI_UJI_UJI_{model}_random_clean.csv"
        if not (f_raw.exists() and f_cln.exists()):
            continue
        a, b = pd.read_csv(f_raw), pd.read_csv(f_cln)
        if not (a["raw_id"].to_numpy() == b["raw_id"].to_numpy()).all():
            raise RuntimeError("Urutan berita UJI mentah dan bersih berbeda")
        r = paired_bootstrap_test(a["label"], a["pred"], b["pred"], n_bootstraps=1000, seed=SEED)
        rows.append({"eksperimen": "E1_UJI", "model": model, "a": "mentah (text_raw)", "b": "bersih (text_clean_v31)",
                     **r, "p_value_laporan": "< 0,001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"})
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "D4_UJI_SIGNIFIKANSI.csv", index=False)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=REPO / "results" / "anggota_d")
    ap.add_argument("--run", action="append", default=None, help="run_key tertentu (boleh diulang)")
    ap.add_argument("--rencana", action="store_true", help="tampilkan rencana run tanpa membuka UJI")
    add_final_args(ap)
    a = ap.parse_args()

    if not Path(a.manifest).is_file():
        print(f"[TERKUNCI] Manifest {a.manifest} belum ada. Bekukan dulu dengan d2_bekukan.py.")
        return 2
    man = load_manifest(a.manifest)
    runs = [r for r in man["runs"] if a.run is None or r["run_key"] in a.run]
    if a.run:
        unknown = set(a.run) - {r["run_key"] for r in man["runs"]}
        if unknown:
            print(f"[DITOLAK] run_key tidak ada di manifest: {sorted(unknown)}")
            return 2
    done = completed_runs(a.uji_log)

    print(f"Manifest  : {a.manifest} ({man['status']}, sidik {manifest_sha256(man)[:16]}…)")
    print(f"Run       : {len(runs)}")
    for r in runs:
        print(f"  {'SELESAI ' if r['run_key'] in done else 'belum   '} {r['run_key']:34s} {r['split']:14s} "
              f"{r.get('text_col', ''):15s} {r['model']} {r.get('hyperparameter', {})}")
    if a.rencana:
        print("\n[RENCANA] UJI tidak dibuka.")
        return 0
    if man["status"] != STATUS_DIBEKUKAN:
        print("[TERKUNCI] Manifest belum DIBEKUKAN.")
        return 2

    a.outdir.mkdir(parents=True, exist_ok=True)
    exp_log = a.outdir / "experiments_uji.csv"
    ringkasan_path = a.outdir / "D4_UJI_RINGKASAN.csv"
    hasil, gagal = [], []
    for r in runs:
        if r["run_key"] in done and not a.alasan_ulang:
            print(f"\n[LEWATI] {r['run_key']} sudah SELESAI.")
            continue
        print(f"\n=== {r['run_key']} ===", flush=True)
        try:
            with open_uji(a.dataset, run_key=r["run_key"], split_col=r["split"], final=a.final,
                          manifest_path=a.manifest, log_path=a.uji_log, operator=a.operator,
                          alasan_ulang=a.alasan_ulang if r["run_key"] in done else None,
                          skrip=Path(__file__).name) as sesi:
                row = (run_agent if is_agent_run(r) else run_baseline)(sesi, r, a.outdir, exp_log)
            row["waktu"] = time.strftime("%Y-%m-%d %H:%M:%S")
            hasil.append(row)
            print(f"  macro-F1 {row['macro_f1']} {row['macro_f1_ci_95']}"
                  + (f" | coverage agen {row['coverage_agen']} | kerugian {row['expected_loss_agen']}"
                     if row["jenis"] == "agen" else ""))
        except UjiTerkunci as e:
            print(f"[TERKUNCI] {e}")
            return 2
        except Exception as e:  # dicatat GAGAL oleh UjiSession; lanjut ke run berikutnya
            print(f"[GAGAL] {r['run_key']}: {type(e).__name__}: {e}")
            gagal.append(r["run_key"])

    if hasil:
        new = pd.DataFrame(hasil)
        if ringkasan_path.exists():
            old = pd.read_csv(ringkasan_path)
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(ringkasan_path, index=False)
    significance(a.outdir, man["runs"])

    import sklearn
    meta = {
        "tugas": "D4 evaluasi final di UJI", "tanggal": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": Path(a.manifest).name, "manifest_sha256": manifest_sha256(man),
        "tanggal_beku": man.get("tanggal_beku"), "disetujui_oleh": man.get("disetujui_oleh"),
        "versi_data": man["versi_data"], "dataset_sha256": man["dataset_sha256"],
        "run_selesai": sorted(completed_runs(a.uji_log)), "run_gagal_sesi_ini": gagal,
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "catatan": ["Baseline dilatih ulang pada LATIH split masing-masing (seed 42) dengan konfigurasi manifest.",
                    "ECE dan Brier LinearSVM baseline memakai sigmoid skor SVM (tidak terkalibrasi), jadi tidak bermakna.",
                    "Model di dalam agen adalah LinearSVM_calibrated_v31 (Platt B3)."],
    }
    (a.outdir / "D4_METADATA.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSelesai: {len(hasil)} run, gagal: {gagal or '-'}. Hasil di {a.outdir}")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
