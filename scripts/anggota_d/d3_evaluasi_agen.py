"""D3: Evaluasi agen utuh di VALIDASI (Anggota D).

Agen milik C (preprocessing -> scope_guard -> find_similar -> model + decide)
dijalankan pada setiap berita VALIDASI random_split, dengan masukan mentah
(judul + isi), persis seperti di aplikasi. UJI tidak dibaca.

Yang diukur:
  1. Kinerja agen utuh untuk C_FN = 3, 5, 10 (C_FP = 2, C_R = 1):
     coverage, galat pada berita yang diputuskan, macro-F1 selektif,
     FP/FN/ditolak, dan kerugian harapan, dengan CI bootstrap 95%.
  2. Pembanding pada data yang sama:
       - model_ambang_0.5      : model saja, tanpa penolakan, p >= 0,5
       - model_ambang_biaya    : model saja, tanpa penolakan, ambang biaya
                                 optimal C_FP/(C_FP+C_FN) (pembanding adil)
       - model_selektif        : model saja dengan penolakan (cara B3)
       - agen                  : semua lapisan
  3. Kontribusi tiap lapisan, termasuk precision aturan hoaks serupa dengan
     batas bawah Clopper-Pearson 95% (catatan C nomor 8).
  4. Kinerja per sumber berita.
  5. Kurva risk-coverage.
  6. Kalibrasi: ECE di VALIDASI apa adanya (seperti B3, optimis karena Platt
     dipasang di VALIDASI) dan ECE cross-fit 2 lipatan (jujur).
  7. Kesetaraan: keputusan agen (preprocessing sendiri) dibanding model pada
     kolom text_clean_v31 dataset.
  8. Waktu respons p50/p95.

Contoh (dari root repo):
    python scripts/anggota_d/d3_evaluasi_agen.py --dataset /path/DATASET_V3_1_KANDIDAT_OPERASIONAL.csv

Keluaran di --outdir (bawaan results/anggota_d):
    D3_PREDIKSI_AGEN_VALIDASI.csv  satu baris per berita
    D3_RINGKASAN_SISTEM.csv        sistem x C_FN
    D3_LAPISAN.csv                 kontribusi lapisan (C_FN = 5)
    D3_PER_SUMBER.csv              per sumber (C_FN = 5)
    D3_RISK_COVERAGE.csv/.png      kurva risk-coverage
    D3_KESALAHAN_AGEN.csv          berita yang diputuskan salah (C_FN = 5)
    D3_METADATA.json               versi, SHA-256, kalibrasi, kesetaraan, latensi
"""
from __future__ import annotations

import argparse
import hashlib
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

from src.agent import load_default_agent  # noqa: E402
from src.config import HOAKS, LAYER_MODEL, LAYER_SCOPE_GUARD, LAYER_SIMILAR, PERLU_VERIFIKASI, VALID  # noqa: E402
from src.evaluate import compute_ece, compute_reliability_bins  # noqa: E402

DATA_VERSION = "V3.1_A2_PROVISIONAL"
SHA_A = "a790c0a6175455f385529f98a48dbd5ee8d939805494168ad698bd56343bb295"
C_FP, C_R = 2.0, 1.0
C_FNS = (3.0, 5.0, 10.0)
MAIN_C_FN = 5.0
SEED = 42
N_BOOT = 1000
ACT_CODE = {VALID: 0, HOAKS: 1, PERLU_VERIFIKASI: -1}


# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def split_body(title, text_raw) -> str:
    """Sama dengan scripts/ukur_latensi.py milik C: buang judul di awal text_raw."""
    t, r = str(title or "").strip(), str(text_raw or "").strip()
    return r[len(t):].strip() if t and r.startswith(t) else r


def thresholds(c_fn: float) -> tuple[float, float]:
    return C_R / c_fn, 1.0 - C_R / C_FP


def macro_f1(y, d) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, d, average="macro", labels=[0, 1], zero_division=0)) if len(y) else float("nan")


def summarize(y: np.ndarray, dec: np.ndarray, c_fn: float) -> dict:
    """dec: 1 HOAKS, 0 VALID, -1 ditolak."""
    n = len(y)
    m = dec != -1
    yt, yd = y[m], dec[m]
    fp = int(((yd == 1) & (yt == 0)).sum())
    fn = int(((yd == 0) & (yt == 1)).sum())
    rej = int((~m).sum())
    return {
        "n": n, "coverage": m.mean(), "ditolak": rej,
        "galat_diputuskan": float((yt != yd).mean()) if m.any() else float("nan"),
        "macro_f1_selektif": macro_f1(yt, yd),
        "fp": fp, "fn": fn,
        "recall_hoaks_keseluruhan": float(((dec == 1) & (y == 1)).sum() / max(1, (y == 1).sum())),
        "kerugian_harapan": (C_FP * fp + c_fn * fn + C_R * rej) / n,
    }


def bootstrap_ci(y, dec, c_fn, keys=("coverage", "galat_diputuskan", "kerugian_harapan")) -> dict:
    rng = np.random.RandomState(SEED)
    n = len(y)
    vals = {k: [] for k in keys}
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        s = summarize(y[i], dec[i], c_fn)
        for k in keys:
            vals[k].append(s[k])
    return {f"{k}_ci95": f"[{np.nanpercentile(v, 2.5):.4f}, {np.nanpercentile(v, 97.5):.4f}]" for k, v in vals.items()}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def model_decisions(p: np.ndarray, c_fn: float) -> np.ndarray:
    lo, hi = thresholds(c_fn)
    d = np.full(len(p), -1)
    d[p <= lo] = 0
    d[p >= hi] = 1
    return d


def agent_decisions(df: pd.DataFrame, c_fn: float) -> np.ndarray:
    """Keputusan agen untuk C_FN tertentu.

    scope_guard dan similar tidak bergantung pada biaya, jadi tindakannya
    dipakai apa adanya. Berita yang sampai ke model diputuskan ulang dengan
    ambang dari C_FN.
    """
    d = df["aksi"].map(ACT_CODE).to_numpy().copy()
    m = (df["lapisan"] == LAYER_MODEL).to_numpy()
    d[m] = model_decisions(df.loc[m, "p_hoaks"].to_numpy(), c_fn)
    return d


def cross_fit_ece(scores: np.ndarray, y: np.ndarray) -> dict:
    """Platt dipasang di satu lipatan VALIDASI dan dinilai di lipatan lain (2 lipatan, ditukar)."""
    from sklearn.calibration import _sigmoid_calibration
    from sklearn.model_selection import StratifiedKFold
    p = np.empty(len(y))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=SEED).split(scores, y):
        a, b = _sigmoid_calibration(scores[tr], y[tr])
        p[te] = 1.0 / (1.0 + np.exp(a * scores[te] + b))
    return {"ece_crossfit": compute_ece(y, p), "p_crossfit": p}


# ---------------------------------------------------------------------------
def run_agent(agent, va: pd.DataFrame) -> pd.DataFrame:
    out = []
    first = va.iloc[0]
    agent.predict(first["title"], split_body(first["title"], first["text_raw"]))  # pemanasan
    for i, r in enumerate(va.itertuples(index=False), 1):
        title = "" if pd.isna(r.title) else str(r.title)
        body = split_body(title, "" if pd.isna(r.text_raw) else r.text_raw)
        res = agent.predict(title, body, include_timing=True)
        clean = res.get("clean_text")
        p_model = res["p_hoaks"]
        if p_model is None and clean:
            p_model = agent.hoax_probability(clean)  # p untuk analisis, tidak mengubah keputusan
        sim = res.get("similar") or {}
        out.append({
            "raw_id": r.raw_id, "source": r.source, "label": int(r.label),
            "aksi": res["action"], "lapisan": res["layer"], "alasan": res["reason"],
            "p_hoaks": res["p_hoaks"], "p_model": p_model,
            "skor_mirip": sim.get("score"), "judul_mirip": sim.get("title"),
            "teks_bersih_agen": clean or "", "total_ms": res["timing_ms"].get("total"),
        })
        if i % 500 == 0:
            print(f"  {i}/{len(va)} berita", flush=True)
    return pd.DataFrame(out)


def risk_coverage_curve(y: np.ndarray, p: np.ndarray, fixed: np.ndarray) -> pd.DataFrame:
    """Kurva untuk agen: berita yang sudah diputuskan scope_guard/similar ikut
    dihitung sesuai keputusannya; berita model diurutkan dari paling yakin
    (|p - 0,5| terbesar) dan diterima satu per satu dengan keputusan p >= 0,5."""
    model_idx = np.where(fixed == -2)[0]
    order = model_idx[np.argsort(-np.abs(p[model_idx] - 0.5), kind="mergesort")]
    base = np.where(fixed >= 0)[0]
    base_err = int((fixed[base] != y[base]).sum())
    n = len(y)
    rows, errs = [], base_err
    pred = (p >= 0.5).astype(int)
    for k, idx in enumerate(order, 1):
        errs += int(pred[idx] != y[idx])
        acc = len(base) + k
        if k % 10 == 0 or k == len(order):
            rows.append({"coverage": acc / n, "risk_galat": errs / acc, "n_diputuskan": acc})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=REPO / "results" / "anggota_d")
    ap.add_argument("--models-dir", type=Path, default=PROTO / "models")
    ap.add_argument("--text-col", default="text_clean_v31")
    ap.add_argument("--limit", type=int, default=None, help="hanya untuk uji cepat; jangan dilaporkan")
    a = ap.parse_args()
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)

    sha = sha256(a.dataset)
    print(f"SHA-256 dataset: {sha}\nSama dengan file A: {sha == SHA_A}")
    cols = ["raw_id", "source", "label", "title", "text_raw", a.text_col, "random_split"]
    df = pd.read_csv(a.dataset, usecols=cols, encoding="utf-8-sig")
    va = df.loc[df["random_split"] == "VALIDASI"].reset_index(drop=True)
    assert (va["random_split"] == "VALIDASI").all()  # UJI tidak pernah dipakai
    del df
    if a.limit:
        va = va.iloc[: a.limit].reset_index(drop=True)
    print(f"VALIDASI: {len(va)} berita ({int(va.label.sum())} HOAKS)")

    import sklearn
    agent = load_default_agent(a.models_dir)
    for w in agent.warnings:
        print("[PERINGATAN agen]", w)
    print("Status agen:", agent.status())

    t0 = time.time()
    pred = run_agent(agent, va)
    print(f"Agen selesai dalam {time.time() - t0:.1f} detik")
    y = pred["label"].to_numpy()
    p_model = pred["p_model"].astype(float).to_numpy()

    # p model pada kolom bersih dataset (pembanding kesetaraan & pembanding B3)
    p_ds = agent.model.predict_proba(va[a.text_col].fillna("").astype(str).tolist())[:, 1]
    pred["p_model_kolom_dataset"] = p_ds
    pred["teks_sama_dengan_dataset"] = (pred["teks_bersih_agen"].str.strip()
                                       == va[a.text_col].fillna("").astype(str).str.strip())

    # ---------------- 1-2. sistem x C_FN -----------------------------------
    rows = []
    for c_fn in C_FNS:
        lo, hi = thresholds(c_fn)
        t_cost = C_FP / (C_FP + c_fn)
        systems = {
            "model_ambang_0.5": ((p_ds >= 0.5).astype(int), "p >= 0,50"),
            "model_ambang_biaya": ((p_ds >= t_cost).astype(int), f"p >= {t_cost:.4f} (C_FP/(C_FP+C_FN))"),
            "model_selektif": (model_decisions(p_ds, c_fn), f"VALID p <= {lo:.4f}, HOAKS p >= {hi:.2f}"),
            "agen": (agent_decisions(pred, c_fn), f"scope_guard + similar (tau {agent.config.tau_similar}) + ambang {lo:.4f}/{hi:.2f}"),
        }
        for name, (dec, rule) in systems.items():
            s = summarize(y, dec, c_fn)
            s.update(bootstrap_ci(y, dec, c_fn))
            rows.append({"c_fn": c_fn, "c_fp": C_FP, "c_r": C_R, "sistem": name, "aturan": rule, **s})
        if c_fn == MAIN_C_FN:
            pred["keputusan_agen_cfn5"] = pd.Series(agent_decisions(pred, c_fn)).map({1: HOAKS, 0: VALID, -1: PERLU_VERIFIKASI})
    ring = pd.DataFrame(rows)
    for c in ring.columns:
        if ring[c].dtype == float:
            ring[c] = ring[c].round(6)
    ring.to_csv(out / "D3_RINGKASAN_SISTEM.csv", index=False)

    # ---------------- 3. lapisan ---------------------------------------------
    dec5 = agent_decisions(pred, MAIN_C_FN)
    lay = []
    for layer in (LAYER_SCOPE_GUARD, LAYER_SIMILAR, LAYER_MODEL):
        m = (pred["lapisan"] == layer).to_numpy()
        k = int(m.sum())
        row = {"lapisan": layer, "n": k, "porsi": k / len(y), "n_label_hoaks": int(y[m].sum())}
        dm = dec5[m]
        if layer == LAYER_SCOPE_GUARD:
            al = pred.loc[m, "alasan"].astype(str)
            row["n_alasan_bahasa"] = int(al.str.contains("bahasa").sum())
            row["n_alasan_lainnya"] = int(k - row["n_alasan_bahasa"])
            # Apa yang akan diputuskan model seandainya scope_guard tidak ada
            pm = pred.loc[m, "p_model"].astype(float).to_numpy()
            row["model_akan_benar_tanpa_guard"] = int((model_decisions(np.nan_to_num(pm, nan=0.5), MAIN_C_FN) == y[m]).sum())
        if layer == LAYER_SIMILAR:
            pm = pred.loc[m, "p_model"].astype(float).to_numpy()
            row["model_akan_benar_tanpa_similar"] = int((model_decisions(pm, MAIN_C_FN) == y[m]).sum())
        row["n_diputuskan_hoaks"] = int((dm == 1).sum())
        row["n_diputuskan_valid"] = int((dm == 0).sum())
        row["n_perlu_verifikasi"] = int((dm == -1).sum())
        dec_mask = dm != -1
        row["galat_diputuskan"] = float((dm[dec_mask] != y[m][dec_mask]).mean()) if dec_mask.any() else None
        tp = int(((dm == 1) & (y[m] == 1)).sum())
        nh = row["n_diputuskan_hoaks"]
        row["precision_hoaks"] = tp / nh if nh else None
        if nh:
            lo_cp, hi_cp = clopper_pearson(tp, nh)
            row["precision_hoaks_ci95_clopper_pearson"] = f"[{lo_cp:.4f}, {hi_cp:.4f}]"
        lay.append(row)
    pd.DataFrame(lay).round(6).to_csv(out / "D3_LAPISAN.csv", index=False)

    # ---------------- 4. per sumber -----------------------------------------
    src_rows = []
    for src, g in pred.assign(_d=dec5).groupby("source"):
        yy, dd = g["label"].to_numpy(), g["_d"].to_numpy()
        s = summarize(yy, dd, MAIN_C_FN)
        s.pop("macro_f1_selektif")  # tiap sumber hanya berisi satu kelas, macro-F1 tidak bermakna
        src_rows.append({"source": src, "n_hoaks": int(yy.sum()), **s,
                         "porsi_scope_guard": float((g["lapisan"] == LAYER_SCOPE_GUARD).mean()),
                         "porsi_similar": float((g["lapisan"] == LAYER_SIMILAR).mean())})
    pd.DataFrame(src_rows).round(6).to_csv(out / "D3_PER_SUMBER.csv", index=False)

    # ---------------- 5. risk-coverage --------------------------------------
    fixed = np.where(pred["lapisan"] == LAYER_MODEL, -2, pred["aksi"].map(ACT_CODE))
    # berita scope_guard berstatus PERLU VERIFIKASI (-1) tidak pernah diputuskan
    fixed_dec = fixed.copy()
    rc_agent = risk_coverage_curve(y, np.nan_to_num(p_model, nan=0.5),
                                   np.where(fixed_dec == -1, -3, fixed_dec))
    rc_agent["sistem"] = "agen"
    rc_model = risk_coverage_curve(y, p_ds, np.full(len(y), -2))
    rc_model["sistem"] = "model_saja"
    rc = pd.concat([rc_model, rc_agent], ignore_index=True).round(6)
    rc.to_csv(out / "D3_RISK_COVERAGE.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for name, g in rc.groupby("sistem"):
            ax.plot(g["coverage"], g["risk_galat"] * 100, label=name.replace("_", " "))
        for _, r in ring.loc[ring["sistem"] == "agen"].iterrows():
            ax.scatter(r["coverage"], r["galat_diputuskan"] * 100, zorder=3)
            ax.annotate(f"agen C_FN={int(r['c_fn'])}", (r["coverage"], r["galat_diputuskan"] * 100),
                        textcoords="offset points", xytext=(-70, 6), fontsize=8)
        ax.set_xlabel("Coverage (porsi berita yang diputuskan)")
        ax.set_ylabel("Galat pada berita yang diputuskan (%)")
        ax.set_title("Risk-coverage di VALIDASI (V3.1)")
        ax.set_xlim(0.9, 1.0)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "D3_RISK_COVERAGE.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # matplotlib tidak wajib
        print("[INFO] grafik tidak dibuat:", e)

    # ---------------- kesalahan ---------------------------------------------
    err = pred.assign(_d=dec5).loc[lambda d: (d["_d"] != -1) & (d["_d"] != d["label"])]
    err = err.assign(jenis=np.where(err["label"] == 1, "FN (hoaks lolos)", "FP (valid dituduh)"))
    err[["raw_id", "source", "label", "jenis", "lapisan", "alasan", "p_hoaks", "p_model", "skor_mirip", "judul_mirip"]] \
        .to_csv(out / "D3_KESALAHAN_AGEN.csv", index=False)

    # ---------------- 6. kalibrasi -------------------------------------------
    base = agent.model.calibrated_classifiers_[0].estimator
    scores = base.decision_function(va[a.text_col].fillna("").astype(str).tolist())
    cf = cross_fit_ece(scores, y)
    kal = {
        "ece_validasi_seperti_b3": compute_ece(y, p_ds),
        "ece_crossfit_2_lipatan": cf["ece_crossfit"],
        "catatan": ("Platt B3 dipasang di VALIDASI, jadi ECE di VALIDASI optimis. "
                    "ECE cross-fit: Platt dipasang di separuh VALIDASI dan dinilai di separuh lainnya, "
                    "lalu ditukar. Angka ini yang jujur untuk laporan sebelum UJI."),
        "reliability_bins_validasi": compute_reliability_bins(y, p_ds),
    }

    # ---------------- 7. kesetaraan -----------------------------------------
    d_ds = model_decisions(p_ds, MAIN_C_FN)
    d_ag = model_decisions(np.nan_to_num(p_model, nan=0.5), MAIN_C_FN)
    has_p = ~np.isnan(p_model)
    kes = {
        "porsi_teks_bersih_sama_persis": float(pred["teks_sama_dengan_dataset"].mean()),
        "n_berita_dengan_p": int(has_p.sum()),
        "selisih_p_maks": float(np.nanmax(np.abs(p_model - p_ds))),
        "keputusan_model_berbeda_cfn5": int((d_ds[has_p] != d_ag[has_p]).sum()),
    }

    # ---------------- 8. latensi --------------------------------------------
    t = pred["total_ms"].astype(float)
    lat = {"p50_ms": float(t.quantile(0.5)), "p95_ms": float(t.quantile(0.95)),
           "mesin": f"{platform.system()} {platform.machine()} | Python {platform.python_version()}"}

    meta = {
        "tugas": "D3 evaluasi agen di VALIDASI", "tanggal": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_version": DATA_VERSION, "dataset_sha256": sha, "sama_dengan_file_a": sha == SHA_A,
        "split": "random_split = VALIDASI", "n": int(len(y)), "n_hoaks": int(y.sum()),
        "uji_dibaca": False, "limit": a.limit,
        "biaya": {"c_fp": C_FP, "c_r": C_R, "c_fn": list(C_FNS), "c_fn_utama": MAIN_C_FN},
        "agen": {**agent.status(), "model_path": str(Path(agent.model_path).name),
                 "data_version_model": agent.model_card.get("data_version")},
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "bootstrap": {"n": N_BOOT, "seed": SEED},
        "kalibrasi": {k: v for k, v in kal.items()},
        "kesetaraan_preprocessing": kes, "latensi": lat,
    }
    (out / "D3_METADATA.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    pred.drop(columns=["teks_bersih_agen"]).to_csv(out / "D3_PREDIKSI_AGEN_VALIDASI.csv", index=False)

    # ---------------- cetak ringkas -----------------------------------------
    show = ["c_fn", "sistem", "coverage", "galat_diputuskan", "fp", "fn", "ditolak", "kerugian_harapan"]
    print("\n" + ring[show].to_string(index=False))
    print("\nLapisan (C_FN=5):\n" + pd.DataFrame(lay)[["lapisan", "n", "n_label_hoaks", "n_diputuskan_hoaks",
                                                      "n_perlu_verifikasi", "precision_hoaks"]].to_string(index=False))
    print(f"\nECE seperti B3: {kal['ece_validasi_seperti_b3']:.4f} | ECE cross-fit: {kal['ece_crossfit_2_lipatan']:.4f}")
    print(f"Kesetaraan: {kes}")
    print(f"Latensi: p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms")
    print(f"\nHasil di {out}")


if __name__ == "__main__":
    main()
