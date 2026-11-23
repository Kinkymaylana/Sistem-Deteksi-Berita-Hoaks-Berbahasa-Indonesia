"""Uji fungsional 10 skenario (Rencana Kerja, Anggota C butir 5).

Skenario: VALID, HOAKS, teks pendek, teks bahasa Inggris, hoaks beredar ulang,
input kosong, judul saja, teks sangat panjang, CSV 20 berita, tombol Jelaskan.

Agen dirakit dari file asli di models/ (load_default_agent). Contoh berita
diambil dari VALIDASI (dan satu hoaks LATIH untuk skenario beredar ulang,
karena indeks hoaks serupa dibangun dari LATIH). UJI tidak dibaca.

Kolom "lulus" diisi dari aturan yang tertulis di kolom "harapan". Skenario
VALID dan HOAKS bisa gagal secara sah bila model salah; hasilnya tetap dicatat
apa adanya.

Contoh:
    python scripts/uji_fungsional.py --data data/DATASET_V3_1.csv
Keluaran: results/uji_fungsional.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import pandas as pd

from _common import ROOT, read_rows
from src.agent import MODELS_DIR, load_default_agent
from src.batch import process, queue
from src.config import HOAKS, PERLU_VERIFIKASI, VALID
from src.explain import lime_explanation, svm_word_contributions

EN_TEXT = ("The government announced on Monday that the new toll road will open next month, "
           "and officials said the project was completed ahead of schedule despite heavy rain.")


def split_body(title: str, text_raw: str) -> str:
    t, r = (title or "").strip(), (text_raw or "").strip()
    return r[len(t):].strip() if t and r.startswith(t) else r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split-col", default="random_split")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "uji_fungsional.csv")
    args = ap.parse_args()

    agent = load_default_agent(args.models_dir)
    rows, _ = read_rows(args.data, ["raw_id", "label", "title", "text_raw"], args.split_col)
    rng = random.Random(args.seed)
    val = [r for r in rows if r[args.split_col] == "VALIDASI"]
    val_valid = rng.choice([r for r in val if r["label"] == "0"])
    val_hoaks = rng.choice([r for r in val if r["label"] == "1"])
    latih_hoaks = rng.choice([r for r in rows if r[args.split_col] == "LATIH" and r["label"] == "1"])
    csv20 = rng.sample(val, min(20, len(val)))
    long_parts = [split_body(r["title"], r["text_raw"]) for r in val if r["label"] == "0"]
    long_body, n = [], 0
    for p in long_parts:
        long_body.append(p)
        n += len(p.split())
        if n >= 5000:
            break

    results = []

    def run(no, name, title, body, harapan, check, sumber=""):
        t0 = time.perf_counter()
        err, res = "", None
        try:
            res = agent.predict(title, body)
        except Exception as e:  # dicatat, bukan disembunyikan
            err = f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        results.append({
            "no": no, "skenario": name, "sumber_contoh": sumber,
            "input_ringkas": (f"{title or ''} | {body or ''}")[:120],
            "harapan": harapan,
            "hasil_tindakan": res and res["action"], "hasil_lapisan": res and res["layer"],
            "p_hoaks": None if not res or res["p_hoaks"] is None else round(res["p_hoaks"], 4),
            "alasan": res["reason"] if res else err,
            "hoaks_mirip": "" if not res or not res["similar"] else
                f"{res['similar']['title']} ({res['similar']['score']:.3f})",
            "waktu_ms": round(ms, 1),
            "lulus": bool(res and not err and check(res)),
        })
        return res

    run(1, "Berita VALID", val_valid["title"], split_body(val_valid["title"], val_valid["text_raw"]),
        "VALID (label dataset)", lambda r: r["action"] == VALID, f"VALIDASI raw_id {val_valid['raw_id']}")
    hoaks_res = run(2, "Berita HOAKS", val_hoaks["title"], split_body(val_hoaks["title"], val_hoaks["text_raw"]),
                    "HOAKS (label dataset)", lambda r: r["action"] == HOAKS, f"VALIDASI raw_id {val_hoaks['raw_id']}")
    run(3, "Teks pendek", "Vaksin bikin magnet", "", "PERLU VERIFIKASI oleh scope_guard",
        lambda r: r["action"] == PERLU_VERIFIKASI and r["layer"] == "scope_guard")
    run(4, "Teks bahasa Inggris", "", EN_TEXT, "PERLU VERIFIKASI oleh scope_guard (aturan bahasa)",
        lambda r: r["action"] == PERLU_VERIFIKASI and r["layer"] == "scope_guard")
    run(5, "Hoaks beredar ulang", latih_hoaks["title"],
        split_body(latih_hoaks["title"], latih_hoaks["text_raw"]) + " Mohon disebarkan.",
        "HOAKS oleh lapisan similar", lambda r: r["action"] == HOAKS and r["layer"] == "similar",
        f"LATIH raw_id {latih_hoaks['raw_id']} + 2 kata tambahan")
    run(6, "Input kosong", "", "", "PERLU VERIFIKASI oleh scope_guard, tanpa galat",
        lambda r: r["action"] == PERLU_VERIFIKASI and r["layer"] == "scope_guard")
    run(7, "Judul saja", val_valid["title"], "", "Salah satu tindakan, tanpa galat",
        lambda r: r["action"] in (VALID, HOAKS, PERLU_VERIFIKASI), f"VALIDASI raw_id {val_valid['raw_id']}")
    run(8, "Teks sangat panjang", "Gabungan berita", " ".join(long_body),
        "Salah satu tindakan, tanpa galat", lambda r: r["action"] in (VALID, HOAKS, PERLU_VERIFIKASI),
        f"{len(long_body)} berita VALID VALIDASI, {n} kata")

    # 9. CSV 20 berita
    t0 = time.perf_counter()
    err, ok, info = "", False, ""
    try:
        df = pd.DataFrame({"title": [r["title"] for r in csv20],
                           "body": [split_body(r["title"], r["text_raw"]) for r in csv20]})
        q = queue(process(agent, df))
        ps = q["antrean"]["p_hoaks"].tolist()
        ok = sum(len(v) for v in q.values()) == len(csv20) and ps == sorted(ps, reverse=True)
        info = (f"antrean {len(q['antrean'])}, cocok hoaks lama {len(q['cocok_hoaks_lama'])}, "
                f"di luar cakupan {len(q['di_luar_cakupan'])}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    results.append({"no": 9, "skenario": "CSV 20 berita", "sumber_contoh": f"{len(csv20)} berita VALIDASI",
                    "input_ringkas": "", "harapan": "Semua baris diproses; antrean urut p menurun",
                    "hasil_tindakan": "", "hasil_lapisan": "", "p_hoaks": None, "alasan": err or info,
                    "hoaks_mirip": "", "waktu_ms": round((time.perf_counter() - t0) * 1000, 1), "lulus": ok})

    # 10. Tombol Jelaskan
    t0 = time.perf_counter()
    err, ok, info = "", False, ""
    try:
        clean = agent.clean(val_hoaks["title"], split_body(val_hoaks["title"], val_hoaks["text_raw"]))
        ex = svm_word_contributions(agent.model, clean)
        info = "SVM ke HOAKS: " + ", ".join(w for w, _ in ex["ke_hoaks"][:5])
        ok = bool(ex["ke_hoaks"] or ex["ke_valid"])
        try:
            lime = lime_explanation(agent.model, clean, num_samples=args.lime_samples)
            info += " | LIME: " + ", ".join(w for w, _ in lime[:5])
        except ImportError as e:
            ok, info = False, info + f" | LIME gagal: {e}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    results.append({"no": 10, "skenario": "Tombol Jelaskan", "sumber_contoh": f"VALIDASI raw_id {val_hoaks['raw_id']}",
                    "input_ringkas": "", "harapan": "Kontribusi kata SVM dan bobot LIME tampil",
                    "hasil_tindakan": hoaks_res and hoaks_res["action"], "hasil_lapisan": "", "p_hoaks": None,
                    "alasan": err or info, "hoaks_mirip": "",
                    "waktu_ms": round((time.perf_counter() - t0) * 1000, 1), "lulus": ok})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"Status lapisan: {agent.status()}")
    for wmsg in agent.warnings:
        print("PERINGATAN:", wmsg)
    for r in results:
        print(f"{r['no']:>2}. {r['skenario']:<22} lulus={r['lulus']!s:<5} {r['hasil_tindakan'] or ''} {r['alasan'][:90]}")
    print(f"Lulus {sum(r['lulus'] for r in results)}/{len(results)} -> {args.out}")


if __name__ == "__main__":
    main()
