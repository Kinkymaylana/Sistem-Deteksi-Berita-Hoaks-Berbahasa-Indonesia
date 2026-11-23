"""D2: Bekukan keputusan kelompok sebelum UJI dibuka.

Membaca draf manifest (results/anggota_d/PEMBEKUAN_UJI_DRAF.json), memeriksa
bahwa semua keputusan sudah DISEPAKATI dan semua berkas ada, menghitung
SHA-256 dataset serta setiap berkas yang dibekukan, lalu menulis
results/anggota_d/PEMBEKUAN_UJI.json berstatus DIBEKUKAN.

Contoh (dari root repo):
    python scripts/anggota_d/d2_bekukan.py --dataset /path/DATASET_V3_1_KANDIDAT_OPERASIONAL.csv \
        --disetujui "Anggota A,Anggota B,Anggota C,Anggota D"

    # hanya memeriksa draf tanpa menulis apa pun
    python scripts/anggota_d/d2_bekukan.py --dataset ... --cek

Pembekuan tidak bisa ditimpa. Membatalkan pembekuan (--batalkan "alasan")
hanya boleh selama UJI belum pernah dibuka, dan tercatat di uji_log.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "prototipe" / "anggota_c"))

from src.uji_lock import (  # noqa: E402
    LOG_DIBUKA, LOG_SELESAI, LOG_ULANG, STATUS_DIBEKUKAN, append_uji_log,
    manifest_sha256, read_uji_log, sha256_file,
)

D = REPO / "results" / "anggota_d"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draf", default=str(D / "PEMBEKUAN_UJI_DRAF.json"))
    ap.add_argument("--out", default=str(D / "PEMBEKUAN_UJI.json"))
    ap.add_argument("--uji-log", default=str(D / "uji_log.csv"))
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--disetujui", default="", help="nama yang menyetujui, dipisah koma")
    ap.add_argument("--cek", action="store_true", help="periksa saja, jangan tulis manifest")
    ap.add_argument("--batalkan", default=None, metavar="ALASAN",
                    help="batalkan pembekuan yang ada (hanya jika UJI belum pernah dibuka)")
    a = ap.parse_args()

    out = Path(a.out)
    if a.batalkan is not None:
        return batalkan(out, Path(a.uji_log), a.batalkan)

    if out.exists() and json.loads(out.read_text(encoding="utf-8")).get("status") == STATUS_DIBEKUKAN:
        print(f"[DITOLAK] {out} sudah DIBEKUKAN. Pembekuan tidak bisa ditimpa. "
              "Gunakan --batalkan \"alasan\" jika UJI belum pernah dibuka.")
        return 2

    draf = json.loads(Path(a.draf).read_text(encoding="utf-8"))
    masalah = []

    belum = [k["id"] for k in draf["keputusan"] if k.get("status") != "DISEPAKATI"]
    if belum:
        masalah.append(f"keputusan belum DISEPAKATI: {belum}")
    for k in draf["keputusan"]:
        if k.get("status") == "DISEPAKATI" and not str(k.get("keputusan_akhir", "")).strip():
            masalah.append(f"keputusan {k['id']} DISEPAKATI tetapi keputusan_akhir kosong")

    setuju = [s.strip() for s in a.disetujui.split(",") if s.strip()] or draf.get("disetujui_oleh", [])
    if not setuju:
        masalah.append("--disetujui kosong: tulis siapa saja yang menyetujui pembekuan")

    keys = [r["run_key"] for r in draf["runs"]]
    if len(keys) != len(set(keys)):
        masalah.append("run_key ganda di daftar runs")

    if not a.dataset.is_file():
        masalah.append(f"dataset tidak ditemukan: {a.dataset}")
        sha = ""
    else:
        sha = sha256_file(a.dataset)
        harap = draf.get("dataset_sha256_harapan")
        if harap and sha != harap:
            masalah.append(f"SHA-256 dataset {sha[:12]}… berbeda dari dataset_sha256_harapan {harap[:12]}…")

    berkas = []
    for rel in draf["berkas_dibekukan"]:
        p = REPO / rel
        if not p.is_file():
            masalah.append(f"berkas tidak ada: {rel}")
            continue
        berkas.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size})

    print(f"Dataset   : {a.dataset}\nSHA-256   : {sha}")
    print(f"Runs      : {len(keys)} -> {', '.join(keys)}")
    print(f"Berkas    : {len(berkas)}/{len(draf['berkas_dibekukan'])} ditemukan")
    print(f"Disetujui : {setuju}")
    if masalah:
        print("\n[BELUM BISA DIBEKUKAN]")
        for m in masalah:
            print("  -", m)
        return 1
    if a.cek:
        print("\n[OK] Draf memenuhi syarat. Jalankan tanpa --cek untuk membekukan.")
        return 0

    manifest = {k: v for k, v in draf.items() if k not in ("dataset_sha256_harapan", "catatan_draf")}
    manifest.update({
        "status": STATUS_DIBEKUKAN,
        "dataset_file": a.dataset.name,
        "dataset_sha256": sha,
        "berkas_dibekukan": berkas,
        "disetujui_oleh": setuju,
        "tanggal_beku": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DIBEKUKAN] {out}\nSidik manifest: {manifest_sha256(manifest)}")
    print("Commit file ini SEBELUM menjalankan UJI, agar waktu pembekuan tercatat di riwayat git.")
    return 0


def batalkan(out: Path, log: Path, alasan: str) -> int:
    if not out.exists():
        print(f"[INFO] {out} tidak ada, tidak ada yang dibatalkan.")
        return 0
    if not alasan.strip():
        print("[DITOLAK] --batalkan butuh alasan tertulis.")
        return 2
    dibuka = [r for r in read_uji_log(log) if r["status"] in (LOG_DIBUKA, LOG_SELESAI, LOG_ULANG)]
    if dibuka:
        print(f"[DITOLAK] UJI sudah pernah dibuka ({len(dibuka)} entri di {log}). "
              "Pembekuan tidak boleh dibatalkan setelah UJI dilihat.")
        return 2
    m = json.loads(out.read_text(encoding="utf-8"))
    arsip = out.with_name(f"PEMBEKUAN_UJI_DIBATALKAN_{datetime.now():%Y%m%d_%H%M%S}.json")
    m["status"] = "DIBATALKAN"
    m["alasan_batal"] = alasan
    arsip.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    out.unlink()
    append_uji_log(log, {"run_key": "-", "status": "BATAL_BEKU", "alasan": alasan,
                         "manifest_sha256": manifest_sha256(m)})
    print(f"[DIBATALKAN] Manifest diarsipkan ke {arsip.name}. Perbaiki draf lalu bekukan ulang.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
