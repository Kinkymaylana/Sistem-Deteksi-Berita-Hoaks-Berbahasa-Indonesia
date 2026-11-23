"""D2: Tampilkan status kunci UJI.

Memeriksa manifest pembekuan, keutuhan berkas yang dibekukan, dan run mana
yang sudah SELESAI di uji_log.csv. Tidak membuka data UJI.

    python scripts/anggota_d/d2_status_uji.py
    python scripts/anggota_d/d2_status_uji.py --dataset /path/DATASET_V3_1_KANDIDAT_OPERASIONAL.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "prototipe" / "anggota_c"))

from src.uji_lock import (  # noqa: E402
    LOG_SELESAI, STATUS_DIBEKUKAN, check_frozen_files, load_manifest, manifest_sha256,
    read_uji_log, sha256_file,
)

D = REPO / "results" / "anggota_d"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(D / "PEMBEKUAN_UJI.json"))
    ap.add_argument("--uji-log", default=str(D / "uji_log.csv"))
    ap.add_argument("--dataset", type=Path, default=None, help="opsional: cek SHA-256 dataset")
    a = ap.parse_args()

    if not Path(a.manifest).is_file():
        print(f"Status: BELUM DIBEKUKAN ({a.manifest} belum ada). UJI terkunci.")
        print("Langkah: isi results/anggota_d/PEMBEKUAN_UJI_DRAF.json lalu jalankan d2_bekukan.py.")
        return 0
    m = load_manifest(a.manifest)
    print(f"Status manifest : {m['status']}  (dibekukan {m.get('tanggal_beku', '-')}, "
          f"oleh {', '.join(m.get('disetujui_oleh', [])) or '-'})")
    print(f"Sidik manifest  : {manifest_sha256(m)}")
    print(f"Versi data      : {m['versi_data']}  SHA-256 {m['dataset_sha256'][:16]}…")
    if a.dataset:
        ok = sha256_file(a.dataset) == m["dataset_sha256"]
        print(f"Dataset diberikan cocok: {ok}")
    probs = check_frozen_files(m, a.manifest)
    print(f"Berkas dibekukan: {len(m['berkas_dibekukan'])}, berubah/hilang: {len(probs)}")
    for p in probs:
        print("   -", p)

    log = read_uji_log(a.uji_log)
    done = Counter(r["run_key"] for r in log if r["status"] == LOG_SELESAI)
    status = Counter(r["status"] for r in log)
    print(f"\nuji_log.csv     : {len(log)} entri {dict(status)}")
    print(f"{'run_key':40s} {'split':15s} status")
    for r in m["runs"]:
        k = r["run_key"]
        st = "SELESAI" if done[k] == 1 else (f"SELESAI x{done[k]} (DIULANG)" if done[k] > 1 else "belum")
        print(f"{k:40s} {r['split']:15s} {st}")
    siap = m["status"] == STATUS_DIBEKUKAN and not probs
    print(f"\nUJI {'boleh dibuka dengan --final' if siap else 'TERKUNCI'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
