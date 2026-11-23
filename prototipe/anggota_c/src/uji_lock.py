"""Kunci data UJI (pemilik: Anggota D, tugas D2).

Data UJI hanya boleh dibuka satu kali per run, setelah kelompok membekukan
semua keputusan. Modul ini menegakkan aturan itu dan mencatat jejaknya.

Syarat membuka UJI (semuanya wajib):
  1. Skrip dipanggil dengan --final (final=True).
  2. Manifest pembekuan ada dan berstatus DIBEKUKAN.
  3. SHA-256 dataset sama dengan yang tercatat di manifest.
  4. SHA-256 setiap berkas yang dibekukan (kode, model) masih sama.
  5. run_key terdaftar di manifest (run harus direncanakan sebelum UJI dibuka).
  6. split_col terdaftar untuk run itu.
  7. run_key belum pernah SELESAI di uji_log.csv. Mengulang hanya bisa dengan
     alasan tertulis (alasan_ulang), dan tercatat sebagai ULANG.

Setiap upaya, baik berhasil maupun ditolak, ditulis ke uji_log.csv.

Contoh pemakaian di skrip UJI:

    from src.uji_lock import add_final_args, open_uji

    ap = argparse.ArgumentParser()
    add_final_args(ap)
    a = ap.parse_args()
    with open_uji(a.dataset, run_key="UJI_LinearSVM_random_clean",
                  split_col="random_split", final=a.final,
                  manifest_path=a.manifest, log_path=a.uji_log,
                  operator=a.operator, alasan_ulang=a.alasan_ulang) as sesi:
        uji = sesi.load()                     # DataFrame baris UJI saja
        latih = sesi.load_latih()             # LATIH dari split yang sama
        ...
        m = calculate_metrics(y, pred, y_prob=prob)
        sesi.log_experiment(results_csv_path, metrics=m, ...)   # status UJI

Catatan jujur: ini kunci prosedural. Orang tetap bisa membuka CSV secara
manual. Fungsinya membuat setiap akses lewat kode kelompok tercatat dan
bisa diperiksa, sehingga laporan dapat menyatakan "UJI dibuka sekali,
setelah pembekuan pada tanggal X" dengan bukti.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

STATUS_DRAF = "DRAF"
STATUS_DIBEKUKAN = "DIBEKUKAN"

LOG_DIBUKA = "DIBUKA"
LOG_SELESAI = "SELESAI"
LOG_GAGAL = "GAGAL"
LOG_DITOLAK = "DITOLAK"
LOG_ULANG = "ULANG"

UJI = "UJI"
LATIH = "LATIH"
VALIDASI = "VALIDASI"

UJI_LOG_COLUMNS = [
    "waktu", "run_key", "status", "operator", "skrip", "split", "n_baris_uji",
    "dataset_sha256", "manifest_sha256", "git_commit", "alasan", "ringkasan",
]

REQUIRED_MANIFEST_KEYS = ("status", "versi_data", "dataset_sha256", "runs", "berkas_dibekukan", "keputusan")


class UjiTerkunci(PermissionError):
    """UJI tidak boleh dibuka; pesan menjelaskan syarat yang belum terpenuhi."""


# ---------------------------------------------------------------------------
# Bantuan
# ---------------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(manifest: Mapping) -> str:
    """Sidik isi manifest (urutan kunci tidak berpengaruh)."""
    blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _git_commit(cwd: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # git tidak ada (misalnya Colab tanpa clone)
        return ""


def _operator(op: str | None) -> str:
    if op:
        return op
    try:
        return getpass.getuser()
    except Exception:
        return "tidak_diketahui"


def load_manifest(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise UjiTerkunci(f"Manifest pembekuan tidak ditemukan: {path}. "
                          "Bekukan dulu dengan scripts/anggota_d/d2_bekukan.py.")
    with path.open(encoding="utf-8") as fh:
        m = json.load(fh)
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in m]
    if missing:
        raise ValueError(f"Manifest {path} tidak lengkap, kunci hilang: {missing}")
    return m


def resolve_frozen_path(rel: str, manifest_path) -> Path:
    """Path berkas di manifest ditulis relatif terhadap root repo."""
    root = Path(manifest_path).resolve().parent
    # manifest ada di <repo>/results/anggota_d/ -> naik dua tingkat
    for cand in (root.parent.parent / rel, root / rel, Path(rel)):
        if cand.exists():
            return cand
    return root.parent.parent / rel


def check_frozen_files(manifest: Mapping, manifest_path) -> list[str]:
    """Kembalikan daftar masalah (kosong = semua berkas cocok)."""
    problems = []
    for item in manifest.get("berkas_dibekukan", []):
        p = resolve_frozen_path(item["path"], manifest_path)
        if not p.is_file():
            problems.append(f"{item['path']}: berkas tidak ada")
            continue
        got = sha256_file(p)
        if got != item["sha256"]:
            problems.append(f"{item['path']}: SHA-256 berubah ({got[:12]}… ≠ {item['sha256'][:12]}…)")
    return problems


# ---------------------------------------------------------------------------
# uji_log.csv
# ---------------------------------------------------------------------------
def read_uji_log(log_path) -> list[dict]:
    p = Path(log_path)
    if not p.is_file() or p.stat().st_size == 0:
        return []
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def append_uji_log(log_path, row: Mapping[str, Any]) -> dict:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.is_file() or p.stat().st_size == 0
    if not new:
        with p.open(encoding="utf-8-sig", newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != UJI_LOG_COLUMNS:
            raise ValueError(f"Header {p} bukan format uji_log. Jangan menimpa log lama.")
    full = {c: row.get(c, "") for c in UJI_LOG_COLUMNS}
    full["waktu"] = full["waktu"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for k, v in list(full.items()):
        if isinstance(v, (dict, list)):
            full[k] = json.dumps(v, ensure_ascii=False)
    with p.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=UJI_LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)
    return full


def completed_runs(log_path) -> set[str]:
    return {r["run_key"] for r in read_uji_log(log_path) if r["status"] == LOG_SELESAI}


# ---------------------------------------------------------------------------
# Pemeriksaan sebelum membuka UJI
# ---------------------------------------------------------------------------
def check_can_open(dataset_path, run_key: str, split_col: str, final: bool,
                   manifest_path, log_path, alasan_ulang: str | None = None,
                   dataset_sha: str | None = None) -> tuple[dict, str, bool]:
    """Periksa semua syarat. Kembalikan (manifest, sha dataset, ulang?).

    Melempar UjiTerkunci dengan pesan yang menjelaskan syarat yang gagal.
    """
    if not final:
        raise UjiTerkunci("UJI terkunci. Jalankan skrip dengan --final hanya untuk run UJI resmi "
                          "setelah pembekuan; untuk percobaan gunakan VALIDASI.")
    m = load_manifest(manifest_path)
    if m["status"] != STATUS_DIBEKUKAN:
        raise UjiTerkunci(f"Manifest berstatus {m['status']}, belum {STATUS_DIBEKUKAN}. "
                          "Keputusan kelompok belum dibekukan.")
    belum = [k["id"] for k in m.get("keputusan", []) if k.get("status") != "DISEPAKATI"]
    if belum:
        raise UjiTerkunci(f"Keputusan belum disepakati: {belum}")
    runs = {r["run_key"]: r for r in m["runs"]}
    if run_key not in runs:
        raise UjiTerkunci(f"run_key {run_key!r} tidak terdaftar di manifest. Run UJI harus "
                          f"direncanakan sebelum pembekuan. Terdaftar: {sorted(runs)}")
    if split_col != runs[run_key]["split"]:
        raise UjiTerkunci(f"Run {run_key} terdaftar untuk split {runs[run_key]['split']!r}, bukan {split_col!r}")
    sha = dataset_sha or sha256_file(dataset_path)
    if sha != m["dataset_sha256"]:
        raise UjiTerkunci(f"SHA-256 dataset ({sha[:12]}…) berbeda dari manifest "
                          f"({m['dataset_sha256'][:12]}…). Pakai file yang dibekukan.")
    problems = check_frozen_files(m, manifest_path)
    if problems:
        raise UjiTerkunci("Berkas yang dibekukan berubah setelah pembekuan:\n  - " + "\n  - ".join(problems))
    ulang = run_key in completed_runs(log_path)
    if ulang and not (alasan_ulang and alasan_ulang.strip()):
        raise UjiTerkunci(f"Run {run_key} sudah SELESAI di {log_path}. UJI hanya dijalankan sekali. "
                          "Jika benar-benar harus diulang (misalnya skrip crash setelah menulis log), "
                          "berikan --alasan-ulang; pengulangan akan tercatat dan wajib dilaporkan.")
    return m, sha, ulang


# ---------------------------------------------------------------------------
# Sesi UJI
# ---------------------------------------------------------------------------
class UjiSession:
    def __init__(self, dataset_path, run_key, split_col, manifest, manifest_path,
                 log_path, operator, dataset_sha, ulang, alasan_ulang, skrip):
        self.dataset_path = Path(dataset_path)
        self.run_key = run_key
        self.split_col = split_col
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.log_path = Path(log_path)
        self.operator = operator
        self.dataset_sha = dataset_sha
        self.manifest_sha = manifest_sha256(manifest)
        self.ulang = ulang
        self.alasan_ulang = alasan_ulang or ""
        self.skrip = skrip
        self.run = {r["run_key"]: r for r in manifest["runs"]}[run_key]
        self.ringkasan: dict[str, Any] = {}
        self.n_uji: int | str = ""
        self._df = None

    # -- data --------------------------------------------------------------
    def _frame(self, columns: Iterable[str] | None):
        import pandas as pd
        if self._df is None:
            self._df = pd.read_csv(self.dataset_path, encoding="utf-8-sig")
        df = self._df
        return df if columns is None else df[[*columns, self.split_col]]

    def load(self, columns: Iterable[str] | None = None):
        """Baris UJI dari split run ini."""
        df = self._frame(columns)
        uji = df.loc[df[self.split_col] == UJI].copy()
        self.n_uji = len(uji)
        return uji

    def load_latih(self, columns: Iterable[str] | None = None, include_validasi: bool = False):
        """Baris LATIH (opsional + VALIDASI) dari split yang sama, untuk melatih model final."""
        df = self._frame(columns)
        keep = {LATIH, VALIDASI} if include_validasi else {LATIH}
        return df.loc[df[self.split_col].isin(keep)].copy()

    # -- pencatatan --------------------------------------------------------
    def log_experiment(self, results_csv_path, metrics: Mapping, kondisi: str | None = None,
                       model_name: str | None = None, hyperparameter=None, seed: int | None = None,
                       run_id: str | None = None):
        """Tulis baris berstatus UJI ke log eksperimen (format evaluate.py)."""
        from src.evaluate import log_experiment
        row = log_experiment(
            results_csv_path=results_csv_path,
            run_id=run_id or self.run_key,
            versi_data=self.manifest["versi_data"],
            kondisi=kondisi or self.run.get("kondisi", self.run_key),
            split=self.split_col,
            model_name=model_name or self.run.get("model", ""),
            hyperparameter=self.run.get("hyperparameter", {}) if hyperparameter is None else hyperparameter,
            seed=self.manifest.get("seed", 42) if seed is None else seed,
            metrics=metrics,
            status_evaluasi=UJI,
            allow_uji=True,
        )
        for k in ("macro_f1", "macro_f1_ci_95", "ece", "coverage_agen"):
            if metrics.get(k) is not None:
                self.ringkasan[k] = metrics[k]
        return row

    def _log(self, status: str, alasan: str = "", ringkasan: Mapping | None = None):
        append_uji_log(self.log_path, {
            "run_key": self.run_key, "status": status, "operator": self.operator,
            "skrip": self.skrip, "split": self.split_col, "n_baris_uji": self.n_uji,
            "dataset_sha256": self.dataset_sha, "manifest_sha256": self.manifest_sha,
            "git_commit": _git_commit(self.manifest_path.parent), "alasan": alasan,
            "ringkasan": dict(ringkasan) if ringkasan else "",
        })

    def __enter__(self):
        self._log(LOG_ULANG if self.ulang else LOG_DIBUKA, self.alasan_ulang)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._log(LOG_SELESAI, self.alasan_ulang, self.ringkasan)
        else:
            self._log(LOG_GAGAL, f"{exc_type.__name__}: {exc}"[:500], self.ringkasan)
        return False  # jangan menelan error


def open_uji(dataset_path, run_key: str, split_col: str, final: bool,
             manifest_path, log_path, operator: str | None = None,
             alasan_ulang: str | None = None, skrip: str | None = None) -> UjiSession:
    """Periksa syarat dan kembalikan sesi UJI (pakai dengan `with`).

    Jika syarat gagal, upaya itu dicatat sebagai DITOLAK lalu UjiTerkunci dilempar.
    """
    op = _operator(operator)
    skrip = skrip or Path(sys.argv[0]).name
    try:
        m, sha, ulang = check_can_open(dataset_path, run_key, split_col, final,
                                       manifest_path, log_path, alasan_ulang)
    except (UjiTerkunci, ValueError) as e:
        try:
            m_sha = manifest_sha256(load_manifest(manifest_path))
        except Exception:
            m_sha = ""
        append_uji_log(log_path, {
            "run_key": run_key, "status": LOG_DITOLAK, "operator": op, "skrip": skrip,
            "split": split_col, "manifest_sha256": m_sha, "alasan": str(e)[:500],
        })
        raise
    return UjiSession(dataset_path, run_key, split_col, m, manifest_path, log_path,
                      op, sha, ulang, alasan_ulang, skrip)


# ---------------------------------------------------------------------------
# Argumen baris perintah bersama
# ---------------------------------------------------------------------------
def default_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[3]
    d = repo / "results" / "anggota_d"
    return d / "PEMBEKUAN_UJI.json", d / "uji_log.csv"


def add_final_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    man, log = default_paths()
    g = parser.add_argument_group("kunci UJI (D2)")
    g.add_argument("--final", action="store_true",
                   help="buka data UJI (hanya untuk run resmi setelah pembekuan)")
    g.add_argument("--manifest", default=str(man), help="manifest pembekuan")
    g.add_argument("--uji-log", default=str(log), help="log akses UJI")
    g.add_argument("--operator", default=os.environ.get("UJI_OPERATOR"),
                   help="nama yang menjalankan (default: nama pengguna sistem)")
    g.add_argument("--alasan-ulang", default=None,
                   help="wajib jika run sudah pernah SELESAI; tercatat sebagai ULANG")
    return parser
