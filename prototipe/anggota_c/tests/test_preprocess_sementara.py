"""Uji src/preprocess.py sementara (Anggota C).

Memastikan salinan aturan V3 identik dengan notebook Anggota A dan perilaku
pembungkus clean_news sesuai yang didokumentasikan. Uji kesetaraan terhadap
text_clean dataset ada di scripts/uji_paritas_preprocess.py (butuh dataset).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

from src import preprocess as P

NB_DIR = (Path(__file__).resolve().parents[2] / "kinkymaylana" / "sistem-deteksi-berita-hoaks-berbahasa-indonesia"
          / "notebooks" / "anggota_a")
NOTEBOOK = NB_DIR / "EDA_Preprocessing_V3_1.ipynb"
NOTEBOOK_A2 = NB_DIR / "A2_PEMBERSIHAN_V3_1_COLAB.ipynb"


def test_label_di_judul_dihapus():
    assert P.clean_news("[SALAH] Vaksin bikin magnet", "") == "Vaksin bikin magnet"
    assert P.clean_news("(HOAX+HASUT): Pesan berantai", "") == "Pesan berantai"


def test_dateline_media_di_awal_isi_dihapus():
    assert P.clean_news("Judul", "Jakarta, CNN Indonesia -- Presiden hadir.") == "Judul Presiden hadir."
    assert P.clean_news("Judul", "KOMPAS.com - Banjir melanda.") == "Judul Banjir melanda."
    assert P.clean_news("Judul", "TEMPO.CO, Jakarta - Menteri bicara.") == "Judul Menteri bicara."


def test_url_html_iklan_dihapus():
    got = P.clean_news("Judul", "Isi https://contoh.invalid/a <b>tebal</b> ADVERTISEMENT SCROLL TO RESUME CONTENT akhir")
    assert got == "Judul Isi tebal akhir"


def test_v31_hanya_menghapus_penanda():
    body = "Isi berita.Lihat Juga :Judul lain Lanjutan"
    assert "Lihat Juga" in P.clean_news("J", body, v31=False)
    assert P.clean_news("J", body) == "J Isi berita. Judul lain Lanjutan"  # default V3.1


def test_v31_sumber_tidak_diketahui_memakai_aturan_kompas():
    body = "Hoaks itu mengutip kompas.com - katanya"
    assert P.clean_news("J", body) == "J Hoaks itu mengutip katanya"
    assert P.clean_news("J", body, source="TurnBackHoax") == "J Hoaks itu mengutip kompas.com - katanya"


def test_clean_text_satu_teks():
    assert P.clean_text("KOMPAS.com - Banjir melanda kota.") == "Banjir melanda kota."


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="Clone repo kelompok tidak ada di lokasi pengujian")
def test_salinan_identik_dengan_notebook_anggota_a():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    src = "\n".join("".join(nb["cells"][6]["source"]).splitlines()[11:57])
    ns = {"re": re, "unicodedata": unicodedata}
    exec(src, ns)
    for name in ["WORD_RE", "RE_PREFIX", "RE_PARTS", "RE_URL", "RE_HTML", "RE_CNN",
                 "RE_KOMPAS", "RE_TEMPO", "RE_BOILER", "RE_GAMBAS"]:
        assert ns[name].pattern == getattr(P, name).pattern, name
    assert ns["POS"] == P.POS and ns["NEG"] == P.NEG
    for s in ["[SALAH] a b", "Jakarta, CNN Indonesia -- x", "KOMPAS.com - y\nLIHAT JUGA: z", "a === = === b"]:
        for b in (False, True):
            assert ns["clean_piece"](s, body=b) == P.clean_piece(s, body=b)
        assert ns["tag_info"](s) == P.tag_info(s)


@pytest.mark.skipif(not NOTEBOOK_A2.exists(), reason="Clone repo kelompok tidak ada di lokasi pengujian")
def test_salinan_v31_identik_dengan_notebook_a2():
    import collections
    nb = json.loads(NOTEBOOK_A2.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][4]["source"])
    src = src[src.index("NAV = re.compile"):src.index("def run(")]
    ns = {"re": re, "collections": collections, "unicodedata": unicodedata}
    exec(src, ns)
    for name in ["NAV", "DATELINE", "PHOTO"]:
        assert ns[name].pattern == getattr(P, name).pattern, name
    for t in ["a.Lihat Juga :b", "KOMPAS.com - x. Foto: KOMPAS.com/ANDI PRATAMA y", "kutip kompas.com - z"]:
        for src_ in ["Kompas", "CNN", "TurnBackHoax", None]:
            assert ns["clean_text_v31"](t, src_) == P.clean_text_v31(t, src_)


def test_tahap_v31_memakai_file_asli_anggota_a():
    from src import preprocess_v31 as A
    assert P.clean_text_v31 is A.clean_text_v31
    assert P.NAV is A.NAV and P.DATELINE is A.DATELINE and P.PHOTO is A.PHOTO
    body = "Hoaks itu mengutip kompas.com - katanya"
    assert P.clean_news("J", body, source="UNKNOWN") == "J " + A.clean_v31_for_unknown_source(body)
