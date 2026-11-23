"""Satu sumber aturan A2 V3.1 untuk pemeriksaan A3.
source diketahui hanya selama audit / reproduksi korpus; bukan saat inferensi pengguna umum.
"""
import re,collections,unicodedata
NAV = re.compile(r'(?i)(?<!\w)(?:lihat\s+juga|baca\s+juga)\s*:\s*')
DATELINE = re.compile(r'(?i)(?<!\w)kompas\s*\.\s*com\s*[-–—]\s*')
PHOTO = re.compile(r'(?i)(?<!\w)kompas\s*\.\s*com\s*/\s*[A-Z][A-Z .-]{4,60}(?=\s|$)')
WORD = re.compile(r"(?u)\b[\w]+(?:[-’'][\w]+)*\b")
CSV_COLS = ['raw_id','source','label','year','title','url','text_raw','text_clean','extract_mode','word_count','label_prefix','decision','reasons','audit_notes','random_split','temporal_split','temporal_main']

def clean_text_v31(text, source):
    """Only remove clearly recognizable formatting markers, not arbitrary media names.
    Returns (cleaned_text, matched marker counts, warnings).
    """
    s = unicodedata.normalize('NFKC',text or '').replace('\u00a0',' ')
    hits=collections.Counter(); warnings=[]
    def del_photo(m): hits['PHOTO_CREDIT']+=1; return ' '
    # Photo credit deletion conservative; review sampled changes
    if source == 'Kompas': s = PHOTO.sub(del_photo,s)
    def strip_nav(m):
        hits['NAV_MARKER']+=1
        # Whole linked headline cannot safely be identified in flattened V3 text.
        # Remove marker ONLY; preserve adjacent words; mark for review.
        warnings.append('RELATED_HEADLINE_MAY_REMAIN')
        return ' '
    s = NAV.sub(strip_nav,s)
    def strip_dateline(m):
        hits['KOMPAS_DATELINE']+=1
        return ' '
    # Remove mechanically only for Kompas original items; for hoax quotations preserve.
    if source == 'Kompas':s=DATELINE.sub(strip_dateline,s)
    elif DATELINE.search(s): warnings.append('KOMPAS_DATELINE_IN_OTHER_SOURCE_REVIEW')
    s=re.sub(r'\s+',' ',s).strip()
    return s,hits,sorted(set(warnings))



def clean_v31_for_known_source(text, source):
    return clean_text_v31(text, source)[0]

def clean_v31_for_unknown_source(text):
    """Mode tanpa identitas sumber: tidak menghapus dateline Kompas secara selektif."""
    return clean_text_v31(text, 'UNKNOWN')[0]
