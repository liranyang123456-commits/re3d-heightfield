# -*- coding: utf-8 -*-
"""Verify clean-stage completeness and build the Displays submission zip."""
import re
import shutil
import zipfile
from pathlib import Path

PAPER = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\paper")
STAGE = PAPER / "_cmig_clean_stage"
FIGS = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\figures")

tex = (STAGE / "paper_cmig.tex").read_text(encoding="utf-8")
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
print("referenced figures:", len(figs))

# ensure every referenced figure is present in the stage dir
missing = []
for f in figs:
    name = f if Path(f).suffix else f + ".pdf"
    name = Path(name).name
    dst = STAGE / name
    if not dst.is_file():
        src = FIGS / name
        if src.is_file():
            shutil.copy(src, dst)
            print("copied", name)
        else:
            missing.append(f)
print("missing:", missing)

# supplementary figures live in figures/ too
supp = (STAGE / "supplementary_material.tex").read_text(encoding="utf-8")
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", supp):
    name = f if Path(f).suffix else f + ".pdf"
    name = Path(name).name
    dst = STAGE / name
    if not dst.is_file():
        src = FIGS / name
        if src.is_file():
            shutil.copy(src, dst)
            print("copied (supp)", name)
        else:
            print("MISSING (supp)", f)

# copy cover letter and highlights sources
for extra in ["cover_letter.tex", "highlights.tex", "references.bib", "elsarticle-num.bst"]:
    src = PAPER / extra
    if src.is_file() and not (STAGE / extra).is_file():
        shutil.copy(src, STAGE / extra)

# build the zip
zip_path = PAPER / "Displays_submission.zip"
if zip_path.exists():
    zip_path.unlink()
include_ext = {".tex", ".bib", ".bst", ".pdf", ".txt"}
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(STAGE.iterdir()):
        if p.suffix.lower() in include_ext:
            z.write(p, p.name)
print("wrote", zip_path, f"{zip_path.stat().st_size / 1e6:.2f} MB")
with zipfile.ZipFile(zip_path) as z:
    print("zip entries:", len(z.namelist()))
