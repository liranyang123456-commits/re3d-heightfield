#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shrink the bitmap payload of the manuscript figures so the compiled PDF scrolls smoothly.

Several qualitative figures embed screen-resolution RGB arrays with lossless Flate
compression, which makes the merged manuscript ~32 MB and stalls PDF viewers. This
pass re-encodes the embedded rasters through Ghostscript at TARGET_DPI with DCT
compression and leaves text, axes and line art as vectors.

Ghostscript expresses image resolution in the figure's own coordinate space, but each
figure is typeset much smaller than its natural size. The downsampling target is
therefore derived per figure from its page width so that every raster lands at
TARGET_ONPAGE_DPI once typeset, comfortably above Elsevier's 300 dpi halftone floor.
Originals are preserved in figures/_orig_hires/.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

GS = Path(r"D:\Ctex\Ghostscript\gs10.01.1\bin\gswin64c.exe")
BASE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results")
FIG_DIR = BASE / "figures"
STAGE_DIR = BASE / "paper" / "_cmig_clean_stage"
BACKUP_DIR = FIG_DIR / "_orig_hires"

TARGET_ONPAGE_DPI = 400
JPEG_QUALITY = 92
# Widest text block the figures are typeset into (elsarticle review, inches).
TYPESET_WIDTH_IN = 6.5
# Only figures above this size carry a bitmap payload worth re-encoding.
SIZE_THRESHOLD_MB = 0.8


def page_width_pt(pdf: Path) -> float:
    ps = f"({pdf.as_posix()}) (r) file runpdfbegin 1 pdfgetpage /MediaBox get 2 get == quit"
    out = subprocess.run([str(GS), "-q", "-dNODISPLAY", "-dNOSAFER", "-c", ps],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[-1])


def optimize(src: Path, dst: Path, dpi: int) -> None:
    subprocess.run([
        str(GS), "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
        "-dNOPAUSE", "-dBATCH", "-dQUIET", "-dSAFER",
        "-dDetectDuplicateImages=true",
        "-dDownsampleColorImages=true", "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}",
        "-dDownsampleGrayImages=true", "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={dpi}",
        "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
        "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
        f"-dJPEGQ={JPEG_QUALITY}",
        "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dCompressFonts=true",
        f"-sOutputFile={dst}", str(src),
    ], check=True)


def main() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    # Re-runs must start from the pristine originals, not from an earlier pass, so a
    # figure stays in scope once it has been backed up even though it is now small.
    targets = sorted({f for f in FIG_DIR.glob("*.pdf")
                      if f.stat().st_size > SIZE_THRESHOLD_MB * 1e6}
                     | {FIG_DIR / f.name for f in BACKUP_DIR.glob("*.pdf")})

    print(f"{'figure':<42}{'before':>10}{'after':>10}{'saved':>9}{'gs dpi':>9}{'on-page':>9}")
    total_before = total_after = 0.0
    for src in targets:
        original = BACKUP_DIR / src.name
        if not original.is_file():
            shutil.copy(src, original)

        before = original.stat().st_size / 1e6
        shrink = TYPESET_WIDTH_IN / (page_width_pt(original) / 72.0)
        dpi = int(round(TARGET_ONPAGE_DPI * shrink))

        tmp = src.with_suffix(".opt.pdf")
        optimize(original, tmp, dpi)
        after = tmp.stat().st_size / 1e6

        if after >= before:
            tmp.unlink()
            print(f"{src.name:<42}{before:>9.2f}M{'skipped':>10}")
            continue

        tmp.replace(src)
        total_before += before
        total_after += after
        print(f"{src.name:<42}{before:>9.2f}M{after:>9.2f}M"
              f"{100 * (1 - after / before):>8.0f}%{dpi:>9}{TARGET_ONPAGE_DPI:>9}")

        staged = STAGE_DIR / src.name
        if staged.is_file():
            shutil.copy(src, staged)

    print(f"\n{'TOTAL':<42}{total_before:>9.2f}M{total_after:>9.2f}M"
          f"{100 * (1 - total_after / max(total_before, 1e-9)):>8.0f}%")
    print(f"originals kept in {BACKUP_DIR}")


if __name__ == "__main__":
    main()
