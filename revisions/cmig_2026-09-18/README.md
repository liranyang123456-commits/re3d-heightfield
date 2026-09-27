CMIG revision workspace — 2026-09-18
====================================
This folder is a *copy* of the paper sources. The original
`re3d_cmpb_results/paper/` is unchanged.

What changed vs the 2026-09-16 submission pack
----------------------------------------------
1. Highlight 2 now names the chessboard phantom. The old wording
   ("keep Normal 3D where learned clouds collapse") over-reached:
   new-capture P1 and V3 still collapse.
2. Discussion: new paragraph "Reading the new-capture set as a stress test"
   (V1/V2/P1/V3, SGM-by-construction, no overall ranking).
3. Cover letter: explicit non-ranking sentence with VGGT ATE 3.02 vs 3.89 mm
   and scale 91x (numbers unchanged).
4. graphicspath extended so this nested folder still finds `figures/`.

Local model (qwen3-coder:30b, then unloaded)
--------------------------------------------
See LOCAL_LLM_REVIEW.md. Several LLM suggestions were *not* applied:
- did not add extra highlights (CMIG allows 3--5)
- did not claim robustness to low texture / motion (P1/V3 contradict that)
- did not rewrite the abstract (would have dropped the 2D ablation and VGGT honesty)
- did not invent a SOTA ranking

Numbers
-------
All quantitative results are copied from the original manuscript / results JSON.
Do not treat this folder as a new experiment.

Compile (from this paper/ directory after figures resolve)
----------------------------------------------------------
pdflatex paper_cmig.tex
bibtex paper_cmig
pdflatex paper_cmig.tex
pdflatex paper_cmig.tex
