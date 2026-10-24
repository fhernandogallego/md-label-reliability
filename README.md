# Reliability of short molecular dynamics runs as screening labels

Code and derived data for the paper *"Reliability of Short Molecular Dynamics Runs as Screening Labels:
Noise Ceilings, Time Dependence and Adaptive Replicate Allocation"* (F. Hernando-Gallego, D. Vasilev-Vasilev,
D. García-Santos, A. Campo-Camacho, A. Gómez-Gómez). Preprint: ChemRxiv,
https://doi.org/10.26434/chemrxiv.15010895/v1.

The analysis uses the public HTP-MD collection of polymer electrolyte simulations (Xie et al., *Nat. Commun.*
2022, 13, 3415; *APL Mach. Learn.* 2023, 1, 046108). Raw trajectories are not redistributed here; the scripts
download what they need from the HTP-MD platform.

## Contents

| Path | What it is |
|---|---|
| `*.py`, `actualizar.sh` | Analysis pipeline. `actualizar.sh` runs every step and regenerates all numbers, tables and figures |
| `resultados/` | Derived results: per-trajectory table, structures (SMILES), per-trajectory caches of the prefix and segment analyses, and one JSON file per analysis |
| `paper/` | LaTeX sources of the manuscript (`jcim.tex`) and Supporting Information (`si.tex`). Every number in the text is a macro written by the scripts (`numeros*.tex`, `tabla_*.tex`) |

Main scripts: `analizar_curvas.py` (regime and lag dependence), `incertidumbre.py` and `techo.py` (reliability,
noise ceiling, noise sources), `analizar_prefijos.py`, `referencias.py` and `frontera.py` (truncated
trajectories, stopping rules, cost-error frontier), `segmentos.py` (ageing within runs), `ralentizacion.py` (what slows down with the lithium: anion, polymer
and energy relaxation), `asignacion.py` and
`validacion_asignacion.py` (adaptive replicate allocation and its check on real runs), `ml_demo.py`
(random forest against the noise ceiling), `generalidad.py` (the same analyses for the anion diffusivity), `toc.py` (table-of-contents graphic);
`comun.py` holds the shared random seed. Script comments are in Spanish.

## Reproducing the results

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./actualizar.sh
cd paper && pdflatex si && pdflatex jcim && bibtex jcim && pdflatex jcim && pdflatex jcim
```

Without the raw data in `data/`, the scripts regenerate their tables, macros and figures from the saved
results in `resultados/`. Downloading and processing the raw trajectories (`descargar_*.py`,
`extraer_iones.py`) needs several hundred GB and is only required to recompute the trajectory-level analyses.

## License

Code: MIT (`LICENSE`). Derived data in `resultados/`: CC BY 4.0 (`LICENSE-data`). Please cite the paper and the
HTP-MD sources when using them.
