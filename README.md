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
| `paper/` | LaTeX sources of the manuscript (`jcs.tex`; `jcim.tex` is the earlier JCIM version) and Supplementary Material (`si.tex`). Every number in the text is a macro written by the scripts (`numeros*.tex`, `tabla_*.tex`) |

Main scripts: `analizar_curvas.py` (regime and lag dependence), `incertidumbre.py` and `techo.py` (reliability,
noise ceiling, noise sources), `analizar_prefijos.py`, `referencias.py` and `frontera.py` (truncated
trajectories, stopping rules, cost-error frontier), `segmentos.py` (ageing within runs), `ralentizacion.py` (what slows down with the lithium: anion, polymer
and energy relaxation), `solvatacion.py` (local polymer environment of lithium and ion aggregation), `asignacion.py` and
`validacion_asignacion.py` (adaptive replicate allocation and its check on real runs), `ml_demo.py`
(random forest against the noise ceiling), `generalidad.py` (the same analyses for the anion diffusivity and the ionic conductivity), `toc.py` (table-of-contents graphic);
`comun.py` holds the shared random seed. Script comments are in Spanish.

## Reproducing the results

The primary statistical population is one original 5 ns label per screening system, excluding
literature polymers and the system available only as a replica. The broader collection is retained
as a sensitivity analysis. Allocation targets the expected short-run label, not equilibrium transport.
The fingerprint model is compatible with an uncertain noise ceiling; saturation is not established.

**Pending input in this working revision:** the original per-trajectory conductivity export is
not present in this checkout. The previous aggregate summary is preserved as
`resultados/generalidad_pre_revision.json`, but is not used as a substitute for recalculation.
Until the CSV is imported, an explicit partial draft excluding conductivity can be built with:

```bash
./actualizar.sh --reuse-campaigns --sin-conductividad
```

The full pipeline fails early with an actionable error if the conductivity input is missing;
it never silently mixes the old population's summary with the revised analyses.

### From compact inputs (no raw trajectories or network access)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./actualizar.sh
```

This reads `resultados/tabla_por_trayectoria.csv`, `estructuras.csv`, `conductividad.csv`,
and the published per-trajectory caches. It recomputes reliability, models, calibrated campaigns,
tables, macros and figures, and compiles `paper/si.pdf` and `paper/jcs.pdf` if TeX is installed.
Run logs are saved outside the repository in a temporary directory; errors stop the pipeline and
print their log. `HTPMD_PYTHON` can select an existing Python interpreter.

`./actualizar.sh --reuse-campaigns` redraws allocation results from saved Monte Carlo campaigns
instead of repeating that expensive simulation. It checks that their population size and calibrated
reliability match the current analysis. Other analyses are still recomputed. For a complete scientific
recomputation use the default command.

Small-input checks can also be run directly:

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python verificar_resultados.py
```

### Recomputing trajectory-level inputs

The raw trajectories are not included. Their download and processing requires several hundred GB.
The acquisition scripts use the public HTP-MD API; its availability is an external dependency.
Start from a CSV export of the HTP-MD interface (`Sample ID` and
`Li-ion Conductivity (S/cm)` columns). Import the conductivity values without converting their
reported units:

```bash
.venv/bin/python importar_conductividad.py /path/to/molality-vs-conductivity.csv
```

The importer verifies coverage of all published trajectory IDs, records the source file hash and
writes `resultados/conductividad.csv`. For acquisition, retain the original export at
`data/molality-vs-conductivity.csv`, then run `descargar_curvas.py`, `descargar_rutas.py`, and
`descargar_meta.py`; `extraer_iones.py --largas` extracts long-run trajectories. The analysis rejects
an incomplete set of downloaded MSD curves rather than overwriting the published population.
Trajectory-derived caches retain their original preprocessing. Polymer-contact caches use inconsistent
atom-type selections and are archived for transparency; they are not interpreted as coordination shells.

To compile the documents alone:

```bash
cd paper
pdflatex -halt-on-error si
pdflatex -halt-on-error si
pdflatex -halt-on-error jcs
bibtex jcs
pdflatex -halt-on-error jcs
pdflatex -halt-on-error jcs
```

## License

Code: MIT (`LICENSE`). Derived data in `resultados/`: CC BY 4.0 (`LICENSE-data`). Please cite the paper and the
HTP-MD sources when using them.
