# Sources and reuse notices

## HTP-MD conductivity input

`conductividad.csv` is a compact input table, not a new conductivity calculation. It selects the
trajectory identifier and reported conductivity from the HTP-MD interface export obtained on
6 October 2026, aligns rows with `tabla_por_trayectoria.csv`, and preserves numerical values in
the reported S/cm units. All 6,270 trajectory IDs are covered, including independent replicas
and long runs. The original molality column is not needed by this analysis and is omitted.

Original export SHA-256:
`7d22950a792dd89cf14350d9045d8ae8eb4e2f12d8499cf63396d42b45e119b5`.
The machine-readable record is `conductividad_procedencia.json`.

The source is [HTP-MD](https://www.htpmd.matr.io/). The [official dataset page](https://data.matr.io/htp/)
identifies its license as [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
This third-party input retains those terms; the repository's CC BY 4.0 notice for the authors'
original contributions does not relicense the source data. No unit conversion is applied, and
using the reported units is not an independent validation of absolute conductivity accuracy.

Please cite Toyota Research Institute / MIT and:

- Xie et al., *Nature Communications* 13, 3415 (2022), DOI [10.1038/s41467-022-30994-1](https://doi.org/10.1038/s41467-022-30994-1).
- Xie et al., *APL Machine Learning* 1, 046108 (2023), DOI [10.1063/5.0160937](https://doi.org/10.1063/5.0160937).

## SCAYLE pilot

`piloto_scayle.json` archives the authors' aggregate results for 49 completed runs (50 planned)
in 10 screening systems. Its source-summary hash and limitations are recorded in
`piloto_scayle_procedencia.json`. The archived analysis compared production re-runs, new
velocities at the original structure, and annealed preparations.

`piloto.py` formats those results into manuscript macros. The per-run labels and trajectories
are not included in this snapshot, so the public compact-input pipeline does not independently
recalculate the pilot bootstrap intervals. This small protocol check is not an experimental
validation of the adaptive allocation policy and does not identify an irreducible structural
variance component.
