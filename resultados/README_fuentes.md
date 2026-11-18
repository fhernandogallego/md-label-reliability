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

`etiquetas_piloto.csv` contains 70 rows in 10 screening systems: 50 new 5 ns production runs
(one re-run, two new-velocity runs and two annealed preparations per system), plus the 20
original/replica HTP-MD control labels. `extremos` and `medio` are natural logarithms of lithium
diffusivity in angstrom^2/ns, for the end-point and origin-averaged (lag T/2) estimators;
`T_ns` is production duration in ns. The authors supplied this table in private-source commit
`bedfdfe`; its bytes and the authors' summary are preserved, with SHA-256 hashes in
`piloto_scayle_procedencia.json`. The HTP-MD controls retain the source notices above.

All 50 planned runs are complete. One annealed run (`9-0-413643629-0`, `rec2`) failed during
melting and was repeated with a new velocity seed; this table includes the completed replacement.

`piloto.py` recalculates `piloto_scayle.json` and the manuscript macros from the labels, using
2,000 percentile bootstrap resamples by system (seed 20261014). It preserves all modes within
each resampled system and compares every estimate and interval endpoint with the unmodified
author-supplied `piloto_scayle_referencia.json` (absolute tolerance 1e-12). It also checks the
control labels against `tabla_por_trayectoria.csv`; `verificar_resultados.py` reproduces the
random selection (seed 20261104) among the 92 eligible screening systems.

For the annealed and HTP-MD pair intervals, dispersion is RMS(pair difference)/sqrt(2), which
includes any systematic pair shift. The separately reported `sd_total_htpmd_mismos_sistemas`
uses the centered sample SD of pair differences/sqrt(2). The same-structure estimate is the
square root of the mean within-system sample variance. These distinct statistics reproduce
the original pilot analysis; they are not interchangeable variance-component estimates.

The raw pilot trajectories are not redistributed. The compact labels suffice to reproduce the
reported statistics, but do not independently validate trajectory processing. This small protocol
check does not validate adaptive allocation or identify an irreducible structural variance component.
