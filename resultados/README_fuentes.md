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

## SCAYLE pilot extension

`etiquetas_ampliacion.csv` holds per-run labels (ln D in Å²/ns, end-point and origin-averaged at
T/2) for runs that reuse the pilot starting structures: two runs at 393 K in each of the 10 pilot
systems, two runs in a 2×2×2 replicated box in 4 of them, and two runs from structures generated
with the HTP-MD equilibration protocol in 3 of them (not used in the manuscript). It also repeats
the HTP-MD control labels of those systems. `ampliacion.py` computes the ratios, dynamic-noise
estimates and system-bootstrap intervals (`ampliacion.json`, `paper/numeros_ampliacion.tex`).
With these sample sizes the intervals are descriptive. The replicated box keeps the starting
structure, so these runs do not test the structural share of the noise.
The four-system box comparison has no prespecified equivalence margin; its intervals do not
establish negligible finite-size bias. The point variance ratio only describes dynamic precision
under the stated independence assumptions, and wall-time costs were not benchmarked.

## Long-run target and approximate detectability

`objetivo_largo.json` contains calibrated Gaussian sensitivity scenarios whose oracle recall is
the mean recall from selecting on the exact expected short label. This is not an empirical
ceiling for arbitrary policies or equilibrium transport. `detectabilidad` retains the simulated
advantage distribution's mean, SD and the fraction exceeding 1.96 times that SD. No null
distribution, actual bootstrap test or per-campaign noise refitting is simulated. The rough
`n80_aproximado` extrapolation assumes Gaussian errors and SD proportional to N^(-1/2); it is
not a validated sample-size requirement. Version 1.0.11 replaces the earlier misleading
`potencia` field names without changing the simulation design or numerical estimates.

The `cota_fiabilidad` fields in `ml_demo.json` are plug-in diagnostics of the population inequality
r >= corr(prediction, label)^2 under independent evaluation noise. They do not define a new
reliability estimate or a calibrated combined confidence interval. The replica-based estimate
and interval remain the reported reliability and ceiling.
