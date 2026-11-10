#!/bin/bash
# Regenera resultados desde entradas compactas/cachés; con data/ también procesa curvas completas.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${HTPMD_PYTHON:-}" ]]; then
  PY="$HTPMD_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi
REUSE_CAMPAIGNS=0
WITHOUT_CONDUCTIVITY=0
for arg in "$@"; do
  case "$arg" in
    --reuse-campaigns) REUSE_CAMPAIGNS=1 ;;
    --sin-conductividad) WITHOUT_CONDUCTIVITY=1 ;;
    *) echo "Uso: ./actualizar.sh [--reuse-campaigns] [--sin-conductividad]" >&2; exit 2 ;;
  esac
done
LOG_DIR="${HTPMD_LOG_DIR:-${TMPDIR:-/tmp}/md-label-reliability-logs}"
mkdir -p "$LOG_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG_DIR/matplotlib}"
run() {
  local script="$1"
  shift
  echo "Ejecutando $script"
  if ! "$PY" "$script" "$@" > "$LOG_DIR/${script%.py}.log" 2>&1; then
    cat "$LOG_DIR/${script%.py}.log" >&2
    return 1
  fi
}
if [[ $WITHOUT_CONDUCTIVITY -eq 0 && ! -f resultados/conductividad.csv ]]; then
  if [[ -f data/molality-vs-conductivity.csv ]]; then
    run importar_conductividad.py data/molality-vs-conductivity.csv
  else
    echo "Falta la entrada de conductividad. Ejecuta importar_conductividad.py /ruta/al/export.csv." >&2
    echo "Para un borrador parcial explícito: ./actualizar.sh --sin-conductividad" >&2
    exit 1
  fi
fi
run analizar_curvas.py
run sensibilidad.py
run incertidumbre.py
run analizar_prefijos.py
run segmentos.py
run ralentizacion.py
run solvatacion.py
run referencias.py
run frontera.py
run techo.py
run auditoria_muestra.py
if [[ $REUSE_CAMPAIGNS -eq 1 ]]; then run asignacion.py tablas; else run asignacion.py; fi
run validacion_asignacion.py
run ml_demo.py
if [[ $WITHOUT_CONDUCTIVITY -eq 1 ]]; then run generalidad.py --sin-conductividad; else run generalidad.py; fi
run figuras_diagnostico.py
run toc.py
run hacer_numeros.py
if [[ $WITHOUT_CONDUCTIVITY -eq 1 ]]; then run verificar_resultados.py --sin-conductividad; else run verificar_resultados.py; fi
if command -v pdflatex >/dev/null; then
  (
    cd paper
    pdflatex -halt-on-error -interaction=batchmode si
    pdflatex -halt-on-error -interaction=batchmode si
    pdflatex -halt-on-error -interaction=batchmode jcs
    bibtex jcs
    pdflatex -halt-on-error -interaction=batchmode jcs
    pdflatex -halt-on-error -interaction=batchmode jcs
  ) > "$LOG_DIR/latex.log" 2>&1 || { cat "$LOG_DIR/latex.log" >&2; exit 1; }
  echo "PDF regenerados: paper/jcs.pdf y paper/si.pdf"
else
  echo "Análisis completo. Falta pdflatex para compilar los PDF." >&2
fi
echo "Reproducción completada. Logs: $LOG_DIR"
