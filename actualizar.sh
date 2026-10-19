#!/bin/bash
# Recalcula todo con los datos que haya en ese momento y regenera tablas, figuras y cifras del manuscrito.
# Uso: ./actualizar.sh        (se puede ejecutar mientras la extracción sigue en marcha: analiza solo lo nuevo)
set -e; cd "$(dirname "$0")"
for p in .venv/bin/python ../../.venv/bin/python; do
  if [ -x "$p" ]; then PY="$p"; break; fi
done
PY="${PY:-python3}"
$PY analizar_curvas.py > /dev/null 2>&1 && echo "curvas, pares y réplicas: ok"
$PY sensibilidad.py > /dev/null 2>&1 && echo "sensibilidad: ok"
$PY incertidumbre.py > /dev/null 2>&1 && echo "intervalos y selección: ok"
$PY analizar_prefijos.py 2>/dev/null | grep -E "trayectorias de 50 ns|validación|deriva" || true
$PY segmentos.py 2>/dev/null | grep -E "trayectorias de 50 ns|sin datos" || true    # segmentos consecutivos de 5 ns (necesita data/iones o las cachés)
$PY referencias.py 2>/dev/null | grep -E "trayectorias|acotada" || true          # ~4 min: sensibilidad a la referencia (apéndice)
$PY frontera.py > /dev/null 2>&1 && echo "frontera coste-error: ok"
$PY techo.py > /dev/null 2>&1 && echo "techo de ruido: ok"             # después de incertidumbre.py y analizar_prefijos.py
$PY asignacion.py > /dev/null 2>&1 && echo "asignación adaptativa: ok"   # ~3 min; después de techo.py
$PY validacion_asignacion.py > /dev/null 2>&1 && echo "validación real de la asignación: ok"
$PY ml_demo.py > /dev/null 2>&1 && echo "demostración de aprendizaje automático: ok"   # ~1 min; necesita rdkit (requirements.txt)
$PY hacer_numeros.py | head -1
if command -v tectonic >/dev/null; then (cd paper && tectonic main.tex 2>&1 | grep -E "error|Writing") 
elif command -v pdflatex >/dev/null; then (cd paper && pdflatex -interaction=batchmode main && bibtex main && pdflatex -interaction=batchmode main && pdflatex -interaction=batchmode main) > /dev/null && echo "PDF regenerado"
else echo "Sin compilador de LaTeX en este equipo: las cifras y tablas están actualizadas; falta compilar paper/main.tex (p. ej. en Overleaf)."; fi
