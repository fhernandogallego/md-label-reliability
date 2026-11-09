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
$PY ralentizacion.py 2>/dev/null | head -1 || true                       # qué cambia durante la ralentización (agrega resultados/ralentizacion/; calcular: --calcular)
$PY solvatacion.py > /dev/null 2>&1 && echo "solvatación del Li y agregación iónica: ok"   # cachés en resultados/solvatacion/ (calcular: necesita data/iones)
$PY referencias.py 2>/dev/null | grep -E "trayectorias|acotada" || true          # ~4 min: sensibilidad a la referencia (apéndice)
$PY frontera.py > /dev/null 2>&1 && echo "frontera coste-error: ok"
$PY techo.py > /dev/null 2>&1 && echo "techo de ruido: ok"             # después de incertidumbre.py y analizar_prefijos.py
$PY asignacion.py > /dev/null 2>&1 && echo "asignación adaptativa: ok"   # ~3 min; después de techo.py
$PY validacion_asignacion.py > /dev/null 2>&1 && echo "validación real de la asignación: ok"
$PY ml_demo.py > /dev/null 2>&1 && echo "demostración de aprendizaje automático: ok"   # ~1 min; necesita rdkit (requirements.txt)
$PY generalidad.py > /dev/null 2>&1 && echo "generalidad (anión TFSI y conductividad): ok"   # ~1 min; necesita rdkit
$PY toc.py > /dev/null 2>&1 && echo "gráfico de la tabla de contenidos: ok"
$PY hacer_numeros.py | head -1
if command -v pdflatex >/dev/null; then   # si.tex antes que jcs.tex: el manuscrito toma de si.aux las referencias S (xr)
  (cd paper && pdflatex -interaction=batchmode si && pdflatex -interaction=batchmode si \
     && pdflatex -interaction=batchmode jcs && bibtex jcs && pdflatex -interaction=batchmode jcs && pdflatex -interaction=batchmode jcs) > /dev/null \
     && echo "PDF regenerado (paper/jcs.pdf y paper/si.pdf)"
else echo "Sin pdflatex en este equipo: las cifras y tablas están actualizadas; falta compilar paper/si.tex y paper/jcs.tex (p. ej. en Overleaf)."; fi
