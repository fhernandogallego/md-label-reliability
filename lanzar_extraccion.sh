#!/bin/bash
# Lanza (o reanuda) la extracción de las 133 simulaciones de 50 ns en segundo plano, independiente del terminal.
#   ./lanzar_extraccion.sh [trabajos=3]     lanzar o reanudar (salta las ya hechas)
#   ./lanzar_extraccion.sh estado           cuántas van y últimas líneas del registro
#   ./lanzar_extraccion.sh parar            detenerla (se puede reanudar después)
# Para escribir en un disco externo:  HTPMD_DEST=/media/$USER/DISCO/iones ./lanzar_extraccion.sh
# Freno de disco: no empieza otra trayectoria si quedan menos de 40 GB libres (HTPMD_MIN_LIBRE_GB).
cd "$(dirname "$0")"; D="${HTPMD_DEST:-data/iones}"; export HTPMD_DEST="$D"; mkdir -p "$D"; PIDF="$D/extraccion.pid"
for p in .venv/bin/python ../../.venv/bin/python; do
  if [ -x "$p" ]; then PY="$p"; break; fi
done
PY="${PY:-python3}"
vivo() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "${1:-3}" in
  estado)
    vivo && echo "EN MARCHA (pid $(cat $PIDF))" || echo "PARADA"
    echo "destino: $D | libre en ese disco: $(df -h "$D" | tail -1 | awk '{print $4}')"
    echo "hechas: $(ls $D/*.npz 2>/dev/null | grep -vc parcial) de 133 (+1 de prueba de 5 ns si existe 9425.npz) | ocupan $(du -shL "$D" 2>/dev/null | cut -f1)"
    tail -n 4 $D/extraccion.log 2>/dev/null ;;
  parar)
    vivo && { pkill -TERM -s "$(cat $PIDF)"; echo "detenida"; } || echo "no estaba en marcha"     # mata toda la sesión (proceso principal y sus trabajadores)
    rm -f $D/*.parcial.npz "$PIDF" ;;
  *)
    vivo && { echo "ya está en marcha (pid $(cat $PIDF))"; exit 0; }
    rm -f $D/*.parcial.npz
    nohup setsid $PY extraer_iones.py --largas --trabajos "${1:-3}" > $D/salida.txt 2>&1 &
    echo $! > "$PIDF"; echo "lanzada con ${1:-3} trabajos (pid $!). Seguimiento: ./lanzar_extraccion.sh estado" ;;
esac
