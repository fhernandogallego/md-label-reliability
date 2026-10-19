#!/bin/bash
# Prepara el paquete para arXiv a partir de jcim.tex y si.tex: un único PDF (manuscrito + Supporting Information al final),
# referencias S1, S2... fijadas en el propio fuente (sin depender de si.aux), bibliografía ya compilada (.bbl).
# Uso: paper/arxiv/preparar_arxiv.sh   → paper/arxiv/arxiv_paquete.tar.gz, paper/arxiv/ms.pdf (comprobación) y metadatos.txt
set -e
cd "$(dirname "$0")/.."                                   # paper/
compila() { pdflatex -interaction=nonstopmode "$1" > /dev/null 2>&1 || { grep -n "^!" "${1%.tex}.log" | head; exit 1; }; }
compila si.tex; compila si.tex
compila jcim.tex; bibtex jcim > /dev/null; compila jcim.tex; compila jcim.tex

B=arxiv/paquete; rm -rf "$B"; mkdir -p "$B/figures"
python3 - <<'PY'
import re
s = open("jcim.tex").read()
etiquetas = [l.strip() for l in open("si.aux") if l.startswith("\\newlabel{si:")]
s = s.replace("\\usepackage{xr-hyper}\n", "\\usepackage{pdfpages}\n")
s = s.replace("\\externaldocument{si}\n", "\\makeatletter\n" + "\n".join(etiquetas) + "\n\\makeatother\n")
s = s.replace("\\end{document}", "\\clearpage\n\\includepdf[pages=-]{si.pdf}\n\\end{document}")
s = re.sub(r"^%  Build:.*\n(%  .*\n)*", "", s, flags=re.M)        # quita las instrucciones de compilación locales
open("arxiv/paquete/ms.tex", "w").write(s)
PY
cp autores.tex preambulo_comun.tex numeros*.tex tabla_*.tex si.pdf "$B/"
cp jcim.bbl "$B/ms.bbl"
# arXiv publica el fuente: quitar todos los comentarios LaTeX (notas internas en español) de los .tex del paquete
python3 - "$B" <<'PY'
import glob, re, sys
for f in glob.glob(sys.argv[1] + "/*.tex"):
    salida = []
    for l in open(f):
        if l.lstrip().startswith("%"): continue                      # línea solo de comentario
        salida.append(re.sub(r"(?<!\\)%.*$", "%", l.rstrip("\n")) + "\n" if re.search(r"(?<!\\)%", l) else l)
    open(f, "w").write("".join(salida))
PY
for f in $(grep -o "figures/[A-Za-z0-9_]*\.pdf" jcim.tex | sort -u); do cp "$f" "$B/figures/"; done

# comprobación: compilar el paquete aislado, como lo hará arXiv (sin bibtex, con el .bbl)
( cd "$B" && for i in 1 2 3; do pdflatex -interaction=nonstopmode ms.tex > /dev/null 2>&1 || true; done
  grep -n "^!\|Reference .* undefined\|Citation .* undefined" ms.log | head && test -z "$(grep -c '^!' ms.log | grep -v '^0$')" )
cp "$B/ms.pdf" arxiv/ms.pdf
( cd "$B" && rm -f ms.aux ms.log ms.out ms.pdf ms.blg acs-ms.bib *.xwm && tar czf ../arxiv_paquete.tar.gz . )
echo "paquete: paper/arxiv/arxiv_paquete.tar.gz ($(du -h arxiv/arxiv_paquete.tar.gz | cut -f1)); PDF de comprobación: paper/arxiv/ms.pdf ($(pdfinfo arxiv/ms.pdf | grep Pages | awk '{print $2}') páginas)"
