"""Tamaño de cada sistema de HTP-MD: átomos, Li⁺, TFSI⁻, cadenas de polímero y caja de simulación.

Fuente: relaxed.lmp (estructura de partida, formato LAMMPS data) dentro del zip de cada trayectoria, leída sin descargar el
zip (peticiones Range). Para no leer 6.270 ficheros de ~2 MB, el recuento es en dos niveles:
  1) cabecera de relaxed.lmp en TODAS las trayectorias: nº de átomos y caja (lx, ly, lz);
  2) iones y cadenas: Li⁺ y TFSI⁻ del nombre de la tanda en la ruta (nion_50 / 50_ion; los que no lo indican se cuentan
     del fichero completo); átomos por cadena con RDKit a partir de poly_smiles (meta.json, con hidrógenos explícitos y
     extremos cerrados con H); cadenas = (átomos − Li − 15·TFSI) / átomos por cadena;
  3) validación: en una muestra aleatoria se lee relaxed.lmp ENTERO y se cuentan directamente los átomos de tipo catión,
     de tipo anión (un N por TFSI) y las moléculas del polímero; se comprueba que coinciden con el cálculo.

Uso:  python tamanos.py descargar [k n]   cabeceras (trozo k de n para lanzar varios en paralelo; reanudable)
      python tamanos.py validar           recuento completo en N_VAL trayectorias al azar
      python tamanos.py tabla             resultados/tamanos.csv (+ resumen en pantalla)
Datos intermedios en data/tamanos/ (fuera de git).
"""
import csv, io, json, os, re, sys, time, zipfile
import numpy as np
from htpmd_remoto import Remoto

D = "data/tamanos"; os.makedirs(D, exist_ok=True)
N_VAL, SEMILLA = 30, 20261020
ATOMOS_TFSI = 15                                     # C2F6NO4S2


def abrir(i, buf=1 << 16):
    return zipfile.ZipFile(io.BufferedReader(Remoto(i), buf))


def cabecera(texto):
    n = int(re.search(r"^\s*(\d+)\s+atoms\s*$", texto, re.M).group(1))
    lados = [float(b) - float(a) for a, b in re.findall(r"^\s*(\S+)\s+(\S+)\s+[xyz]lo\s+[xyz]hi", texto, re.M)]
    return n, lados


def descargar(ids):
    for j, i in enumerate(ids, 1):
        f = f"{D}/{i}.json"
        if os.path.exists(f): continue
        for intento in range(3):
            try:
                with abrir(i).open("relaxed.lmp") as fh: txt = fh.read(4096).decode("ascii", "replace")
                n, lados = cabecera(txt)
                json.dump({"n_atomos": n, "lx": lados[0], "ly": lados[1], "lz": lados[2]}, open(f + ".tmp", "w")); os.replace(f + ".tmp", f); break
            except Exception as e:
                err = repr(e); time.sleep(10 * (intento + 1))
        else:
            print(f"{i}: ERROR {err}", flush=True)
        if j % 200 == 0: print(f"{j}/{len(ids)}", flush=True)


def recuento_completo(i):
    """Cuenta en relaxed.lmp completo: átomos, Li (tipo catión), TFSI (tipo anión, un N por ion) y moléculas de polímero."""
    meta = json.load(open(f"data/meta/{i}.json")); tc, ta = meta["cation_raw_type"], meta["anion_raw_type"]
    p0, p1 = meta["polymer_raw_type_range"]
    txt = abrir(i, 1 << 20).read("relaxed.lmp").decode("ascii", "replace")
    n, lados = cabecera(txt)
    seccion = txt.split("Atoms", 1)[1].split("\n\n", 2)[1]                      # bloque de líneas tras "Atoms # full"
    mol, tipo = [], []
    for l in seccion.splitlines():
        p = l.split()
        if len(p) >= 7: mol.append(int(p[1])); tipo.append(int(p[2]))
    mol, tipo = np.array(mol), np.array(tipo)
    poli = (tipo >= max(p0, 1)) & (tipo <= p1)
    return {"n_atomos": n, "atomos_leidos": int(len(tipo)), "n_li": int((tipo == tc).sum()), "n_tfsi": int((tipo == ta).sum()),
            "n_cadenas": int(len(np.unique(mol[poli]))), "atomos_polimero": int(poli.sum()), "lx": lados[0], "ly": lados[1], "lz": lados[2]}


def atomos_por_cadena(poly_smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(poly_smiles)
    return Chem.AddHs(m).GetNumAtoms() if m is not None else None


def iones_por_ruta(ruta):
    m = re.search(r"nion_(\d+)|_(\d+)_ion_", ruta or "")
    return int(m.group(1) or m.group(2)) if m else None


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "tabla"
    ids = [r["Sample ID"] for r in csv.DictReader(open("data/molality-vs-conductivity.csv"))]
    if accion == "descargar":
        if len(sys.argv) == 4:
            k, nt = int(sys.argv[2]), int(sys.argv[3]); ids = [x for j, x in enumerate(ids) if j % nt == k]
        descargar(ids)
    elif accion == "validar":
        rutas = {r["id"]: r["ruta"] for r in csv.DictReader(open("data/rutas.csv"))}
        sin_nombre = [i for i in ids if iones_por_ruta(rutas.get(i)) is None]                # los que no dicen el nº de iones
        rng = np.random.default_rng(SEMILLA)
        muestra = sorted(set(rng.choice(ids, N_VAL, replace=False)) | set(sin_nombre))
        out = {}
        for j, i in enumerate(muestra, 1):
            f = f"{D}/completo_{i}.json"
            if not os.path.exists(f): json.dump(recuento_completo(i), open(f, "w"))
            out[i] = json.load(open(f))
            if j % 20 == 0: print(f"validación {j}/{len(muestra)}", flush=True)
        print(f"recuento completo en {len(out)} trayectorias ({len(sin_nombre)} sin nº de iones en la ruta)")
    else:
        import pandas as pd
        T = pd.read_csv("resultados/tabla_por_trayectoria.csv", usecols=["id", "sistema"])
        rutas = {int(r["id"]): r["ruta"] for r in csv.DictReader(open("data/rutas.csv"))}
        filas, comprob = [], []
        for i in T.id:
            f = f"{D}/{i}.json"
            if not os.path.exists(f): filas.append({"id": i}); continue
            h = json.load(open(f)); meta = json.load(open(f"data/meta/{i}.json"))
            comp = json.load(open(f"{D}/completo_{i}.json")) if os.path.exists(f"{D}/completo_{i}.json") else None
            nion = iones_por_ruta(rutas.get(i))
            n_li = comp["n_li"] if comp else nion; n_tfsi = comp["n_tfsi"] if comp else nion
            apc = atomos_por_cadena(meta["poly_smiles"])
            cad = (h["n_atomos"] - n_li - ATOMOS_TFSI * n_tfsi) / apc if (apc and n_li is not None) else None
            fila = {"id": i, "n_atomos": h["n_atomos"], "n_li": n_li, "n_tfsi": n_tfsi,
                    "n_cadenas": comp["n_cadenas"] if comp else (int(round(cad)) if cad is not None else None),
                    "atomos_por_cadena": apc, "lx_A": round(h["lx"], 3), "ly_A": round(h["ly"], 3), "lz_A": round(h["lz"], 3),
                    "volumen_nm3": round(h["lx"] * h["ly"] * h["lz"] / 1000, 3), "fuente_iones_cadenas": "recuento" if comp else "calculo"}
            filas.append(fila)
            if comp is not None and cad is not None:
                comprob.append({"id": i, "cadenas_calc": cad, "cadenas_contadas": comp["n_cadenas"], "li_ruta": nion, "li_contados": comp["n_li"],
                                "tfsi_contados": comp["n_tfsi"], "atomos_ok": comp["atomos_leidos"] == comp["n_atomos"]})
            elif fila["n_cadenas"] is not None and cad is not None and abs(cad - round(cad)) > 0.01:
                fila["fuente_iones_cadenas"] = "calculo_no_entero"
        E = T.merge(pd.DataFrame(filas), on="id", how="left")
        E.to_csv("resultados/tamanos.csv", index=False)
        C = pd.DataFrame(comprob)
        print(f"{len(E)} trayectorias; con cabecera {E.n_atomos.notna().sum()}; cadenas no enteras {(E.fuente_iones_cadenas == 'calculo_no_entero').sum()}")
        if len(C):
            print(f"validación en {len(C)}: cadenas calculadas = contadas en {np.mean(np.isclose(C.cadenas_calc, C.cadenas_contadas)):.0%}; "
                  f"Li de la ruta = contados en {np.mean(C.dropna(subset=['li_ruta']).li_ruta == C.dropna(subset=['li_ruta']).li_contados):.0%} de {C.li_ruta.notna().sum()} con nº en la ruta; átomos leídos = cabecera en {C.atomos_ok.mean():.0%}")
        for c in ("n_atomos", "n_li", "n_tfsi", "n_cadenas", "lx_A", "volumen_nm3"):
            x = E[c].dropna(); print(f"  {c}: mediana {x.median():g}, rango {x.min():g}–{x.max():g}")
