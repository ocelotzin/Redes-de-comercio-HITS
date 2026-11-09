#!/usr/bin/env python3
"""
calcular_relevancia.py - Paso lento: calcula PageRank y HITS por anio y seleccion de producto
y guarda una tabla larga (anio, seleccion, pais, metricas) en CSV.
Los graficos se hacen despues con graficar_relevancia.py, sin recalcular.

Ejemplos:
  python calcular_relevancia.py --anios 1995-2024 --carpeta datos
  python calcular_relevancia.py --anios 1995-2024 --carpeta datos --comparar todos sector:maquinaria hs:27
  python calcular_relevancia.py --baci BACI_HS92_Y1995_V202601.csv --buscar coffee --min-flujo 1000
  python calcular_relevancia.py --listar-sectores
"""
import argparse
import sys

import pandas as pd

from baci_comun import (args_calculo, args_datos, args_producto, cargar_comercio,
                        grafo_relevancia, preparar_datos, tabla_relevancia)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args_datos(p)
    args_producto(p)
    args_calculo(p)
    p.add_argument("--salida", default="historial_relevancia.csv", help="CSV de salida (default historial_relevancia.csv)")
    return p.parse_args()


def main():
    a = parse_args()
    paises, _, archivos, selecciones = preparar_datos(a)
    anios = sorted(archivos)

    tablas = []
    for y in anios:
        print(f"[{y}] cargando {archivos[y].name} ...")
        df = cargar_comercio(archivos[y])
        for nombre, pref in selecciones:
            G = grafo_relevancia(df, pref, paises, a.alpha, a.min_flujo)
            if G is None:
                print(f"   [aviso] '{nombre}': sin flujos en {y}.")
                continue
            t = tabla_relevancia(G, "pagerank").rename(columns={"posicion": "posicion_pagerank"})
            t.insert(0, "anio", y)
            t.insert(1, "seleccion", nombre)
            t["alpha"] = a.alpha                 # parametros de calculo: viajan con la tabla
            t["min_flujo_kUSD"] = a.min_flujo    # para que los graficos los muestren en el titulo
            tablas.append(t)
            print(f"   {nombre}: {G.number_of_nodes()} paises, {G.number_of_edges()} aristas")
        del df
    if not tablas:
        sys.exit("No quedaron datos con esa seleccion de producto.")

    hist = pd.concat(tablas, ignore_index=True)
    hist.to_csv(a.salida, index=False)
    print(f"\nGuardado: {a.salida}  ({len(hist):,} filas | {len(anios)} anio(s) | "
          f"{hist['seleccion'].nunique()} seleccion(es) | alpha={a.alpha:g}, min-flujo={a.min_flujo:g} kUSD)")
    print("Siguiente paso: python graficar_relevancia.py {ranking|barras|histograma|evolucion} --historial " + a.salida)


if __name__ == "__main__":
    main()
