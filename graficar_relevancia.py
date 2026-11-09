#!/usr/bin/env python3
"""
graficar_relevancia.py - Ranking y graficos de relevancia a partir del CSV de calcular_relevancia.py.
No recalcula nada: lee el historial y dibuja (segundos).

Subcomandos:
  ranking      tabla ordenada de relevancia en consola (y opcional CSV)
  barras       barras con los N paises mas relevantes por metrica
  histograma   distribucion de cada metrica entre paises (escala log) + concentracion
  evolucion    evolucion anual de la relevancia (necesita varios anios en el historial)

Cada titulo incluye metrica, producto, anio(s) y parametros de calculo (alpha, flujo minimo, peso).

Ejemplos:
  python graficar_relevancia.py ranking    --ordenar-por hub --top 30
  python graficar_relevancia.py barras     --anio 2024 --top 15
  python graficar_relevancia.py histograma --anio 2010 --metrica pagerank
  python graficar_relevancia.py evolucion  --paises-lista USA CHN DEU MEX --metrica autoridad --eje posicion
  python graficar_relevancia.py evolucion  --historial otro.csv --seleccion maquinaria --salida evo.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from baci_comun import (COL_ORDEN, METRICAS, ORDENES, etiqueta_metrica, imprimir_ranking, parametros,
                        ruta_salida)

COLORES_MET = {"pagerank": "#1f77b4", "pagerank_prov": "#2ca02c", "hub": "#ff7f0e", "autoridad": "#d62728"}

NBINS = 21


def _rejilla(n):
    return (1, 1) if n == 1 else ((1, 2) if n == 2 else (2, 2))


def _encabezado(fig, titulo, params):
    fig.suptitle(titulo, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965, params, ha="center", va="top", fontsize=8.5, color="#333333")


def _cerrar(fig, salida, mostrar):
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    if salida:
        fig.savefig(salida, dpi=150, bbox_inches="tight")
        print(f"Imagen guardada en: {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


def grafico_barras(tabla, titulo, params, alpha, salida, mostrar, metricas, top):
    f, c = _rejilla(len(metricas))
    fig, axs = plt.subplots(f, c, figsize=(7 * c, 0.33 * top * f + 3 * f), squeeze=False)
    for ax, m in zip(axs.flat, metricas):
        d = tabla.nlargest(top, m)[::-1]
        vals = d[m].to_numpy() * 100
        ax.barh(d["iso3"], vals, color=COLORES_MET[m], edgecolor="black", linewidth=0.5)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8)
        ax.set_title(etiqueta_metrica(m, alpha), fontsize=11)
        ax.set_xlabel("% del total de la métrica")
        ax.margins(x=0.15)
    for ax in axs.flat[len(metricas):]:
        ax.axis("off")
    _encabezado(fig, titulo, params)
    _cerrar(fig, salida, mostrar)


def grafico_histograma(tabla, titulo, params, alpha, salida, mostrar, metricas):
    f, c = _rejilla(len(metricas))
    fig, axs = plt.subplots(f, c, figsize=(7 * c, 4.6 * f), squeeze=False)
    for ax, m in zip(axs.flat, metricas):
        v = tabla[m].to_numpy() * 100
        v = v[v > 0]
        if len(v) < 2:
            ax.text(0.5, 0.5, "datos insuficientes", ha="center", transform=ax.transAxes)
            continue
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), NBINS + 1)
        ax.hist(v, bins=bins, color=COLORES_MET[m], edgecolor="black", linewidth=0.6)
        ax.set_xscale("log")
        top5 = tabla.nlargest(5, m)
        ax.set_title(f"{etiqueta_metrica(m, alpha)}\nlos 5 primeros concentran {top5[m].sum() * 100:.0f}% "
                     f"· mediana {np.median(v):.3f}%", fontsize=10)
        ax.set_xlabel("valor de la métrica por país (% del total, escala log)")
        ax.set_ylabel("n.º de países")
        ax.text(0.97, 0.95, "Top: " + ", ".join(top5["iso3"].head(4)), transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    for ax in axs.flat[len(metricas):]:
        ax.axis("off")
    _encabezado(fig, titulo, params)
    _cerrar(fig, salida, mostrar)


def grafico_evolucion(hist, titulo, params, alpha, salida, mostrar, metricas, top, iso_lista, eje):
    """hist: tabla larga (anio, iso3, metricas...) de UNA seleccion de producto."""
    seleccion = {}
    for m in metricas:
        piv = hist.pivot_table(index="anio", columns="iso3", values=m)
        cols = [c for c in iso_lista if c in piv.columns] if iso_lista else list(piv.mean().nlargest(top).index)
        seleccion[m] = (piv, cols)
    todos = sorted({c for _, cols in seleccion.values() for c in cols})
    paleta = cm.tab20(np.linspace(0, 1, 20))
    color = {c: paleta[i % 20] for i, c in enumerate(todos)}

    f, c = _rejilla(len(metricas))
    fig, axs = plt.subplots(f, c, figsize=(8 * c, 5.3 * f), squeeze=False)
    for ax, m in zip(axs.flat, metricas):
        piv, cols = seleccion[m]
        if eje == "posicion":
            datos = piv.rank(axis=1, ascending=False, method="min")[cols]
        else:
            datos = piv[cols] * 100
        for col in cols:
            ax.plot(datos.index, datos[col], marker="o", ms=3, lw=1.8, color=color[col], label=col)
        if eje == "posicion":
            ax.invert_yaxis()
            ax.set_ylabel("posición en el ranking (1 = más relevante)")
        else:
            ax.set_ylabel("% del total de la métrica")
        ax.set_title(etiqueta_metrica(m, alpha), fontsize=11)
        ax.set_xlabel("año")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    for ax in axs.flat[len(metricas):]:
        ax.axis("off")
    _encabezado(fig, titulo, params)
    _cerrar(fig, salida, mostrar)


# --------------------------------------------------------------------------- #
# Lectura del historial y programa principal
# --------------------------------------------------------------------------- #
COLUMNAS = ["anio", "seleccion", "iso3", "pais", "alpha", "min_flujo_kUSD"] + METRICAS


def cargar_historial(ruta):
    if not Path(ruta).exists():
        sys.exit(f"No encuentro {ruta}. Genera primero el historial con calcular_relevancia.py (o usa --historial).")
    h = pd.read_csv(ruta)
    faltan = [c for c in COLUMNAS if c not in h.columns]
    if faltan:
        sys.exit(f"{ruta} no parece un historial de calcular_relevancia.py (faltan columnas: {faltan}).")
    return h


def elegir_selecciones(hist, texto):
    todas = list(dict.fromkeys(hist["seleccion"]))
    if not texto:
        return todas
    ex = [s for s in todas if s.lower() == texto.lower()]
    sub = ex or [s for s in todas if texto.lower() in s.lower()]
    if not sub:
        sys.exit(f"Ninguna seleccion coincide con '{texto}'. Disponibles: {todas}")
    return sub


def resolver_iso(texto, hist):
    """ISO3, ISO2 aproximado o parte del nombre -> iso3 presente en el historial."""
    t = texto.strip()
    tabla = hist[["iso3", "pais"]].drop_duplicates()
    m = tabla[tabla["iso3"].str.upper() == t.upper()]
    if m.empty:
        m = tabla[tabla["pais"].str.contains(t, case=False, regex=False)]
    if m.empty:
        sys.exit(f"No encuentro el pais '{texto}' en el historial (usa ISO3, p.ej. MEX, o parte del nombre).")
    return m.iloc[0]["iso3"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{ranking,barras,histograma,evolucion}")

    def comunes(sp, con_anio=True):
        sp.add_argument("--historial", default="historial_relevancia.csv", help="CSV de calcular_relevancia.py")
        sp.add_argument("--seleccion", help="Seleccion de producto (nombre o parte); default: todas las del historial")
        if con_anio:
            sp.add_argument("--anio", type=int, help="Anio a mostrar (default: el ultimo del historial)")
        sp.add_argument("--top", type=int, default=20, help="Paises a mostrar (default 20)")
        sp.add_argument("--salida", help="Archivo de salida (con varias selecciones se anade sufijo)")
        sp.add_argument("--no-mostrar", action="store_true", help="No abrir ventana de matplotlib")

    r = sub.add_parser("ranking", help="Tabla de relevancia ordenada")
    comunes(r)
    r.add_argument("--ordenar-por", choices=ORDENES, default="pagerank")
    r.add_argument("--csv", help="Guarda el ranking completo en este CSV")

    for nombre, ayuda in (("barras", "Barras de relevancia"), ("histograma", "Distribucion de la relevancia")):
        b = sub.add_parser(nombre, help=ayuda)
        comunes(b)
        b.add_argument("--metrica", choices=METRICAS, help="Una sola metrica (default: las cuatro)")

    e = sub.add_parser("evolucion", help="Evolucion anual de la relevancia")
    comunes(e, con_anio=False)
    e.add_argument("--metrica", choices=METRICAS, help="Una sola metrica (default: las cuatro)")
    e.add_argument("--paises-lista", nargs="+", metavar="PAIS", help="Solo estos paises (ISO3 o nombre); default: top --top")
    e.add_argument("--eje", choices=["valor", "posicion"], default="valor", help="Valor de la metrica (%%) o posicion")
    return p.parse_args()


def main():
    a = parse_args()
    hist = cargar_historial(a.historial)
    anios = sorted(int(x) for x in hist["anio"].unique())
    alpha = float(hist["alpha"].iloc[0])
    min_flujo = float(hist["min_flujo_kUSD"].iloc[0])
    selecciones = elegir_selecciones(hist, a.seleccion)
    varias = len(selecciones) > 1
    metricas = [a.metrica] if getattr(a, "metrica", None) else METRICAS
    mostrar = not a.no_mostrar
    base = a.salida or ("ranking.csv" if a.cmd == "ranking" else f"{a.cmd}.png")
    mets_txt = "métricas: " + ", ".join(etiqueta_metrica(m, alpha) for m in metricas)

    if a.cmd == "evolucion":
        if len(anios) < 2:
            sys.exit("'evolucion' necesita varios anios en el historial; este solo tiene " + str(anios))
        iso_lista = [resolver_iso(x, hist) for x in a.paises_lista] if a.paises_lista else None
    else:
        anio = a.anio or anios[-1]
        if anio not in anios:
            sys.exit(f"--anio {anio} no esta en el historial {anios[0]}-{anios[-1]} (anios: {anios}).")

    for nombre in selecciones:
        sub = hist[hist["seleccion"] == nombre]
        etiq_prod = "todos los productos" if nombre == "todos los productos" else f"producto: {nombre}"
        salida = ruta_salida(base, nombre, varias)

        if a.cmd == "evolucion":
            titulo = f"Evolución de la relevancia comercial · {anios[0]}–{anios[-1]} · {etiq_prod}"
            params = parametros(
                alpha, min_flujo, f"años: {len(anios)} ({anios[0]}–{anios[-1]})", mets_txt,
                "eje Y = " + ("posición en el ranking" if a.eje == "posicion" else "% del total de la métrica"),
                ("países: " + ", ".join(iso_lista)) if iso_lista else f"países: top {a.top} por promedio de la métrica")
            grafico_evolucion(sub, titulo, params, alpha, salida, mostrar, metricas, a.top, iso_lista, a.eje)
            continue

        tabla = sub[sub["anio"] == anio]
        if tabla.empty:
            print(f"[aviso] '{nombre}': sin datos en {anio}.")
            continue
        orden = COL_ORDEN.get(getattr(a, "ordenar_por", "pagerank"), getattr(a, "ordenar_por", "pagerank"))
        tabla = tabla.sort_values(orden, ascending=False).reset_index(drop=True)
        tabla["posicion"] = tabla.index + 1

        if a.cmd == "ranking":
            print(f"\n=== {anio} | {etiq_prod} | α={alpha:g}, flujo mínimo={min_flujo:g} kUSD ===")
            imprimir_ranking(tabla, a.ordenar_por, a.top)
            if a.csv:
                ruta = ruta_salida(a.csv, nombre, varias)
                tabla.to_csv(ruta, index=False)
                print(f"\nRanking completo guardado en: {ruta}")
        elif a.cmd == "barras":
            titulo = f"Relevancia de los países · {anio} · {etiq_prod}"
            params = parametros(alpha, min_flujo, f"año {anio}", mets_txt, f"top {a.top} países por métrica")
            grafico_barras(tabla, titulo, params, alpha, salida, mostrar, metricas, a.top)
        else:
            titulo = f"Distribución de la relevancia entre países · {anio} · {etiq_prod}"
            params = parametros(alpha, min_flujo, f"año {anio}", mets_txt,
                                f"{NBINS} intervalos logarítmicos", f"{len(tabla)} países")
            grafico_histograma(tabla, titulo, params, alpha, salida, mostrar, metricas)


if __name__ == "__main__":
    main()
