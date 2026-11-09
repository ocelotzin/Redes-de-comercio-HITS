#!/usr/bin/env python3
"""
ver_red.py - Red dirigida de comercio (exportador -> importador) de UN anio, con networkx.

Tamano y color de los nodos: comercio, balanza o cualquier metrica de relevancia
(PageRank mercado/proveedor, HITS hub/autoridad). Muestra ademas un ranking en consola.

Ejemplos:
  python ver_red.py                                          # global, 1995 (--baci)
  python ver_red.py --anios 2024 --carpeta datos --sector maquinaria --tam-por hub --color-por autoridad
  python ver_red.py --anios 2024 --carpeta datos --pais MEX --sector transporte
  python ver_red.py --hs 27 --ordenar-por hub --top-nodes 25 --top-edges 80 --interactivo
  python ver_red.py --anios 2024 --carpeta datos --comparar todos sector:textiles
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from baci_comun import (METRICAS, ORDENES, args_calculo, args_datos, args_producto,
                        cargar_comercio, etiqueta_metrica, grafo_relevancia, imprimir_ranking,
                        imprimir_resumen, parametros, preparar_datos, resolver_pais, ruta_salida,
                        tabla_relevancia)


def filtrar_global(G, top_nodes, top_edges, min_valor, ordenar_por):
    if top_nodes:
        elegidos = sorted(G.nodes, key=lambda n: G.nodes[n][ordenar_por], reverse=True)[:top_nodes]
        H = G.subgraph(elegidos).copy()
    else:
        H = G.copy()
    aristas = sorted(H.edges(data="weight"), key=lambda e: e[2], reverse=True)
    if min_valor:
        aristas = [e for e in aristas if e[2] >= min_valor]
    if top_edges:
        aristas = aristas[:top_edges]
    conservar = {(u, v) for u, v, _ in aristas}
    H.remove_edges_from([e for e in list(H.edges) if e not in conservar])
    H.remove_nodes_from(list(nx.isolates(H)))
    return H


def red_ego(G, centro, top_partners):
    sal = sorted(G.out_edges(centro, data="weight"), key=lambda e: e[2], reverse=True)[:top_partners]
    ent = sorted(G.in_edges(centro, data="weight"), key=lambda e: e[2], reverse=True)[:top_partners]
    H = nx.DiGraph()
    for n in {centro} | {v for _, v, _ in sal} | {u for u, _, _ in ent}:
        H.add_node(n, **G.nodes[n])
    for u, v, w in sal + ent:
        H.add_edge(u, v, weight=w)
    return H


def dibujar(H, titulo, params, alpha, salida, mostrar, layout, tam_por, color_por, centro=None, semilla=42, dpi=160):
    fig, ax = plt.subplots(figsize=(15, 11))
    nodos = list(H.nodes)

    G_lay = nx.Graph()
    G_lay.add_nodes_from(nodos)
    for u, v, w in H.edges(data="weight"):
        a = G_lay.get_edge_data(u, v, {"weight": 0})["weight"]
        G_lay.add_edge(u, v, weight=a + np.log10(w + 1))
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(G_lay, weight=None)
    elif layout == "circular":
        orden = sorted(nodos, key=lambda n: H.nodes[n][tam_por], reverse=True)
        pos = nx.circular_layout(G_lay, scale=1)
        pos = dict(zip(orden, [pos[n] for n in nodos]))
    elif layout == "shell" and centro is not None:
        pos = nx.shell_layout(H, nlist=[[centro], [n for n in nodos if n != centro]])
    else:
        pos = nx.spring_layout(G_lay, weight="weight", k=2.4 / np.sqrt(max(len(H), 1)),
                               iterations=300, seed=semilla)

    # Tamano
    val = np.array([H.nodes[n][tam_por] for n in nodos], dtype=float)
    tam = 250 + 3000 * np.sqrt(val / val.max()) if val.max() > 0 else np.full(len(nodos), 400.0)

    # Color
    if color_por == "balanza":
        LIM = 0.5
        cmap, norm = cm.coolwarm_r, Normalize(-LIM, LIM)
        cvals = np.array([H.nodes[n]["balanza_rel"] for n in nodos])
        cb_label = (f"Balanza relativa (X − M) / (X + M), saturada en ±{LIM}\n"
                    "azul = exportador neto  |  rojo = importador neto")
    else:
        cmap = cm.YlOrRd
        cvals = np.array([H.nodes[n][color_por] for n in nodos], dtype=float)
        norm = Normalize(0, cvals.max() if cvals.max() > 0 else 1)
        cb_label = etiqueta_metrica(color_por, alpha) + "\n(fracción del total, suma 1)"
    nx.draw_networkx_nodes(H, pos, nodelist=nodos, node_size=tam, node_color=cvals, cmap=cmap,
                           vmin=norm.vmin, vmax=norm.vmax, edgecolors="black", linewidths=0.8, ax=ax)

    # Aristas
    pesos = np.array([w for _, _, w in H.edges(data="weight")], dtype=float)
    if len(pesos):
        lw = np.log10(pesos + 1)
        rel = (lw - lw.min()) / (np.ptp(lw) + 1e-9)
        for (u, v), w, a in zip(H.edges, 0.4 + 5.5 * rel, 0.25 + 0.5 * rel):
            nx.draw_networkx_edges(H, pos, edgelist=[(u, v)], width=w, alpha=float(a),
                                   edge_color="#555555", arrows=True, arrowstyle="-|>", arrowsize=14,
                                   connectionstyle="arc3,rad=0.12",
                                   node_size=tam[nodos.index(u)], ax=ax)

    nx.draw_networkx_labels(H, pos, {n: H.nodes[n]["iso3"] for n in nodos},
                            font_size=8, font_weight="bold", ax=ax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.01)
    cb.set_label(cb_label)
    ax.set_title(f"{titulo}\n{params}\n"
                 "grosor de flecha = valor comercial (miles USD)  ·  flecha: exportador → importador",
                 fontsize=10.5)
    ax.axis("off")
    fig.tight_layout()
    if salida:
        fig.savefig(salida, dpi=dpi, bbox_inches="tight")
        print(f"\nImagen guardada en: {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


def exportar_pyvis(H, titulo, ruta, tam_por, color_por):
    try:
        from pyvis.network import Network
    except ImportError:
        print("\n[aviso] pyvis no esta instalado: pip install pyvis  (omito la version interactiva)")
        return
    net = Network(height="900px", width="100%", directed=True, bgcolor="#ffffff", notebook=False, heading=titulo)
    maxv = max(d[tam_por] for _, d in H.nodes(data=True))
    if color_por == "balanza":
        cmap, norm, clave = cm.coolwarm_r, Normalize(-0.5, 0.5), "balanza_rel"
    else:
        cmap, clave = cm.YlOrRd, color_por
        norm = Normalize(0, max(d[color_por] for _, d in H.nodes(data=True)))
    for n, d in H.nodes(data=True):
        r, g, b, _ = cmap(norm(d[clave]))
        net.add_node(n, label=d["iso3"], color="#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)),
                     size=10 + 50 * np.sqrt(d[tam_por] / maxv),
                     title=(f"{d['nombre']}<br>Exportaciones: {d['exportaciones']:,.0f} kUSD"
                            f"<br>Importaciones: {d['importaciones']:,.0f} kUSD"
                            f"<br>PageRank mercado: {d['pagerank'] * 100:.2f}%"
                            f"<br>PageRank proveedor: {d['pagerank_prov'] * 100:.2f}%"
                            f"<br>Hub: {d['hub'] * 100:.2f}% | Autoridad: {d['autoridad'] * 100:.2f}%"))
    wmax = max(w for _, _, w in H.edges(data="weight"))
    for u, v, w in H.edges(data="weight"):
        net.add_edge(u, v, value=float(w / wmax) * 10,
                     title=f"{H.nodes[u]['iso3']} -> {H.nodes[v]['iso3']}: {w:,.0f} kUSD")
    net.write_html(str(ruta))
    print(f"HTML interactivo guardado en: {ruta}")


def modo_red(a, G, paises, nombre_sel, anio, salida):
    filtro_txt = "" if nombre_sel == "todos los productos" else f" | producto: {nombre_sel}"
    etiq = f"{anio}" if anio else ""
    imprimir_resumen(G, f"Red completa {etiq}{filtro_txt}")
    tabla = tabla_relevancia(G, a.ordenar_por)
    imprimir_ranking(tabla, a.ordenar_por, a.top_ranking)
    if a.ranking_csv:
        ruta = a.ranking_csv
        tabla.to_csv(ruta, index=False)
        print(f"\nRanking completo guardado en: {ruta}")
    if a.pais:
        centro = resolver_pais(a.pais, paises)
        if centro not in G:
            print("Ese pais no tiene flujos con este filtro de producto.")
            return
        H = red_ego(G, centro, a.top_partners)
        titulo = f"Comercio de {G.nodes[centro]['nombre']} y sus principales socios · {etiq}{filtro_txt}"
        sel_txt = f"socios: top {a.top_partners} por sentido (exportaciones e importaciones)"
        layout, centro_arg = ("shell" if a.layout == "spring" else a.layout), centro
    else:
        H = filtrar_global(G, a.top_nodes, a.top_edges, a.min_valor, a.ordenar_por)
        titulo = (f"Red global de comercio · {etiq}{filtro_txt} "
                  f"({H.number_of_nodes()} países, {H.number_of_edges()} flujos mayores)")
        sel_txt = (f"nodos: top {a.top_nodes or 'todos'} por {etiqueta_metrica(a.ordenar_por, a.alpha)}"
                   f" · aristas: top {a.top_edges or 'todas'}"
                   + (f" · arista mínima = {a.min_valor:,.0f} miles USD" if a.min_valor else ""))
        layout, centro_arg = a.layout, None
    print(f"\nSubgrafo dibujado: {H.number_of_nodes()} nodos, {H.number_of_edges()} aristas")
    params = parametros(a.alpha, a.min_flujo, f"año {anio}" if anio else "año ?", sel_txt,
                        f"tamaño del nodo = {etiqueta_metrica(a.tam_por, a.alpha)}",
                        "color = " + ("balanza relativa" if a.color_por == "balanza"
                                      else etiqueta_metrica(a.color_por, a.alpha)),
                        f"layout = {layout}")
    dibujar(H, titulo, params, a.alpha, salida, not a.no_mostrar, layout, a.tam_por, a.color_por,
            centro_arg, a.semilla)
    if a.interactivo:
        exportar_pyvis(H, titulo, Path(salida).with_suffix(".html"), a.tam_por, a.color_por)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args_datos(p)
    args_producto(p)
    args_calculo(p)
    r = p.add_argument_group("relevancia y ranking")
    r.add_argument("--ordenar-por", choices=ORDENES, default="pagerank",
                   help="Criterio del ranking impreso y de seleccion de nodos (default pagerank)")
    r.add_argument("--top-ranking", type=int, default=20, help="Filas del ranking impreso (default 20)")
    r.add_argument("--ranking-csv", help="Guarda el ranking completo en este CSV")
    v = p.add_argument_group("red")
    v.add_argument("--pais", help="Red de ego de un pais (ISO3, ISO2, codigo o nombre)")
    v.add_argument("--top-partners", type=int, default=12, help="Con --pais: socios por sentido (default 12)")
    v.add_argument("--top-nodes", type=int, default=40, help="Paises a mostrar (0 = todos; default 40)")
    v.add_argument("--top-edges", type=int, default=150, help="Aristas a mostrar (0 = todas; default 150)")
    v.add_argument("--min-valor", type=float, default=0, help="Descarta aristas dibujadas bajo este valor (kUSD)")
    v.add_argument("--tam-por", choices=ORDENES, default="comercio", help="Metrica del tamano del nodo")
    v.add_argument("--color-por", choices=["balanza"] + METRICAS, default="balanza", help="Metrica del color del nodo")
    v.add_argument("--layout", choices=["spring", "kamada", "circular", "shell"], default="spring")
    v.add_argument("--interactivo", action="store_true", help="Ademas HTML interactivo (pyvis)")
    v.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="red_comercio.png", help="PNG de salida (con varias selecciones se anade sufijo)")
    p.add_argument("--no-mostrar", action="store_true", help="No abrir ventana de matplotlib")
    return p.parse_args()


def main():
    a = parse_args()
    paises, _, archivos, selecciones = preparar_datos(a)
    anios = sorted(archivos)
    if len(anios) > 1:
        sys.exit("ver_red.py usa un solo anio. Pasa --anios con un anio (p.ej. --anios 2024).")
    anio = anios[0]
    varias = len(selecciones) > 1
    print(f"Cargando BACI {anio or ''}...")
    df = cargar_comercio(archivos[anio])
    for nombre, pref in selecciones:
        G = grafo_relevancia(df, pref, paises, a.alpha, a.min_flujo)
        if G is None:
            print(f"[aviso] '{nombre}': sin flujos en {anio}.")
            continue
        modo_red(a, G, paises, nombre, anio, ruta_salida(a.salida, nombre, varias))


if __name__ == "__main__":
    main()
