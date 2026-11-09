"""
baci_comun.py - Modulo compartido (no se ejecuta directamente).

Datos BACI (CEPII), filtros de producto, grafo dirigido exportador -> importador,
relevancia (PageRank / HITS) y utilidades usadas por:
  calcular_relevancia.py   calcula las metricas por anio y seleccion de producto -> CSV
  graficar_relevancia.py   ranking, barras, histograma y evolucion a partir de ese CSV
  ver_red.py               red dirigida de un anio

Metricas de relevancia:
  pagerank       relevancia como MERCADO (comprador)
  pagerank_prov  relevancia como PROVEEDOR (PageRank sobre la red invertida)
  hub            HITS hub: buen exportador (vende a buenas autoridades)
  autoridad      HITS autoridad: buen importador (compra a buenos hubs)
El peso de cada arista es el VALOR comercial (columna v de BACI, miles de USD).
"""
import re
import sys
import textwrap
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

METRICAS = ["pagerank", "pagerank_prov", "hub", "autoridad"]

ORDENES = METRICAS + ["comercio", "exportaciones", "importaciones"]

ETIQUETA = {
    "pagerank": "PageRank (mercado)",
    "pagerank_prov": "PageRank (proveedor)",
    "hub": "HITS hub (exportador)",
    "autoridad": "HITS autoridad (importador)",
    "comercio": "comercio total",
    "exportaciones": "exportaciones",
    "importaciones": "importaciones",
}

SECTORES = {
    "animales":     ("I. Animales vivos y productos animales", range(1, 6)),
    "vegetales":    ("II. Productos vegetales", range(6, 15)),
    "grasas":       ("III. Grasas y aceites", range(15, 16)),
    "alimentos":    ("IV. Alimentos preparados, bebidas y tabaco", range(16, 25)),
    "minerales":    ("V. Productos minerales (incluye combustibles, cap. 27)", range(25, 28)),
    "quimicos":     ("VI. Productos quimicos", range(28, 39)),
    "plasticos":    ("VII. Plasticos y caucho", range(39, 41)),
    "pieles":       ("VIII. Pieles y cueros", range(41, 44)),
    "madera":       ("IX. Madera y corcho", range(44, 47)),
    "papel":        ("X. Pulpa, papel y cartón", range(47, 50)),
    "textiles":     ("XI. Textiles y confeccion", range(50, 64)),
    "calzado":      ("XII. Calzado, sombreros y paraguas", range(64, 68)),
    "piedra":       ("XIII. Piedra, cemento, ceramica y vidrio", range(68, 71)),
    "preciosos":    ("XIV. Perlas, gemas y metales preciosos", range(71, 72)),
    "metales":      ("XV. Metales comunes y sus manufacturas", range(72, 84)),
    "maquinaria":   ("XVI. Maquinaria y aparatos electricos", range(84, 86)),
    "transporte":   ("XVII. Material de transporte", range(86, 90)),
    "instrumentos": ("XVIII. Instrumentos opticos, medicos y de precision", range(90, 93)),
    "armas":        ("XIX. Armas y municiones", range(93, 94)),
    "diversos":     ("XX. Manufacturas diversas", range(94, 97)),
    "arte":         ("XXI. Objetos de arte y antiguedades", range(97, 100)),
}

# Columna de la tabla de relevancia que corresponde a cada criterio de orden
COL_ORDEN = {"comercio": "comercio_total_kUSD", "exportaciones": "exportaciones_kUSD",
             "importaciones": "importaciones_kUSD"}


def sin_acentos(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def listar_sectores():
    print("Sectores disponibles (--sector NOMBRE ...):\n")
    for clave, (desc, caps) in SECTORES.items():
        print(f"  {clave:<13} {desc}  [cap. {caps[0]:02d}-{caps[-1]:02d}]")


def etiqueta_metrica(m, alpha=0.85):
    """Nombre de la metrica para titulos de graficas, con su parametro principal."""
    if m == "pagerank":
        return f"PageRank mercado (α={alpha:g})"
    if m == "pagerank_prov":
        return f"PageRank proveedor (α={alpha:g})"
    if m == "hub":
        return "HITS hub (exportador)"
    if m == "autoridad":
        return "HITS autoridad (importador)"
    return {"comercio": "comercio total (miles USD)", "exportaciones": "exportaciones (miles USD)",
            "importaciones": "importaciones (miles USD)"}[m]


def parametros(alpha, min_flujo, *partes):
    """Linea de parametros que se agrega al titulo de cada grafica."""
    umbral = f"flujo mínimo = {min_flujo:,.0f} miles USD" if min_flujo else "sin flujo mínimo"
    txt = " · ".join([*partes, f"α PageRank = {alpha:g}",
                      "peso de aristas = valor comercial (miles USD)", umbral])
    return textwrap.fill(txt, width=150)


def cargar_paises(ruta):
    df = pd.read_csv(ruta, dtype={"country_iso2": str, "country_iso3": str})
    return df.set_index("country_code")


def cargar_productos(ruta):
    if not Path(ruta).exists():
        return None
    p = pd.read_csv(ruta, dtype={"code": str})
    p["code"] = p["code"].str.zfill(6)
    return p


def construir_prefijos(args, productos):
    """Devuelve (set de prefijos HS, lista de textos descriptivos del filtro)."""
    prefijos, desc = set(), []

    for s in args.sector or []:
        clave = sin_acentos(s.lower())
        if clave not in SECTORES:
            sys.exit(f"Sector '{s}' no existe. Usa --listar-sectores para ver los validos.")
        prefijos.update(f"{c:02d}" for c in SECTORES[clave][1])
        desc.append(f"sector {clave}")

    for h in args.hs or []:
        if not h.isdigit() or len(h) > 6:
            sys.exit(f"Codigo HS invalido: '{h}' (usa 1 a 6 digitos).")
        h = h.zfill(2) if len(h) == 1 else h
        prefijos.add(h)
        desc.append(f"HS {h}")
        if productos is not None:
            m = productos[productos["code"].str.startswith(h)]
            if len(m) == 1:
                print(f"  HS {h}: {m.iloc[0]['description']}")
            else:
                print(f"  HS {h}: {len(m)} productos de 6 digitos")

    for texto in args.buscar or []:
        if productos is None:
            sys.exit("--buscar necesita el archivo de productos (--productos).")
        m = productos[productos["description"].str.contains(texto, case=False, regex=False)]
        if m.empty:
            sys.exit(f"Ningun producto contiene '{texto}' en su descripcion (las descripciones estan en ingles).")
        prefijos.update(m["code"])
        desc.append(f"'{texto}'")
        print(f"  Busqueda '{texto}': {len(m)} productos. Ejemplos:")
        for c, d in m.head(5)[["code", "description"]].itertuples(index=False):
            print(f"     {c}  {d[:90]}")
    return prefijos, desc


def cargar_comercio(ruta):
    """Lee BACI (t,i,j,k,v,q). k se lee como entero (HS 6 digitos, p.ej. 080620 -> 80620):
    usa mucha menos memoria que texto, importante con anios recientes (decenas de millones de filas)."""
    return pd.read_csv(ruta, usecols=["i", "j", "k", "v"],
                       dtype={"i": np.int32, "j": np.int32, "k": np.int32, "v": np.float64})


def filtrar_producto(df, prefijos):
    """Filtra por prefijos HS (cadenas de 2, 4 o 6 digitos); conjunto vacio = todos los productos."""
    if not prefijos:
        return df
    k = df["k"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for L in sorted({len(p) for p in prefijos}):
        mask |= np.isin(k // 10 ** (6 - L), [int(p) for p in prefijos if len(p) == L])
    return df[mask]


def agregar_pares(df):
    return df.groupby(["i", "j"], as_index=False)["v"].sum()


def construir_grafo(pares, paises):
    G = nx.DiGraph()
    for i, j, v in pares.itertuples(index=False):
        if i == j or v <= 0:
            continue
        G.add_edge(int(i), int(j), weight=float(v))
    for n in G.nodes:
        exp = G.out_degree(n, weight="weight")
        imp = G.in_degree(n, weight="weight")
        if n in paises.index:
            iso3, nombre = paises.at[n, "country_iso3"], paises.at[n, "country_name"]
        else:
            iso3, nombre = str(n), str(n)
        G.nodes[n].update(
            iso3=iso3, nombre=nombre, exportaciones=exp, importaciones=imp,
            comercio=exp + imp, balanza=exp - imp,
            balanza_rel=(exp - imp) / (exp + imp) if (exp + imp) > 0 else 0.0,
        )
    return G


def resolver_pais(texto, paises):
    t = texto.strip()
    if t.isdigit() and int(t) in paises.index:
        return int(t)
    m = paises[(paises["country_iso3"].str.upper() == t.upper())
               | (paises["country_iso2"].str.upper() == t.upper())]
    if m.empty:
        m = paises[paises["country_name"].str.contains(t, case=False, regex=False)]
    if m.empty:
        sys.exit(f"No encontre el pais '{texto}'. Usa ISO3 (p.ej. MEX), ISO2 o parte del nombre.")
    if len(m) > 1:
        print("Varias coincidencias, uso la primera:")
        print(m[["country_name", "country_iso3"]].to_string())
    return int(m.index[0])


def anadir_relevancia(G, alpha=0.85):
    """Anade a cada nodo pagerank, pagerank_prov, hub y autoridad (todos suman 1)."""
    pr = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=500)
    pr_prov = nx.pagerank(G.reverse(copy=True), alpha=alpha, weight="weight", max_iter=500)
    hubs, auts = nx.hits(G, max_iter=2000, tol=1e-10, normalized=True)  # usa 'weight'
    for n in G.nodes:
        G.nodes[n].update(pagerank=pr[n], pagerank_prov=pr_prov[n], hub=hubs[n], autoridad=auts[n])


def tabla_relevancia(G, ordenar_por):
    filas = []
    for n, d in G.nodes(data=True):
        filas.append({
            "iso3": d["iso3"], "pais": d["nombre"],
            "pagerank": d["pagerank"], "pagerank_prov": d["pagerank_prov"],
            "hub": d["hub"], "autoridad": d["autoridad"],
            "exportaciones_kUSD": d["exportaciones"], "importaciones_kUSD": d["importaciones"],
            "comercio_total_kUSD": d["comercio"], "balanza_kUSD": d["balanza"],
            "balanza_rel": d["balanza_rel"],
            "n_clientes": G.out_degree(n), "n_proveedores": G.in_degree(n),
        })
    df = pd.DataFrame(filas)
    for m in METRICAS:                       # posicion en cada ranking (1 = mas relevante)
        df[f"rank_{m}"] = df[m].rank(ascending=False, method="min").astype(int)
    col = {"comercio": "comercio_total_kUSD", "exportaciones": "exportaciones_kUSD",
           "importaciones": "importaciones_kUSD"}.get(ordenar_por, ordenar_por)
    df = df.sort_values(col, ascending=False).reset_index(drop=True)
    df.insert(0, "posicion", df.index + 1)
    return df


def imprimir_resumen(G, titulo):
    total = sum(w for _, _, w in G.edges(data="weight"))
    print(f"\n=== {titulo} ===")
    print(f"Paises: {G.number_of_nodes()} | Aristas dirigidas: {G.number_of_edges()} "
          f"| Comercio total: {total / 1e6:,.1f} miles de millones USD")
    print(f"Densidad: {nx.density(G):.3f} | Reciprocidad: {nx.reciprocity(G):.3f}")


def imprimir_ranking(df, ordenar_por, top):
    print(f"\nRanking de relevancia, ordenado por {ETIQUETA[ordenar_por]} (top {top}).")
    print("Valores de PageRank y HITS en %; entre parentesis, su posicion en cada ranking.\n")
    filas = []
    for r in df.head(top).itertuples(index=False):
        filas.append({
            "#": r.posicion, "ISO3": r.iso3, "Pais": r.pais[:26],
            "PR mercado": f"{r.pagerank * 100:5.2f} ({r.rank_pagerank})",
            "PR proveedor": f"{r.pagerank_prov * 100:5.2f} ({r.rank_pagerank_prov})",
            "Hub": f"{r.hub * 100:5.2f} ({r.rank_hub})",
            "Autoridad": f"{r.autoridad * 100:5.2f} ({r.rank_autoridad})",
            "Comercio (miles MM USD)": f"{r.comercio_total_kUSD / 1e6:,.1f}",
        })
    print(pd.DataFrame(filas).to_string(index=False))


def parsear_anios(spec):
    anios = []
    for t in spec:
        if "-" in t:
            x, y = t.split("-", 1)
            anios += range(int(x), int(y) + 1)
        else:
            anios.append(int(t))
    return sorted(set(anios))


def localizar_archivos(a):
    """Devuelve {anio: ruta}. Con --anios busca BACI_HS92_Y<anio>_V*.csv en --carpeta."""
    if a.anios:
        archivos, faltan = {}, []
        for y in parsear_anios(a.anios):
            m = sorted(Path(a.carpeta).glob(f"BACI_HS92_Y{y}_V*.csv"))
            if m:
                archivos[y] = m[-1]
            else:
                faltan.append(y)
        if faltan:
            txt = str(faltan) if len(faltan) <= 6 else f"{faltan[:3]} ... {faltan[-2:]} ({len(faltan)} en total)"
            sys.exit(f"No encontre en '{a.carpeta}' los anios {txt}. "
                     "Espero archivos con nombre BACI_HS92_Y<anio>_V*.csv (usa --carpeta).")
        return archivos
    ruta = Path(a.baci)
    if not ruta.exists():
        sys.exit(f"No encuentro el archivo: {ruta} (usa --baci, o --anios con --carpeta)")
    m = re.search(r"_Y(\d{4})_", ruta.name)
    return {int(m.group(1)) if m else 0: ruta}


def parsear_seleccion(spec, productos):
    """'todos' | 'sector:a,b' | 'hs:27,8703' | 'buscar:coffee' -> (nombre, set de prefijos).
    Se le puede poner nombre propio con 'nombre=seleccion', p.ej. 'alimentos=hs:02,03,04'."""
    spec = spec.strip()
    nombre_propio = None
    if "=" in spec and ":" in spec.split("=", 1)[1] and ":" not in spec.split("=", 1)[0]:
        nombre_propio, spec = (x.strip() for x in spec.split("=", 1))
    if spec.lower() in ("todos", "all"):
        return nombre_propio or "todos los productos", set()
    if ":" not in spec:
        sys.exit(f"Seleccion invalida '{spec}'. Formatos: todos | sector:NOMBRE | hs:COD | buscar:TEXTO "
                 "(opcional: nombre=seleccion)")
    tipo, val = spec.split(":", 1)
    items = [v.strip() for v in val.split(",") if v.strip()]
    ns = SimpleNamespace(sector=None, hs=None, buscar=None)
    tipo = tipo.lower()
    if tipo not in ("sector", "hs", "buscar"):
        sys.exit(f"Tipo de seleccion '{tipo}' desconocido (usa sector, hs o buscar).")
    setattr(ns, tipo, items)
    pref, desc = construir_prefijos(ns, productos)
    return nombre_propio or ", ".join(desc), pref


def slug(texto):
    return re.sub(r"[^a-z0-9]+", "_", sin_acentos(texto.lower())).strip("_")[:40] or "seleccion"


def ruta_salida(base, nombre_sel, varias):
    base = Path(base)
    return str(base.with_name(f"{base.stem}_{slug(nombre_sel)}{base.suffix}")) if varias else str(base)


def grafo_relevancia(df, prefijos, paises, alpha, min_flujo):
    """Filtra producto, agrega, construye el grafo y calcula PageRank/HITS. None si queda vacio."""
    sub = filtrar_producto(df, prefijos)
    if sub.empty:
        return None
    pares = agregar_pares(sub)
    if min_flujo:
        pares = pares[pares["v"] >= min_flujo]
    if pares.empty:
        return None
    G = construir_grafo(pares, paises)
    anadir_relevancia(G, alpha)
    return G


# --------------------------------------------------------------------------- #
# Argumentos compartidos por los programas
# --------------------------------------------------------------------------- #
def args_datos(p):
    g = p.add_argument_group("datos BACI")
    g.add_argument("--paises", default="country_codes_V202601.csv", help="CSV de codigos de pais")
    g.add_argument("--productos", default="product_codes_HS92_V202601.csv", help="CSV de codigos de producto")
    g.add_argument("--carpeta", default=".", help="Carpeta con los BACI_HS92_Y<anio>_V*.csv (con --anios)")
    g.add_argument("--anios", nargs="+", metavar="A", help="Rango o lista: 1995-2024  |  1995 2007 2024")
    g.add_argument("--baci", default="BACI_HS92_Y1995_V202601.csv", help="CSV de un anio (si no usas --anios)")


def args_producto(p):
    g = p.add_argument_group("seleccion de producto (sin filtro = todos los productos)")
    g.add_argument("--sector", nargs="+", metavar="NOMBRE", help="Sector(es) HS; ver --listar-sectores")
    g.add_argument("--hs", nargs="+", metavar="COD", help="Prefijos HS de 2, 4 o 6 digitos")
    g.add_argument("--buscar", nargs="+", metavar="TEXTO", help="Texto en la descripcion del producto (en ingles)")
    g.add_argument("--comparar", nargs="+", metavar="SEL",
                   help="Varias selecciones: todos | sector:x,y | hs:27 | buscar:coffee, con nombre opcional (alimentos=hs:02,03). Reemplaza a --sector/--hs/--buscar")
    g.add_argument("--listar-sectores", action="store_true", help="Muestra los sectores y termina")


def args_calculo(p):
    g = p.add_argument_group("calculo de relevancia")
    g.add_argument("--alpha", type=float, default=0.85, help="Amortiguamiento de PageRank (default 0.85)")
    g.add_argument("--min-flujo", type=float, default=0,
                   help="Descarta flujos pais->pais menores a este valor (kUSD) antes de calcular")


def preparar_datos(a):
    """Procesa --listar-sectores y devuelve (paises, productos, archivos{anio: ruta}, selecciones)."""
    if a.listar_sectores:
        listar_sectores()
        sys.exit(0)
    if not Path(a.paises).exists():
        sys.exit(f"No encuentro el archivo: {a.paises} (usa --paises)")
    paises = cargar_paises(a.paises)
    productos = cargar_productos(a.productos)
    archivos = localizar_archivos(a)
    if a.comparar:
        selecciones = [parsear_seleccion(sp, productos) for sp in a.comparar]
    else:
        pref, desc = construir_prefijos(a, productos)
        selecciones = [(", ".join(desc) if desc else "todos los productos", pref)]
    return paises, productos, archivos, selecciones
