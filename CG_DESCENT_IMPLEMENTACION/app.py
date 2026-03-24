import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Descenso del Gradiente Conjugado",
    page_icon="iconos/png/ico_ruler.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

def cargar_css(archivo_css):
    """Carga un archivo .css externo e inyecta los estilos en Streamlit."""
    ruta = os.path.join(os.path.dirname(__file__), archivo_css)
    with open(ruta, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

cargar_css("styles.css")


import base64

_DIR_BASE = os.path.dirname(__file__)

def _icono_b64(nombre_archivo, ancho=60):
    """Convierte un icono PNG a tag HTML en base64."""
    ruta = os.path.join(_DIR_BASE, "iconos", "png", nombre_archivo)
    if not os.path.exists(ruta):
        return f'<div style="width:{ancho}px;height:{ancho}px;margin:0 auto 0.8rem auto;background:#2a2a40;border-radius:12px;"></div>'
    with open(ruta, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (
        f'<img src="data:image/png;base64,{data}" '
        f'width="{ancho}" '
        f'style="display:block; margin:0 auto 0.8rem auto; border-radius:12px;"/>'
    )

def tarjeta_icono(nombre_png, titulo, descripcion, ancho_icono=60):
    """Genera HTML de una tarjeta con icono PNG, título y descripción."""
    icono = _icono_b64(nombre_png, ancho_icono)
    return f"""
    <div class="app-card">
        {icono}
        <strong>{titulo}</strong><br>
        <small style="color:#8888aa;">{descripcion}</small>
    </div>
    """

ICONOS = {
    "axb":          "sistema_axb.png",
    "memoria":      "memoria_eficiente.png",
    "convergencia": "convergencia.png",
    "estructural":  "analisis_estructural.png",
    "fluidos":      "dinamica_fluidos.png",
    "circuitos":    "circuitos_electricos.png",
    "ml":           "machine_learning.png",
    "redes":        "redes_distribucion.png",
    "edp":          "ecuaciones_diferenciales.png",
    "descenso":     "descenso_gradiente.png",
}


def gradiente_conjugado(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Método de Descenso del Gradiente Conjugado para resolver Ax = b

    Parámetros:
        A        : Matriz simétrica definida positiva (n x n)
        b        : Vector del lado derecho (n,)
        x0       : Punto inicial (default: vector cero)
        tol      : Tolerancia para el criterio de paro
        max_iter : Número máximo de iteraciones

    Retorna:
        x         : Vector solución
        historial : Lista de diccionarios con datos de cada iteración
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    p = r.copy()

    rs_old = r @ r
    historial = [{
        "k": 0, "x": x.copy(), "r": r.copy(), "p": p.copy(),
        "alpha": None, "beta": None, "rs": rs_old
    }]

    for k in range(max_iter):
        Ap = A @ p

        alpha = rs_old / (p @ Ap)

        x = x + alpha * p

        r = r - alpha * Ap

        rs_new = r @ r

        beta = rs_new / rs_old if rs_old != 0 else 0.0

        historial.append({
            "k": k + 1, "x": x.copy(), "r": r.copy(),
            "alpha": alpha, "beta": beta, "rs": rs_new,
            "Ap": Ap.copy(), "p_prev": p.copy()
        })

        if np.sqrt(rs_new) < tol:
            break

        p = r + beta * p
        rs_old = rs_new

    return x, historial


def gradiente_conjugado_2d(A, b, x0=None, tol=1e-10):
    """Versión 2D que retorna puntos para visualización."""
    x = np.zeros(2) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    puntos = [x.copy()]

    for _ in range(100):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        puntos.append(x.copy())
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return np.array(puntos)


def gradiente_conjugado_analisis(A, b, x0=None, tol=1e-10, delta=1e-4, sigma=0.9):
    """
    Versión extendida que registra datos de convergencia,
    condiciones de Wolfe y condición de descenso en cada iteración.

    Condiciones de Wolfe:
      1) Disminución suficiente: f(x_{k+1}) ≤ f(xₖ) + δ·αₖ·gₖᵀdₖ
      2) Curvatura:              g_{k+1}ᵀdₖ ≥ σ·gₖᵀdₖ

    Condición de descenso:       gₖᵀdₖ < 0
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    def phi(v):
        return 0.5 * v @ A @ v - b @ v

    def grad(v):
        return A @ v - b

    g = grad(x)
    d = -g.copy()
    rs_old = g @ g

    datos = {
        "iteracion": [],
        "f_xk": [],
        "norma_residuo": [],
        "norma_grad": [],
        "alpha": [],
        "beta": [],
        "gk_dk": [],
        "es_descenso": [],
        "f_xk1": [],
        "wolfe1_cota": [],
        "wolfe1_ok": [],
        "gk1_dk": [],
        "wolfe2_cota": [],
        "wolfe2_ok": [],
        "norma_error_A": [],
    }

    x_exacta = np.linalg.solve(A, b)
    puntos = [x.copy()]

    for k in range(min(n + 5, 50)):
        f_xk = phi(x)
        gk_dk = g @ d

        Ad = A @ d
        alpha = rs_old / (d @ Ad)

        x_new = x + alpha * d
        g_new = grad(x_new)
        f_xk1 = phi(x_new)
        rs_new = g_new @ g_new

        wolfe1_cota = f_xk + delta * alpha * gk_dk
        wolfe1_ok = f_xk1 <= wolfe1_cota

        gk1_dk = g_new @ d
        wolfe2_cota = sigma * gk_dk
        wolfe2_ok = gk1_dk >= wolfe2_cota

        diff = x_new - x_exacta
        norma_error_A = np.sqrt(max(diff @ A @ diff, 0))

        beta = rs_new / rs_old if rs_old != 0 else 0.0

        datos["iteracion"].append(k)
        datos["f_xk"].append(f_xk)
        datos["f_xk1"].append(f_xk1)
        datos["norma_residuo"].append(np.sqrt(rs_old))
        datos["norma_grad"].append(np.sqrt(g @ g))
        datos["alpha"].append(alpha)
        datos["beta"].append(beta)
        datos["gk_dk"].append(gk_dk)
        datos["es_descenso"].append(gk_dk < 0)
        datos["wolfe1_cota"].append(wolfe1_cota)
        datos["wolfe1_ok"].append(wolfe1_ok)
        datos["gk1_dk"].append(gk1_dk)
        datos["wolfe2_cota"].append(wolfe2_cota)
        datos["wolfe2_ok"].append(wolfe2_ok)
        datos["norma_error_A"].append(norma_error_A)

        x = x_new
        puntos.append(x.copy())

        if np.sqrt(rs_new) < tol:
            break

        d = -g_new + beta * d
        g = g_new
        rs_old = rs_new

    return np.array(puntos), datos


def descenso_gradiente_puro(A, b, x0=None, tol=1e-10, max_iter=200):
    """
    Método de Descenso del Gradiente (Steepest Descent).
    Siempre usa la dirección del gradiente negativo, sin conjugación.
    Esto causa el clásico "zigzag" en valles elongados.
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    def phi(v):
        return 0.5 * v @ A @ v - b @ v

    g = A @ x - b
    puntos = [x.copy()]
    valores_f = [phi(x)]

    for k in range(max_iter):
        d = -g
        Ad = A @ d
        alpha = (g @ g) / (d @ Ad)

        x = x + alpha * d
        g = A @ x - b
        puntos.append(x.copy())
        valores_f.append(phi(x))

        if np.sqrt(g @ g) < tol:
            break

    return np.array(puntos), valores_f


def busqueda_linea_wolfe(f, grad_f, x, d, alpha0=1.0, delta=1e-4, sigma=0.4, max_ls=50):
    """
    Búsqueda de línea con condiciones de Wolfe (backtracking).
    Encuentra α tal que:
      Wolfe I:  f(x + αd) ≤ f(x) + δ·α·∇f(x)ᵀd
      Wolfe II: ∇f(x + αd)ᵀd ≥ σ·∇f(x)ᵀd
    """
    alpha = alpha0
    fx = f(x)
    gx = grad_f(x)
    gd = gx @ d

    for _ in range(max_ls):
        x_new = x + alpha * d
        fx_new = f(x_new)

        if fx_new > fx + delta * alpha * gd:
            alpha *= 0.5
            continue

        gx_new = grad_f(x_new)
        if gx_new @ d < sigma * gd:
            alpha *= 1.5
            continue

        return alpha

    return alpha


def gc_no_lineal(f, grad_f, x0, metodo_beta="PR+", tol=1e-8, max_iter=500):
    """
    Gradiente Conjugado No Lineal para minimizar f(x) general.

    Variantes de βₖ:
      - "FR"  : Fletcher-Reeves
      - "PRP" : Polak-Ribière-Polyak
      - "PR+" : Polak-Ribière+ (con max(0, β))
      - "HS"  : Hestenes-Stiefel
      - "DY"  : Dai-Yuan

    Retorna:
      puntos    : lista de puntos visitados
      datos     : diccionario con métricas por iteración
    """
    x = x0.copy()
    g = grad_f(x)
    d = -g.copy()
    puntos = [x.copy()]

    datos = {
        "iteracion": [], "f_xk": [], "norma_grad": [],
        "alpha": [], "beta": [], "metodo": metodo_beta
    }

    for k in range(max_iter):
        norma_g = np.sqrt(g @ g)
        if norma_g < tol:
            break

        datos["iteracion"].append(k)
        datos["f_xk"].append(f(x))
        datos["norma_grad"].append(norma_g)

        alpha = busqueda_linea_wolfe(f, grad_f, x, d)

        x_new = x + alpha * d
        g_new = grad_f(x_new)

        y = g_new - g

        gg_old = g @ g

        if metodo_beta == "FR":
            beta = (g_new @ g_new) / gg_old if gg_old > 0 else 0.0

        elif metodo_beta == "PRP":
            beta = (g_new @ y) / gg_old if gg_old > 0 else 0.0

        elif metodo_beta == "PR+":
            beta_prp = (g_new @ y) / gg_old if gg_old > 0 else 0.0
            beta = max(0.0, beta_prp)

        elif metodo_beta == "HS":
            dy = d @ y
            beta = (g_new @ y) / dy if abs(dy) > 1e-12 else 0.0

        elif metodo_beta == "DY":
            dy = d @ y
            beta = (g_new @ g_new) / dy if abs(dy) > 1e-12 else 0.0

        else:
            beta = 0.0

        datos["alpha"].append(alpha)
        datos["beta"].append(beta)

        d = -g_new + beta * d

        if g_new @ d > 0:
            d = -g_new.copy()

        x = x_new
        g = g_new
        puntos.append(x.copy())

    return np.array(puntos), datos


def rosenbrock(x):
    """Función de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    dx = -2*(1 - x[0]) + 100 * 2*(x[1] - x[0]**2)*(-2*x[0])
    dy = 100 * 2*(x[1] - x[0]**2)
    return np.array([dx, dy])

def beale(x):
    """Función de Beale: otro clásico de optimización no lineal"""
    t1 = (1.5 - x[0] + x[0]*x[1])**2
    t2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    t3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return t1 + t2 + t3

def grad_beale(x):
    a = 1.5 - x[0] + x[0]*x[1]
    b = 2.25 - x[0] + x[0]*x[1]**2
    c = 2.625 - x[0] + x[0]*x[1]**3
    dx = 2*a*(-1+x[1]) + 2*b*(-1+x[1]**2) + 2*c*(-1+x[1]**3)
    dy = 2*a*x[0] + 2*b*2*x[0]*x[1] + 2*c*3*x[0]*x[1]**2
    return np.array([dx, dy])

def himmelblau(x):
    """Función de Himmelblau: tiene 4 mínimos"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def grad_himmelblau(x):
    dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([dx, dy])


NAV_ITEMS = [
    ("nav_inicio.png",        "Inicio"),
    ("nav_teoria.png",        "Teoría del Método"),
    ("nav_algoritmo.png",     "Algoritmo"),
    ("nav_visualizacion.png", "Visualización 3D"),
    ("nav_gc_vs_gp.png",      "GC vs Gradiente"),
    ("nav_gc_no_lineal.png",  "GC No Lineal"),
    ("nav_codigo.png",        "Código Python"),
    ("nav_ejemplo.png",       "Ejemplo Numérico"),
    ("nav_calculadora.png",   "Calculadora"),
]

NAV_KEYS = [label for _, label in NAV_ITEMS]

if "seccion_idx" not in st.session_state:
    st.session_state.seccion_idx = 0

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1rem;">'
        '<span style="font-size:0.7rem;letter-spacing:3px;color:#00d4aa;'
        'font-family:monospace;text-transform:uppercase;">Navegación</span></div>',
        unsafe_allow_html=True
    )

    for i, (png_name, label) in enumerate(NAV_ITEMS):
        is_active = st.session_state.seccion_idx == i
        col_icon, col_btn = st.columns([0.2, 0.8], gap="small")
        with col_icon:
            ruta_icon = os.path.join(_DIR_BASE, "iconos", "png", png_name)
            if os.path.exists(ruta_icon):
                st.image(ruta_icon, width=28)
            else:
                st.write("")
        with col_btn:
            tipo = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{i}", use_container_width=True, type=tipo):
                st.session_state.seccion_idx = i
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#555;'>DAT-252 · Métodos Numéricos II<br>"
        "UMSA — Informática<br>"
        "Ian Ezequiel Salinas Condori<br>"
        "La Paz, Bolivia · 2026</small>",
        unsafe_allow_html=True
    )

_seccion_map = {
    "Inicio":            "Inicio",
    "Teoría del Método": "Teoría del Método",
    "Algoritmo":         "Algoritmo Paso a Paso",
    "Visualización 3D":  "Visualización 3D",
    "GC vs Gradiente":   "GC vs Gradiente Puro",
    "GC No Lineal":      "GC No Lineal",
    "Código Python":     "Código Python",
    "Ejemplo Numérico":  "Ejemplo Numérico",
    "Calculadora":       "Calculadora Interactiva",
}
seccion = _seccion_map[NAV_KEYS[st.session_state.seccion_idx]]


if seccion == "Inicio":
    st.markdown('<h1 class="main-title">Método de Descenso del<br>Gradiente Conjugado</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Método iterativo de descenso para resolver grandes sistemas lineales simétricos y definidos positivos</p>', unsafe_allow_html=True)
    st.markdown('<p class="meta-info">Ian Ezequiel Salinas Condori · C.I. 13694034 · UMSA Informática · 2026</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(tarjeta_icono(ICONOS["axb"], "Resuelve Ax = b",
            "Sistemas lineales simétricos definidos positivos de gran escala"), unsafe_allow_html=True)
    with col2:
        st.markdown(tarjeta_icono(ICONOS["memoria"], "Eficiente en Memoria",
            "Solo almacena unos pocos vectores, ideal para matrices dispersas"), unsafe_allow_html=True)
    with col3:
        st.markdown(tarjeta_icono(ICONOS["convergencia"], "Convergencia en n pasos",
            "Garantiza solución exacta en máximo n iteraciones (aritmética exacta)"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Aplicaciones")
    c1, c2, c3 = st.columns(3)
    apps = [
        (ICONOS["estructural"], "Análisis Estructural", "Elementos finitos en edificios y puentes"),
        (ICONOS["fluidos"],     "Dinámica de Fluidos",  "Ecuaciones de Navier-Stokes discretizadas"),
        (ICONOS["circuitos"],   "Circuitos Eléctricos", "Redes con miles de nodos"),
        (ICONOS["ml"],          "Machine Learning",     "Optimización de funciones de pérdida"),
        (ICONOS["redes"],       "Redes de Distribución","Presiones en redes de agua/gas"),
        (ICONOS["edp"],         "Ecuaciones Diferenciales", "Diferencias finitas y elementos finitos"),
    ]
    for i, (icono_png, title, desc) in enumerate(apps):
        with [c1, c2, c3][i % 3]:
            st.markdown(tarjeta_icono(icono_png, title, desc) + "<br>", unsafe_allow_html=True)


elif seccion == "Teoría del Método":
    st.markdown("## Fundamentos Teóricos")

    tab1, tab2, tab3, tab4 = st.tabs([
        "El Problema", "¿Por qué Conjugado?", "Convergencia", "¿Cuándo usarlo?"
    ])

    with tab1:
        st.markdown("""
        El **Método de Descenso del Gradiente Conjugado** es el método iterativo más utilizado
        para resolver grandes sistemas lineales de la forma:
        """)
        st.markdown('<div class="formula-box">Ax = b</div>', unsafe_allow_html=True)
        st.markdown("""
        Donde **A** es una matriz cuadrada simétrica y definida positiva, **x** es el vector
        incógnita, y **b** es el vector conocido.

        Este método es un **caso particular de método de descenso**. Resolver el sistema es
        equivalente a minimizar la función cuadrática convexa:
        """)
        st.markdown('<div class="formula-box">φ(x) = ½ xᵀAx − bᵀx</div>', unsafe_allow_html=True)
        st.markdown("""
        El mínimo de φ(x) ocurre exactamente en la solución **x* = A⁻¹b**. El método
        desciende por esta superficie cuadrática usando direcciones inteligentes hasta
        llegar al fondo del "cuenco".
        """)

    with tab2:
        st.markdown("""
        El núcleo del método es generar direcciones de búsqueda **p₀, p₁, ..., p_{n-1}**
        que sean *conjugadas* respecto a la matriz A:
        """)
        st.markdown('<div class="formula-box">pᵢᵀ A pⱼ = 0 &nbsp;&nbsp; (para todo i ≠ j)</div>', unsafe_allow_html=True)
        st.markdown("""
        Esta propiedad garantiza que **cada paso de descenso no arruina el progreso
        logrado en pasos anteriores**.

        A diferencia del gradiente estándar que zigzaguea, las direcciones conjugadas
        convergen en máximo **n iteraciones** (en aritmética exacta).
        """)
        st.markdown("""
        <div class="highlight-box">
            <strong>Idea clave:</strong> El método combina la información del gradiente actual
            con la dirección de búsqueda anterior, generando un descenso más eficiente
            que el gradiente puro.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("La velocidad de convergencia depende del **número de condición** κ(A):")
        st.markdown('<div class="formula-box">‖xₖ − x*‖_A ≤ 2 · ((√κ − 1)/(√κ + 1))ᵏ · ‖x₀ − x*‖_A</div>', unsafe_allow_html=True)
        st.markdown("""
        Donde **κ(A) = λ_max / λ_min**.

        **Propiedades importantes:**
        - Si A tiene **r** valores propios distintos → converge en exactamente **r** iteraciones
        - Si los valores propios se agrupan en pocos *clusters* → convergencia práctica muy rápida
        - El **precondicionamiento** (P ≈ A⁻¹) reduce κ y acelera dramáticamente la convergencia
        """)

    with tab4:
        st.markdown("#### Usar Descenso del Gradiente Conjugado cuando:")
        st.success("**Matrices grandes y dispersas** — Elementos finitos, redes, diferencias finitas. Solo necesita multiplicar A×v.")
        st.success("**Memoria limitada** — Solo almacena unos pocos vectores de tamaño n (vs O(n²) de métodos directos).")

        st.markdown("#### NO usar cuando:")
        st.error("**Matrices densas pequeñas** — La factorización directa (Cholesky/LU) es más eficiente.")
        st.error("**Matrices no simétricas** — Se necesitan variantes como GMRES o BiCGSTAB.")


elif seccion == "Algoritmo Paso a Paso":
    st.markdown("## El Algoritmo de Descenso")

    st.markdown("### Inicialización")
    st.markdown("""
    <div class="step-card">
        Dado un punto inicial <strong>x₀</strong>, calcular:<br>
        • Residuo inicial: <code>r₀ = b − Ax₀</code><br>
        • Primera dirección de descenso: <code>p₀ = r₀</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Bucle iterativo (mientras ‖rₖ‖ ≥ ε)")

    pasos = [
        ("01", "Tamaño de paso", "αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)", "Minimiza φ a lo largo de la dirección de descenso pₖ"),
        ("02", "Actualizar posición", "x_{k+1} = xₖ + αₖ pₖ", "Nos movemos al nuevo punto en la dirección de descenso"),
        ("03", "Actualizar residuo", "r_{k+1} = rₖ − αₖ Apₖ", "Calcula qué tan lejos estamos de la solución"),
        ("04", "Coeficiente conjugado", "β_{k+1} = (r_{k+1}ᵀr_{k+1}) / (rₖᵀrₖ)", "Determina cuánto de la dirección anterior se conserva"),
        ("05", "Nueva dirección", "p_{k+1} = r_{k+1} + β_{k+1} pₖ", "Combina gradiente actual con dirección previa (¡conjugada!)"),
    ]

    for num, titulo, formula, desc in pasos:
        st.markdown(f"""
        <div class="step-card">
            <span class="step-num">Paso {num}</span> — <strong>{titulo}</strong><br>
            <code style="color:#00d4aa; font-size:1.05rem;">{formula}</code><br>
            <small style="color:#8888aa;">{desc}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
        <strong>Eficiencia:</strong> Solo se requiere <strong>un producto matriz-vector</strong> (Ap)
        por iteración. Todo lo demás son productos punto y sumas de vectores — O(n) cada uno.
        El algoritmo optimizado usa <strong>2n² + 9n + 1</strong> operaciones por iteración.
    </div>
    """, unsafe_allow_html=True)


elif seccion == "Visualización 3D":
    st.markdown("## Visualización 3D del Descenso")
    st.markdown("Superficie cuadrática **φ(x) = ½xᵀAx − bᵀx** con el camino de descenso del algoritmo.")

    col_param, col_viz = st.columns([1, 3])

    with col_param:
        st.markdown("### Parámetros")
        a11 = st.slider("A[1,1]", 1.0, 10.0, 4.0, 0.5)
        a12 = st.slider("A[1,2] = A[2,1]", -5.0, 5.0, -1.0, 0.5)
        a22 = st.slider("A[2,2]", 1.0, 10.0, 4.0, 0.5)
        b1 = st.slider("b[1]", -20.0, 20.0, 10.0, 1.0)
        b2 = st.slider("b[2]", -20.0, 20.0, 10.0, 1.0)
        x0_1 = st.slider("x₀[1]", -5.0, 5.0, 0.0, 0.5)
        x0_2 = st.slider("x₀[2]", -5.0, 5.0, 0.0, 0.5)
        st.markdown("---")
        st.markdown("### Wolfe")
        delta_w = st.slider("δ (disminución suficiente)", 0.0001, 0.5, 0.0001, 0.0001, format="%.4f")
        sigma_w = st.slider("σ (curvatura)", 0.1, 0.999, 0.9, 0.01)

    A_2d = np.array([[a11, a12], [a12, a22]])
    b_2d = np.array([b1, b2])
    x0_2d = np.array([x0_1, x0_2])

    eigvals = np.linalg.eigvalsh(A_2d)
    es_def_pos = np.all(eigvals > 0)

    with col_viz:
        if not es_def_pos:
            st.error(f"La matriz NO es definida positiva (valores propios: {eigvals.round(3)}). Ajusta los parámetros.")
        else:
            st.success(f"Matriz definida positiva — valores propios: {eigvals.round(3)} — κ(A) = {eigvals.max()/eigvals.min():.2f}")

            puntos, datos = gradiente_conjugado_analisis(A_2d, b_2d, x0_2d, delta=delta_w, sigma=sigma_w)
            solucion = np.linalg.solve(A_2d, b_2d)

            rng = max(abs(solucion).max() * 2, abs(x0_2d).max() * 2, 5)
            xx = np.linspace(-rng, rng * 1.5, 80)
            yy = np.linspace(-rng, rng * 1.5, 80)
            X, Y = np.meshgrid(xx, yy)

            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    v = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = 0.5 * v @ A_2d @ v - b_2d @ v

            z_path = []
            for pt in puntos:
                z_path.append(0.5 * pt @ A_2d @ pt - b_2d @ pt)

            fig = go.Figure()

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale="Viridis",
                opacity=0.7,
                showscale=False,
                name="φ(x)",
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                )
            ))

            fig.add_trace(go.Scatter3d(
                x=puntos[:, 0], y=puntos[:, 1], z=z_path,
                mode='lines+markers',
                line=dict(color='#00d4aa', width=8),
                marker=dict(size=6, color='#00d4aa'),
                name='Camino DGC'
            ))

            fig.add_trace(go.Scatter3d(
                x=[puntos[0, 0]], y=[puntos[0, 1]], z=[z_path[0]],
                mode='markers',
                marker=dict(size=10, color='#ff6b6b', symbol='diamond'),
                name='Punto Inicial'
            ))

            z_sol = 0.5 * solucion @ A_2d @ solucion - b_2d @ solucion
            fig.add_trace(go.Scatter3d(
                x=[solucion[0]], y=[solucion[1]], z=[z_sol],
                mode='markers',
                marker=dict(size=10, color='#ffcb6b', symbol='diamond'),
                name='Solución'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    zaxis_title="φ(x)",
                    bgcolor='#0a0a0f',
                    xaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                    yaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                    zaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                ),
                paper_bgcolor='#12121a',
                font=dict(color='#e8e8f0'),
                height=550,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(18,18,26,0.8)')
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **Resultado:** Convergencia en **{len(puntos) - 1} iteraciones**
            → Solución: x* = [{solucion[0]:.4f}, {solucion[1]:.4f}]
            """)

    if es_def_pos:
        st.markdown("### Vista de Contornos 2D")
        fig2 = go.Figure()

        fig2.add_trace(go.Contour(
            x=xx, y=yy, z=Z,
            colorscale='Viridis',
            contours=dict(coloring='lines', showlabels=True),
            line_width=1,
            showscale=False,
            name='φ(x)'
        ))

        fig2.add_trace(go.Scatter(
            x=puntos[:, 0], y=puntos[:, 1],
            mode='lines+markers',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=8, color='#00d4aa', line=dict(width=1, color='white')),
            name='Camino DGC'
        ))

        fig2.add_trace(go.Scatter(
            x=[puntos[0, 0]], y=[puntos[0, 1]],
            mode='markers', marker=dict(size=14, color='#ff6b6b', symbol='diamond'),
            name='Inicio'
        ))

        fig2.add_trace(go.Scatter(
            x=[solucion[0]], y=[solucion[1]],
            mode='markers', marker=dict(size=14, color='#ffcb6b', symbol='star'),
            name='Solución'
        ))

        for i in range(len(puntos) - 1):
            fig2.add_annotation(
                x=puntos[i + 1, 0], y=puntos[i + 1, 1],
                ax=puntos[i, 0], ay=puntos[i, 1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1.5, arrowwidth=2,
                arrowcolor="#00d4aa"
            )

        fig2.update_layout(
            xaxis_title="x₁", yaxis_title="x₂",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'),
            height=450,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40', scaleanchor="x"),
            legend=dict(bgcolor='rgba(18,18,26,0.8)')
        )

        st.plotly_chart(fig2, use_container_width=True)

        iters = datos["iteracion"]
        n_iters = len(iters)

        if n_iters > 0:
            st.markdown("---")
            st.markdown("##  Análisis de Convergencia y Condiciones")

            st.markdown("### 1. Convergencia del Residuo y Error")
            st.markdown("""
            <div class="highlight-box">
                <strong>Cota teórica:</strong> &nbsp;
                ‖xₖ − x*‖_A &nbsp;≤&nbsp; 2 · ((√κ − 1)/(√κ + 1))ᵏ · ‖x₀ − x*‖_A
            </div>
            """, unsafe_allow_html=True)

            fig_conv = go.Figure()

            fig_conv.add_trace(go.Scatter(
                x=iters, y=datos["norma_residuo"],
                mode='lines+markers',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=9, color='#00d4aa', symbol='circle'),
                name='‖rₖ‖ (norma residuo)'
            ))

            fig_conv.add_trace(go.Scatter(
                x=iters, y=datos["norma_error_A"],
                mode='lines+markers',
                line=dict(color='#7b68ee', width=3, dash='dash'),
                marker=dict(size=9, color='#7b68ee', symbol='diamond'),
                name='‖xₖ − x*‖_A (error norma-A)'
            ))

            kappa = eigvals.max() / eigvals.min()
            ratio = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
            if len(datos["norma_error_A"]) > 0:
                e0 = datos["norma_error_A"][0] if datos["norma_error_A"][0] > 0 else 1.0
                cota_teorica = [2 * (ratio ** k) * e0 for k in iters]
                fig_conv.add_trace(go.Scatter(
                    x=iters, y=cota_teorica,
                    mode='lines',
                    line=dict(color='#ff6b6b', width=2, dash='dot'),
                    name=f'Cota teórica (κ={kappa:.2f})'
                ))

            fig_conv.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="Magnitud",
                yaxis_type="log",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=420,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
                legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.01, y=0.99),
                hovermode='x unified'
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            st.markdown("### 2. Descenso de la Función Objetivo φ(x)")
            st.markdown("""
            <div class="highlight-box">
                En cada iteración, φ(xₖ₊₁) debe ser <strong>menor</strong> que φ(xₖ). La curva debe descender
                monótonamente hasta alcanzar el mínimo global φ(x*).
            </div>
            """, unsafe_allow_html=True)

            fig_obj = go.Figure()

            fig_obj.add_trace(go.Scatter(
                x=iters, y=datos["f_xk"],
                mode='lines+markers',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=10, color='#00d4aa'),
                name='φ(xₖ)',
                fill='tozeroy',
                fillcolor='rgba(0,212,170,0.08)'
            ))

            fig_obj.add_trace(go.Scatter(
                x=iters, y=datos["f_xk1"],
                mode='markers',
                marker=dict(size=8, color='#ffcb6b', symbol='star'),
                name='φ(x_{k+1})'
            ))

            phi_min = 0.5 * solucion @ A_2d @ solucion - b_2d @ solucion
            fig_obj.add_hline(y=phi_min, line_dash="dash", line_color="#ff6b6b",
                             annotation_text=f"φ(x*) = {phi_min:.4f}",
                             annotation_font_color="#ff6b6b")

            fig_obj.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="φ(x)",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=380,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
                legend=dict(bgcolor='rgba(18,18,26,0.8)'),
                hovermode='x unified'
            )
            st.plotly_chart(fig_obj, use_container_width=True)

            st.markdown("### 3. Condición de Descenso: gₖᵀdₖ < 0")
            st.markdown("""
            <div class="highlight-box">
                Para que dₖ sea una <strong>dirección de descenso</strong>, se requiere que el producto
                <strong>gₖᵀdₖ &lt; 0</strong>. Esto garantiza que al movernos en la dirección dₖ,
                la función φ decrece inicialmente.
            </div>
            """, unsafe_allow_html=True)

            fig_desc = go.Figure()

            colores_desc = ['#00d4aa' if ok else '#ff6b6b' for ok in datos["es_descenso"]]
            fig_desc.add_trace(go.Bar(
                x=iters, y=datos["gk_dk"],
                marker_color=colores_desc,
                name='gₖᵀdₖ',
                text=[f"{'OK' if ok else 'X'}" for ok in datos["es_descenso"]],
                textposition='outside',
                textfont=dict(size=14)
            ))

            fig_desc.add_hline(y=0, line_color="#ffffff", line_width=2,
                              annotation_text="Límite (debe ser < 0)",
                              annotation_font_color="#8888aa")

            fig_desc.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="gₖᵀdₖ",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=380,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
                legend=dict(bgcolor='rgba(18,18,26,0.8)'),
                showlegend=False
            )
            st.plotly_chart(fig_desc, use_container_width=True)

            st.markdown("### 4. Condición de Wolfe I — Disminución Suficiente")
            st.markdown(f"""
            <div class="formula-box">
                f(x_{{k+1}}) &nbsp;≤&nbsp; f(xₖ) + δ · αₖ · gₖᵀdₖ &nbsp;&nbsp; (δ = {delta_w})
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="highlight-box">
                Esta condición asegura que el paso αₖ produce una <strong>reducción real</strong>
                de la función, no solo un descenso infinitesimal. La línea azul (f real)
                debe estar por debajo de la línea roja (cota de Wolfe I).
            </div>
            """, unsafe_allow_html=True)

            fig_w1 = go.Figure()

            fig_w1.add_trace(go.Scatter(
                x=iters, y=datos["f_xk1"],
                mode='lines+markers',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=9, color='#00d4aa'),
                name='f(x_{k+1}) — valor real'
            ))

            fig_w1.add_trace(go.Scatter(
                x=iters, y=datos["wolfe1_cota"],
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=7, color='#ff6b6b', symbol='x'),
                name='f(xₖ) + δ·α·gₖᵀdₖ — cota Wolfe I'
            ))

            for i_k, ok in enumerate(datos["wolfe1_ok"]):
                fig_w1.add_annotation(
                    x=iters[i_k], y=datos["f_xk1"][i_k],
                    text="OK" if ok else "X",
                    showarrow=False,
                    font=dict(size=16, color='#00d4aa' if ok else '#ff6b6b'),
                    yshift=18
                )

            fig_w1.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="f(x)",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=400,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
                legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.01, y=0.99),
                hovermode='x unified'
            )
            st.plotly_chart(fig_w1, use_container_width=True)

            st.markdown("### 5. Condición de Wolfe II — Curvatura")
            st.markdown(f"""
            <div class="formula-box">
                g_{{k+1}}ᵀdₖ &nbsp;≥&nbsp; σ · gₖᵀdₖ &nbsp;&nbsp; (σ = {sigma_w})
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="highlight-box">
                La condición de curvatura previene que los pasos sean <strong>demasiado cortos</strong>.
                Exige que la pendiente en el nuevo punto sea menos negativa que σ veces
                la pendiente original. La línea verde (pendiente real) debe estar
                <strong>por encima</strong> de la línea roja (cota σ·gₖᵀdₖ).
            </div>
            """, unsafe_allow_html=True)

            fig_w2 = go.Figure()

            fig_w2.add_trace(go.Scatter(
                x=iters, y=datos["gk1_dk"],
                mode='lines+markers',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=9, color='#00d4aa'),
                name='g_{k+1}ᵀdₖ — valor real'
            ))

            fig_w2.add_trace(go.Scatter(
                x=iters, y=datos["wolfe2_cota"],
                mode='lines+markers',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=7, color='#ff6b6b', symbol='x'),
                name='σ · gₖᵀdₖ — cota Wolfe II'
            ))

            fig_w2.add_hline(y=0, line_color="#555570", line_width=1, line_dash="dot")

            for i_k, ok in enumerate(datos["wolfe2_ok"]):
                fig_w2.add_annotation(
                    x=iters[i_k], y=datos["gk1_dk"][i_k],
                    text="OK" if ok else "X",
                    showarrow=False,
                    font=dict(size=16, color='#00d4aa' if ok else '#ff6b6b'),
                    yshift=18
                )

            fig_w2.update_layout(
                xaxis_title="Iteración k",
                yaxis_title="Derivada direccional",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=400,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
                legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.01, y=0.99),
                hovermode='x unified'
            )
            st.plotly_chart(fig_w2, use_container_width=True)

            st.markdown("### Tabla Resumen de Condiciones")
            import pandas as pd
            resumen_rows = []
            for i_k in range(n_iters):
                resumen_rows.append({
                    "k": datos["iteracion"][i_k],
                    "αₖ": f"{datos['alpha'][i_k]:.6f}",
                    "φ(xₖ)": f"{datos['f_xk'][i_k]:.6f}",
                    "φ(x_{k+1})": f"{datos['f_xk1'][i_k]:.6f}",
                    "‖rₖ‖": f"{datos['norma_residuo'][i_k]:.2e}",
                    "gₖᵀdₖ": f"{datos['gk_dk'][i_k]:.4f}",
                    "Descenso": "Si" if datos["es_descenso"][i_k] else "No",
                    "Wolfe I": "Si" if datos["wolfe1_ok"][i_k] else "No",
                    "Wolfe II": "Si" if datos["wolfe2_ok"][i_k] else "No",
                })
            df_resumen = pd.DataFrame(resumen_rows)
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="highlight-box">
                <strong>Condición de Zoutendijk:</strong> Si las condiciones de Wolfe se cumplen en cada iteración,
                entonces se garantiza que <strong>lim inf ‖gₖ‖ = 0</strong> cuando k → ∞.
                Esto es fundamental para demostrar la convergencia global del método de descenso del gradiente conjugado no lineal.
            </div>
            """, unsafe_allow_html=True)


elif seccion == "GC vs Gradiente Puro":
    st.markdown("## Gradiente Conjugado vs Gradiente Puro")
    st.markdown("""
    Comparación visual entre el **Método de Descenso del Gradiente Conjugado (DGC)** y el
    **Descenso del Gradiente Puro (Steepest Descent)**. El gradiente puro siempre se mueve
    en la dirección del gradiente negativo, lo que causa un **zigzag** en valles elongados.
    """)

    st.markdown("---")

    st.markdown("### Escenario")
    preset = st.selectbox("Elige un tipo de valle:", [
        "Valle Elongado (κ=10) — zigzag clásico",
        "Valle Muy Elongado (κ=50) — zigzag extremo",
        "Valle Circular (κ≈1) — ambos convergen rápido",
        "Valle Rotado 45° (κ=10) — ejes inclinados",
        "Personalizado"
    ])

    if preset.startswith("Valle Elongado"):
        A_cmp = np.array([[10.0, 0], [0, 1.0]])
        b_cmp = np.array([10.0, 2.0])
        x0_cmp = np.array([-4.0, -3.0])
    elif preset.startswith("Valle Muy Elongado"):
        A_cmp = np.array([[50.0, 0], [0, 1.0]])
        b_cmp = np.array([20.0, 4.0])
        x0_cmp = np.array([-3.0, -4.0])
    elif preset.startswith("Valle Circular"):
        A_cmp = np.array([[3.0, 0], [0, 3.0]])
        b_cmp = np.array([6.0, 6.0])
        x0_cmp = np.array([-3.0, -3.0])
    elif preset.startswith("Valle Rotado"):
        c = np.cos(np.pi / 4)
        s = np.sin(np.pi / 4)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([10.0, 1.0])
        A_cmp = R.T @ D @ R
        b_cmp = np.array([5.0, 5.0])
        x0_cmp = np.array([-4.0, 3.0])
    else:
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            a_c11 = st.number_input("A[1,1]", value=10.0, step=1.0, key="cmp_a11")
            a_c12 = st.number_input("A[1,2]=A[2,1]", value=0.0, step=0.5, key="cmp_a12")
            a_c22 = st.number_input("A[2,2]", value=1.0, step=1.0, key="cmp_a22")
        with col_p2:
            b_c1 = st.number_input("b[1]", value=10.0, step=1.0, key="cmp_b1")
            b_c2 = st.number_input("b[2]", value=2.0, step=1.0, key="cmp_b2")
        with col_p3:
            x_c1 = st.number_input("x₀[1]", value=-4.0, step=0.5, key="cmp_x01")
            x_c2 = st.number_input("x₀[2]", value=-3.0, step=0.5, key="cmp_x02")
        A_cmp = np.array([[a_c11, a_c12], [a_c12, a_c22]])
        b_cmp = np.array([b_c1, b_c2])
        x0_cmp = np.array([x_c1, x_c2])

    eigvals_cmp = np.linalg.eigvalsh(A_cmp)
    es_dp = np.all(eigvals_cmp > 0)

    if not es_dp:
        st.error(f"Matriz no definida positiva (valores propios: {eigvals_cmp.round(3)})")
    else:
        kappa_cmp = eigvals_cmp.max() / eigvals_cmp.min()
        solucion_cmp = np.linalg.solve(A_cmp, b_cmp)

        st.info(f"**κ(A) = {kappa_cmp:.2f}** — Valores propios: {eigvals_cmp.round(3)} — Solución: [{solucion_cmp[0]:.4f}, {solucion_cmp[1]:.4f}]")

        puntos_gc = gradiente_conjugado_2d(A_cmp, b_cmp, x0_cmp)
        puntos_gd, fvals_gd = descenso_gradiente_puro(A_cmp, b_cmp, x0_cmp, max_iter=80)

        fvals_gc = [0.5 * p @ A_cmp @ p - b_cmp @ p for p in puntos_gc]

        rng = max(abs(solucion_cmp).max(), abs(x0_cmp).max()) * 1.5 + 1
        xx = np.linspace(-rng, rng, 120)
        yy = np.linspace(-rng, rng, 120)
        X, Y = np.meshgrid(xx, yy)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                v = np.array([X[i, j], Y[i, j]])
                Z[i, j] = 0.5 * v @ A_cmp @ v - b_cmp @ v

        st.markdown("### Mapa de Contornos — Comparación de Caminos")

        fig_cmp = go.Figure()

        fig_cmp.add_trace(go.Contour(
            x=xx, y=yy, z=Z,
            colorscale='Viridis',
            contours=dict(coloring='lines', showlabels=True),
            line_width=1, showscale=False,
            name='φ(x)', opacity=0.6
        ))

        fig_cmp.add_trace(go.Scatter(
            x=puntos_gd[:, 0], y=puntos_gd[:, 1],
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=5, color='#ff6b6b'),
            name=f'Gradiente Puro ({len(puntos_gd)-1} iters)'
        ))

        max_flechas_gd = min(len(puntos_gd) - 1, 30)
        for i in range(max_flechas_gd):
            fig_cmp.add_annotation(
                x=puntos_gd[i + 1, 0], y=puntos_gd[i + 1, 1],
                ax=puntos_gd[i, 0], ay=puntos_gd[i, 1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor='rgba(255,107,107,0.5)'
            )

        fig_cmp.add_trace(go.Scatter(
            x=puntos_gc[:, 0], y=puntos_gc[:, 1],
            mode='lines+markers',
            line=dict(color='#00d4aa', width=4),
            marker=dict(size=9, color='#00d4aa', line=dict(width=1, color='white')),
            name=f'Gradiente Conjugado ({len(puntos_gc)-1} iters)'
        ))

        for i in range(len(puntos_gc) - 1):
            fig_cmp.add_annotation(
                x=puntos_gc[i + 1, 0], y=puntos_gc[i + 1, 1],
                ax=puntos_gc[i, 0], ay=puntos_gc[i, 1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=2.5,
                arrowcolor='#00d4aa'
            )

        fig_cmp.add_trace(go.Scatter(
            x=[x0_cmp[0]], y=[x0_cmp[1]],
            mode='markers+text',
            marker=dict(size=16, color='#ff6b6b', symbol='diamond', line=dict(width=2, color='white')),
            text=['Inicio'], textposition='top center',
            textfont=dict(color='#ff6b6b', size=13),
            name='Punto Inicial', showlegend=False
        ))

        fig_cmp.add_trace(go.Scatter(
            x=[solucion_cmp[0]], y=[solucion_cmp[1]],
            mode='markers+text',
            marker=dict(size=16, color='#ffcb6b', symbol='star', line=dict(width=2, color='white')),
            text=['Solución x*'], textposition='bottom center',
            textfont=dict(color='#ffcb6b', size=13),
            name='Solución', showlegend=False
        ))

        fig_cmp.update_layout(
            xaxis_title="x₁", yaxis_title="x₂",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'),
            height=600,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40', scaleanchor="x"),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.01, y=0.99, font=dict(size=13))
        )

        st.plotly_chart(fig_cmp, use_container_width=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Iteraciones GC", f"{len(puntos_gc) - 1}",
                      delta=f"-{len(puntos_gd) - len(puntos_gc)} vs GP",
                      delta_color="normal")
        with col_m2:
            st.metric("Iteraciones Gradiente Puro", f"{len(puntos_gd) - 1}")
        with col_m3:
            speedup = (len(puntos_gd) - 1) / max(len(puntos_gc) - 1, 1)
            st.metric("Speedup GC", f"{speedup:.1f}x más rápido")

        st.markdown("---")
        st.markdown("### Convergencia Comparada — φ(xk) vs Iteración")
        st.markdown("""
        <div class="highlight-box">
            Observa cómo el <strong style="color:#00d4aa;">Gradiente Conjugado</strong> desciende de forma
            eficiente al mínimo, mientras el <strong style="color:#ff6b6b;">Gradiente Puro</strong> desciende
            en escalones irregulares porque zigzaguea entre paredes del valle.
        </div>
        """, unsafe_allow_html=True)

        fig_fconv = go.Figure()

        fig_fconv.add_trace(go.Scatter(
            x=list(range(len(fvals_gd))),
            y=fvals_gd,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=5, color='#ff6b6b'),
            name='Gradiente Puro',
            fill='tozeroy',
            fillcolor='rgba(255,107,107,0.05)'
        ))

        fig_fconv.add_trace(go.Scatter(
            x=list(range(len(fvals_gc))),
            y=fvals_gc,
            mode='lines+markers',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=9, color='#00d4aa', symbol='diamond'),
            name='Gradiente Conjugado',
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.05)'
        ))

        phi_min = 0.5 * solucion_cmp @ A_cmp @ solucion_cmp - b_cmp @ solucion_cmp
        fig_fconv.add_hline(y=phi_min, line_dash="dash", line_color="#ffcb6b",
                           annotation_text=f"φ(x*) = {phi_min:.4f}",
                           annotation_font_color="#ffcb6b")

        fig_fconv.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="φ(xₖ)",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=420,
            xaxis=dict(gridcolor='#2a2a40', dtick=max(1, len(fvals_gd)//15)),
            yaxis=dict(gridcolor='#2a2a40'),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.6, y=0.95),
            hovermode='x unified'
        )

        st.plotly_chart(fig_fconv, use_container_width=True)

        st.markdown("### Error en norma-A — Escala Logarítmica")

        errores_gc = []
        for p in puntos_gc:
            diff = p - solucion_cmp
            errores_gc.append(np.sqrt(max(diff @ A_cmp @ diff, 1e-30)))

        errores_gd = []
        for p in puntos_gd:
            diff = p - solucion_cmp
            errores_gd.append(np.sqrt(max(diff @ A_cmp @ diff, 1e-30)))

        ratio_cmp = (np.sqrt(kappa_cmp) - 1) / (np.sqrt(kappa_cmp) + 1)
        max_len = max(len(errores_gc), len(errores_gd))
        cota_k = [2 * (ratio_cmp ** k) * errores_gc[0] for k in range(max_len)]

        fig_err = go.Figure()

        fig_err.add_trace(go.Scatter(
            x=list(range(len(errores_gd))),
            y=errores_gd,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=5, color='#ff6b6b'),
            name='Gradiente Puro'
        ))

        fig_err.add_trace(go.Scatter(
            x=list(range(len(errores_gc))),
            y=errores_gc,
            mode='lines+markers',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=9, color='#00d4aa', symbol='diamond'),
            name='Gradiente Conjugado'
        ))

        fig_err.add_trace(go.Scatter(
            x=list(range(max_len)),
            y=cota_k,
            mode='lines',
            line=dict(color='#7b68ee', width=2, dash='dot'),
            name=f'Cota teórica (κ={kappa_cmp:.1f})'
        ))

        fig_err.update_layout(
            xaxis_title="Iteración k",
            yaxis_title="‖xₖ − x*‖_A",
            yaxis_type="log",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=420,
            xaxis=dict(gridcolor='#2a2a40', dtick=max(1, max_len//15)),
            yaxis=dict(gridcolor='#2a2a40'),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.6, y=0.95),
            hovermode='x unified'
        )

        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("---")
        st.markdown("### Vista 3D — Ambos Caminos sobre la Superficie")

        z_gc = [0.5 * p @ A_cmp @ p - b_cmp @ p for p in puntos_gc]
        z_gd = [0.5 * p @ A_cmp @ p - b_cmp @ p for p in puntos_gd]

        fig_3d = go.Figure()

        fig_3d.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale="Viridis",
            opacity=0.55, showscale=False,
            name="φ(x)"
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=puntos_gd[:, 0], y=puntos_gd[:, 1], z=z_gd,
            mode='lines+markers',
            line=dict(color='#ff6b6b', width=4),
            marker=dict(size=3, color='#ff6b6b'),
            name=f'Gradiente Puro ({len(puntos_gd)-1} it.)'
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=puntos_gc[:, 0], y=puntos_gc[:, 1], z=z_gc,
            mode='lines+markers',
            line=dict(color='#00d4aa', width=8),
            marker=dict(size=6, color='#00d4aa'),
            name=f'Gradiente Conjugado ({len(puntos_gc)-1} it.)'
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=[x0_cmp[0]], y=[x0_cmp[1]], z=[0.5 * x0_cmp @ A_cmp @ x0_cmp - b_cmp @ x0_cmp],
            mode='markers',
            marker=dict(size=10, color='#ff6b6b', symbol='diamond'),
            name='Inicio'
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=[solucion_cmp[0]], y=[solucion_cmp[1]], z=[phi_min],
            mode='markers',
            marker=dict(size=10, color='#ffcb6b', symbol='diamond'),
            name='Solución'
        ))

        fig_3d.update_layout(
            scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="φ(x)",
                bgcolor='#0a0a0f',
                xaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                yaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                zaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
            ),
            paper_bgcolor='#12121a',
            font=dict(color='#e8e8f0'),
            height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(18,18,26,0.8)')
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")
        st.markdown("### ¿Por qué ocurre el zigzag?")
        st.markdown("""
        <div class="highlight-box">
            <strong>Gradiente Puro (Steepest Descent):</strong> En cada paso toma la dirección donde
            φ decrece <em>más rápido</em> (gradiente negativo). Pero en valles elongados, la dirección
            de mayor descenso apunta hacia las <strong>paredes</strong> del valle, no hacia el fondo.
            Entonces rebota de pared en pared, avanzando lentamente hacia la solución.
            <br><br>
            Cuanto mayor es <strong>κ(A)</strong>, más elongado el valle, peor el zigzag.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="highlight-box">
            <strong>Gradiente Conjugado:</strong> Corrige este problema combinando el gradiente actual
            con la dirección anterior mediante el parámetro <strong>β</strong>. La nueva dirección es
            <strong>conjugada</strong> a la anterior respecto a A, lo que significa que cada paso
            optimiza una dimensión <em>independiente</em> del problema. No hay retroceso, no hay zigzag.
            <br><br>
            Para un sistema 2×2, <strong>siempre converge en máximo 2 pasos</strong>, sin importar κ(A).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Resumen Comparativo")
        import pandas as pd
        df_cmp = pd.DataFrame({
            "Característica": [
                "Dirección de búsqueda",
                "Iteraciones (sistema n×n)",
                f"Iteraciones (este ejemplo, κ={kappa_cmp:.1f})",
                "Memoria extra",
                "Efecto en valles elongados",
                "Velocidad de convergencia"
            ],
            "Gradiente Puro": [
                "dₖ = −gₖ (siempre gradiente negativo)",
                "Puede necesitar miles",
                f"{len(puntos_gd) - 1}",
                "Solo 1 vector",
                "Zigzag severo",
                "Lineal — depende de ((κ−1)/(κ+1))²"
            ],
            "Gradiente Conjugado": [
                "dₖ = −gₖ + βₖ dₖ₋₁ (conjugada)",
                "Máximo n",
                f"{len(puntos_gc) - 1}",
                "2-3 vectores",
                "Directo al mínimo",
                "Superlineal — depende de ((√κ−1)/(√κ+1))"
            ]
        })
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)


elif seccion == "GC No Lineal":
    st.markdown("## Gradiente Conjugado No Lineal")
    st.markdown("""
    Cuando la función **f(x)** no es cuadrática, el algoritmo se adapta con dos cambios clave:
    el paso **αₖ** se calcula con una **búsqueda de línea** (condiciones de Wolfe), y existen
    **múltiples fórmulas** para el parámetro **βₖ**, cada una con distintas propiedades.
    """)

    st.markdown("---")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        funcion = st.selectbox("Función de prueba:", [
            "Rosenbrock — f(x,y) = (1−x)² + 100(y−x²)²",
            "Himmelblau — f(x,y) = (x²+y−11)² + (x+y²−7)²",
            "Beale — f(x,y) = Σ(aᵢ − x + x·yⁱ)²"
        ])

    with col_cfg2:
        x0_nl1 = st.number_input("x₀[1]", value=-1.5, step=0.5, key="nl_x01")
        x0_nl2 = st.number_input("x₀[2]", value=-1.0, step=0.5, key="nl_x02")

    x0_nl = np.array([x0_nl1, x0_nl2])

    if funcion.startswith("Rosenbrock"):
        f_test, grad_test = rosenbrock, grad_rosenbrock
        nombre_f = "Rosenbrock"
        sol_exacta = np.array([1.0, 1.0])
        info_f = "Mínimo global en (1, 1) con f = 0. Valle curvo largo y estrecho — difícil para gradiente puro."
    elif funcion.startswith("Himmelblau"):
        f_test, grad_test = himmelblau, grad_himmelblau
        nombre_f = "Himmelblau"
        sol_exacta = np.array([3.0, 2.0])
        info_f = "Tiene 4 mínimos locales: (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848). El resultado depende del punto inicial."
    else:
        f_test, grad_test = beale, grad_beale
        nombre_f = "Beale"
        sol_exacta = np.array([3.0, 0.5])
        info_f = "Mínimo global en (3, 0.5) con f = 0. Superficie con valle plano alargado."

    st.info(f"**{nombre_f}:** {info_f}")

    st.markdown("### Variantes de βₖ a comparar")

    variantes_sel = st.multiselect(
        "Selecciona las variantes:",
        ["FR (Fletcher-Reeves)", "PRP (Polak-Ribière)", "PR+ (Polak-Ribière+)",
         "HS (Hestenes-Stiefel)", "DY (Dai-Yuan)"],
        default=["FR (Fletcher-Reeves)", "PR+ (Polak-Ribière+)", "DY (Dai-Yuan)"]
    )

    mapa_variantes = {
        "FR (Fletcher-Reeves)": "FR",
        "PRP (Polak-Ribière)": "PRP",
        "PR+ (Polak-Ribière+)": "PR+",
        "HS (Hestenes-Stiefel)": "HS",
        "DY (Dai-Yuan)": "DY"
    }

    colores_var = {
        "FR": "#ff6b6b",
        "PRP": "#7b68ee",
        "PR+": "#00d4aa",
        "HS": "#ffcb6b",
        "DY": "#ff9ff3"
    }

    nombres_var = {
        "FR": "Fletcher-Reeves",
        "PRP": "Polak-Ribière",
        "PR+": "Polak-Ribière+",
        "HS": "Hestenes-Stiefel",
        "DY": "Dai-Yuan"
    }

    if len(variantes_sel) == 0:
        st.warning("Selecciona al menos una variante.")
    else:
        resultados = {}
        for v_label in variantes_sel:
            v_key = mapa_variantes[v_label]
            try:
                pts, dat = gc_no_lineal(f_test, grad_test, x0_nl, metodo_beta=v_key, max_iter=500)
                resultados[v_key] = {"puntos": pts, "datos": dat}
            except Exception:
                resultados[v_key] = None

        all_pts = np.vstack([r["puntos"] for r in resultados.values() if r is not None])
        margin = max(abs(all_pts).max(), abs(x0_nl).max(), abs(sol_exacta).max()) + 2
        margin = min(margin, 10)
        xx_nl = np.linspace(-margin, margin, 150)
        yy_nl = np.linspace(-margin, margin, 150)
        X_nl, Y_nl = np.meshgrid(xx_nl, yy_nl)
        Z_nl = np.zeros_like(X_nl)
        for i in range(X_nl.shape[0]):
            for j in range(X_nl.shape[1]):
                Z_nl[i, j] = f_test(np.array([X_nl[i, j], Y_nl[i, j]]))

        z_clip = np.percentile(Z_nl, 95)
        Z_nl_clip = np.clip(Z_nl, None, z_clip)

        st.markdown("### Mapa de Contornos — Caminos de cada variante")

        fig_nl = go.Figure()

        fig_nl.add_trace(go.Contour(
            x=xx_nl, y=yy_nl, z=Z_nl_clip,
            colorscale='Viridis',
            contours=dict(coloring='lines', showlabels=True),
            line_width=1, showscale=False,
            ncontours=30, opacity=0.5,
            name='f(x)'
        ))

        for v_key, res in resultados.items():
            if res is None:
                continue
            pts = res["puntos"]
            fig_nl.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode='lines+markers',
                line=dict(color=colores_var[v_key], width=2.5),
                marker=dict(size=4, color=colores_var[v_key]),
                name=f'{nombres_var[v_key]} ({len(pts)-1} it.)'
            ))

        fig_nl.add_trace(go.Scatter(
            x=[x0_nl[0]], y=[x0_nl[1]],
            mode='markers+text',
            marker=dict(size=14, color='#ff6b6b', symbol='diamond', line=dict(width=2, color='white')),
            text=['Inicio'], textposition='top center',
            textfont=dict(color='#ff6b6b', size=12),
            showlegend=False
        ))

        fig_nl.add_trace(go.Scatter(
            x=[sol_exacta[0]], y=[sol_exacta[1]],
            mode='markers+text',
            marker=dict(size=14, color='#ffcb6b', symbol='star', line=dict(width=2, color='white')),
            text=['Mínimo'], textposition='bottom center',
            textfont=dict(color='#ffcb6b', size=12),
            showlegend=False
        ))

        fig_nl.update_layout(
            xaxis_title="x₁", yaxis_title="x₂",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=550,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40', scaleanchor="x"),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.01, y=0.99, font=dict(size=12))
        )
        st.plotly_chart(fig_nl, use_container_width=True)

        st.markdown("### Convergencia de f(xk)")
        st.markdown("""
        <div class="highlight-box">
            Cómo desciende la función objetivo en cada variante. Las variantes que
            <strong>evitan el estancamiento</strong> (como PR+ y DY) suelen converger más rápido
            en funciones difíciles.
        </div>
        """, unsafe_allow_html=True)

        fig_fconv_nl = go.Figure()

        for v_key, res in resultados.items():
            if res is None:
                continue
            d = res["datos"]
            fig_fconv_nl.add_trace(go.Scatter(
                x=d["iteracion"], y=d["f_xk"],
                mode='lines',
                line=dict(color=colores_var[v_key], width=2.5),
                name=nombres_var[v_key]
            ))

        fig_fconv_nl.update_layout(
            xaxis_title="Iteración k", yaxis_title="f(xₖ)",
            yaxis_type="log",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=400,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40'),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.6, y=0.95),
            hovermode='x unified'
        )
        st.plotly_chart(fig_fconv_nl, use_container_width=True)

        st.markdown("### Norma del Gradiente")
        st.markdown("""
        <div class="highlight-box">
            La norma del gradiente mide qué tan lejos estamos de un punto estacionario.
            <strong>lim ‖∇f‖ → 0</strong> confirma la convergencia (condición de Zoutendijk).
        </div>
        """, unsafe_allow_html=True)

        fig_grad_nl = go.Figure()

        for v_key, res in resultados.items():
            if res is None:
                continue
            d = res["datos"]
            fig_grad_nl.add_trace(go.Scatter(
                x=d["iteracion"], y=d["norma_grad"],
                mode='lines',
                line=dict(color=colores_var[v_key], width=2.5),
                name=nombres_var[v_key]
            ))

        fig_grad_nl.add_hline(y=1e-8, line_dash="dash", line_color="#555570",
                              annotation_text="Tolerancia ε=1e-8",
                              annotation_font_color="#8888aa")

        fig_grad_nl.update_layout(
            xaxis_title="Iteración k", yaxis_title="‖∇f(xₖ)‖",
            yaxis_type="log",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=400,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40'),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.6, y=0.95),
            hovermode='x unified'
        )
        st.plotly_chart(fig_grad_nl, use_container_width=True)

        st.markdown("### Evolución del Parámetro Bk")
        st.markdown("""
        <div class="highlight-box">
            El parámetro <strong>βₖ</strong> controla cuánto de la dirección anterior se conserva.
            Valores de β = 0 significan que el método se <strong>reinicia</strong> al gradiente puro.
            Noten cómo <strong>PR+</strong> fuerza β ≥ 0, reiniciándose automáticamente cuando
            el progreso se estanca.
        </div>
        """, unsafe_allow_html=True)

        fig_beta_nl = go.Figure()

        for v_key, res in resultados.items():
            if res is None:
                continue
            d = res["datos"]
            fig_beta_nl.add_trace(go.Scatter(
                x=d["iteracion"], y=d["beta"],
                mode='lines+markers',
                line=dict(color=colores_var[v_key], width=2),
                marker=dict(size=3, color=colores_var[v_key]),
                name=nombres_var[v_key]
            ))

        fig_beta_nl.add_hline(y=0, line_color="#ffffff", line_width=1, line_dash="dot")

        fig_beta_nl.update_layout(
            xaxis_title="Iteración k", yaxis_title="βₖ",
            paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
            font=dict(color='#e8e8f0'), height=380,
            xaxis=dict(gridcolor='#2a2a40'),
            yaxis=dict(gridcolor='#2a2a40'),
            legend=dict(bgcolor='rgba(18,18,26,0.8)', x=0.6, y=0.95),
            hovermode='x unified'
        )
        st.plotly_chart(fig_beta_nl, use_container_width=True)

        st.markdown("---")
        st.markdown("### Vista 3D de la Superficie")

        fig_3d_nl = go.Figure()

        fig_3d_nl.add_trace(go.Surface(
            x=X_nl, y=Y_nl, z=Z_nl_clip,
            colorscale="Viridis",
            opacity=0.55, showscale=False
        ))

        for v_key, res in resultados.items():
            if res is None:
                continue
            pts = res["puntos"]
            z_pts = [f_test(p) for p in pts]
            z_pts_clip = [min(z, z_clip) for z in z_pts]
            fig_3d_nl.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=z_pts_clip,
                mode='lines+markers',
                line=dict(color=colores_var[v_key], width=5),
                marker=dict(size=3, color=colores_var[v_key]),
                name=nombres_var[v_key]
            ))

        fig_3d_nl.update_layout(
            scene=dict(
                xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x)",
                bgcolor='#0a0a0f',
                xaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                yaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
                zaxis=dict(gridcolor='#2a2a40', color='#8888aa'),
            ),
            paper_bgcolor='#12121a',
            font=dict(color='#e8e8f0'), height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(18,18,26,0.8)')
        )
        st.plotly_chart(fig_3d_nl, use_container_width=True)

        st.markdown("### Tabla Comparativa de Variantes")

        import pandas as pd
        filas_resumen = []
        for v_key, res in resultados.items():
            if res is None:
                filas_resumen.append({
                    "Variante": nombres_var[v_key],
                    "Fórmula βₖ": "—",
                    "Iteraciones": "Error",
                    "f(x*) final": "—",
                    "‖∇f‖ final": "—",
                    "x* final": "—"
                })
                continue
            pts = res["puntos"]
            dat = res["datos"]
            x_final = pts[-1]
            f_final = f_test(x_final)
            g_final = np.sqrt(grad_test(x_final) @ grad_test(x_final))

            formulas = {
                "FR": "|g_{k+1}|² / |gₖ|²",
                "PRP": "g_{k+1}ᵀyₖ / |gₖ|²",
                "PR+": "max(0, g_{k+1}ᵀyₖ / |gₖ|²)",
                "HS": "g_{k+1}ᵀyₖ / dₖᵀyₖ",
                "DY": "|g_{k+1}|² / dₖᵀyₖ"
            }

            filas_resumen.append({
                "Variante": nombres_var[v_key],
                "Fórmula βₖ": formulas.get(v_key, "—"),
                "Iteraciones": len(pts) - 1,
                "f(x*) final": f"{f_final:.2e}",
                "‖∇f‖ final": f"{g_final:.2e}",
                "x* final": f"[{x_final[0]:.4f}, {x_final[1]:.4f}]"
            })

        df_nl = pd.DataFrame(filas_resumen)
        st.dataframe(df_nl, use_container_width=True, hide_index=True)

        st.markdown("### ¿Cuál variante es mejor?")
        st.markdown("""
        <div class="highlight-box">
            No hay una "mejor" universal — depende del problema:<br><br>
            • <strong style="color:#ff6b6b;">Fletcher-Reeves (FR)</strong>: La más simple. Tiene
            convergencia global demostrada, pero puede <strong>estancarse</strong> (jamming) —
            dar pasos muy pequeños sin progresar.<br><br>
            • <strong style="color:#00d4aa;">Polak-Ribière+ (PR+)</strong>: Soluciona el estancamiento
            forzando β ≥ 0. Cuando β resulta negativo, el método se <strong>reinicia</strong>
            automáticamente al gradiente puro. Es la variante más usada en la práctica.<br><br>
            • <strong style="color:#ffcb6b;">Hestenes-Stiefel (HS)</strong>: Satisface la condición de
            conjugación exacta d_{k+1}ᵀyₖ = 0. Buen rendimiento pero puede ser inestable.<br><br>
            • <strong style="color:#ff9ff3;">Dai-Yuan (DY)</strong>: Convergencia global demostrada
            con sólo condiciones de Wolfe. Robusta pero a veces más lenta.<br><br>
            • <strong>Hager-Zhang (CG_DESCENT)</strong>: La más avanzada — garantiza descenso suficiente
            sin importar la precisión de la búsqueda de línea. Es el estado del arte para problemas de gran escala.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
            Dirección no lineal: &nbsp; d_{k+1} = −g_{k+1} + βₖ · dₖ
            <br><small style="color:#8888aa;">donde yₖ = g_{k+1} − gₖ (cambio en el gradiente)</small>
        </div>
        """, unsafe_allow_html=True)


elif seccion == "Código Python":
    st.markdown("## Implementación en Python")

    st.markdown("### Función principal")

    codigo = '''import numpy as np

def gradiente_conjugado(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Método de Descenso del Gradiente Conjugado para Ax = b

    Parámetros:
        A        : Matriz simétrica definida positiva (n x n)
        b        : Vector del lado derecho (n,)
        x0       : Punto inicial (default: vector cero)
        tol      : Tolerancia para el criterio de paro
        max_iter : Número máximo de iteraciones

    Retorna:
        x         : Vector solución
        historial : Lista de puntos visitados
    """
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()

    rs_old = r @ r
    historial = [x.copy()]

    for k in range(max_iter):
        Ap = A @ p

        alpha = rs_old / (p @ Ap)

        x = x + alpha * p
        historial.append(x.copy())

        r = r - alpha * Ap

        rs_new = r @ r

        if np.sqrt(rs_new) < tol:
            print(f"Convergencia en {k+1} iteraciones")
            break

        beta = rs_new / rs_old

        p = r + beta * p

        rs_old = rs_new

    return x, historial


A = np.array([
    [ 3, -1, -2],
    [-1,  4, -3],
    [-2, -3,  6]
], dtype=float)

b = np.array([1, 0, 3], dtype=float)

x_sol, hist = gradiente_conjugado(A, b)

print(f"Solución:      x = {x_sol}")
print(f"Esperado:      x = [4.3636, 4.0909, 4.0]")
print(f"Verificación: Ax = {A @ x_sol}")
print(f"b original:    b = {b}")


A2 = np.array([[ 4, -1],
               [-1,  4]], dtype=float)
b2 = np.array([10, 10], dtype=float)

x2, hist2 = gradiente_conjugado(A2, b2)
print(f"\\nRed de agua — Presiones: P1={x2[0]:.4f}, P2={x2[1]:.4f}")
'''

    st.code(codigo, language="python")

    st.markdown("### Salida esperada")
    st.code("""Convergencia en 3 iteraciones
Solución:      x = [4.36363636 4.09090909 4.        ]
Esperado:      x = [4.3636, 4.0909, 4.0]
Verificación: Ax = [1. 0. 3.]
b original:    b = [1. 0. 3.]

Convergencia en 2 iteraciones
Red de agua — Presiones: P1=3.3333, P2=3.3333""", language="text")

    st.markdown("---")
    st.markdown("### Ejecución en Vivo")
    if st.button("Ejecutar Ejemplo del Documento", use_container_width=True):
        A = np.array([[3, -1, -2], [-1, 4, -3], [-2, -3, 6]], dtype=float)
        b = np.array([1, 0, 3], dtype=float)
        x_sol, hist = gradiente_conjugado(A, b)

        st.success(f"Convergencia en **{len(hist) - 1} iteraciones**")
        st.markdown(f"**Solución:** x = [{x_sol[0]:.4f}, {x_sol[1]:.4f}, {x_sol[2]:.4f}]")
        st.markdown(f"**Verificación:** Ax = {(A @ x_sol).round(10)}")


elif seccion == "Ejemplo Numérico":
    st.markdown("## Ejemplo Paso a Paso — Sistema 3x3")

    st.markdown("### Planteamiento")

    col_a, col_eq, col_x, col_eq2, col_b = st.columns([3, 1, 1, 1, 1])
    with col_a:
        st.markdown("**A =**")
        st.markdown("""
        | | c1 | c2 | c3 |
        |---|---|---|---|
        | f1 | 3 | -1 | -2 |
        | f2 | -1 | 4 | -3 |
        | f3 | -2 | -3 | 6 |
        """)
    with col_eq:
        st.markdown("<br><br><h2>·</h2>", unsafe_allow_html=True)
    with col_x:
        st.markdown("**x =**")
        st.markdown("x₁\n\nx₂\n\nx₃")
    with col_eq2:
        st.markdown("<br><br><h2>=</h2>", unsafe_allow_html=True)
    with col_b:
        st.markdown("**b =**")
        st.markdown("1\n\n0\n\n3")

    st.info("**det(A) = 11 > 0** → Matriz definida positiva ")

    st.markdown("### Iteraciones del Algoritmo")

    A = np.array([[3, -1, -2], [-1, 4, -3], [-2, -3, 6]], dtype=float)
    b = np.array([1, 0, 3], dtype=float)
    x_sol, hist = gradiente_conjugado(A, b)

    import pandas as pd

    rows = []
    for h in hist:
        x_str = f"[{h['x'][0]:.4f}, {h['x'][1]:.4f}, {h['x'][2]:.4f}]"
        alpha_str = f"{h['alpha']:.4f}" if h['alpha'] is not None else "—"
        beta_str = f"{h['beta']:.4e}" if h['beta'] is not None and h['beta'] > 0 else ("—" if h['alpha'] is None else f"{h['beta']:.4f}")
        rs_str = f"{h['rs']:.4e}" if h['rs'] < 0.001 else f"{h['rs']:.4f}"
        rows.append({
            "Iteración": h['k'],
            "αₖ": alpha_str,
            "xₖ": x_str,
            "‖rₖ‖²": rs_str,
            "βₖ": beta_str
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="highlight-box">
         El método converge en <strong>{len(hist)-1} iteraciones</strong> (n = 3 dimensiones),
        confirmando la teoría de convergencia finita para sistemas lineales.<br><br>
        <strong>Solución final:</strong> x = [{x_sol[0]:.4f}, {x_sol[1]:.4f}, {x_sol[2]:.4f}]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Convergencia del Residuo")
    residuos = [h['rs'] for h in hist]
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=list(range(len(residuos))), y=residuos,
        mode='lines+markers',
        line=dict(color='#00d4aa', width=3),
        marker=dict(size=10, color='#00d4aa'),
        name='‖rₖ‖²'
    ))
    fig_conv.update_layout(
        xaxis_title="Iteración k",
        yaxis_title="‖rₖ‖²",
        yaxis_type="log",
        paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
        font=dict(color='#e8e8f0'),
        height=350,
        xaxis=dict(gridcolor='#2a2a40', dtick=1),
        yaxis=dict(gridcolor='#2a2a40'),
    )
    st.plotly_chart(fig_conv, use_container_width=True)


elif seccion == "Calculadora Interactiva":
    st.markdown("## Calculadora — Resuelve tu propio sistema")

    st.markdown("Ingresa una matriz **simétrica definida positiva** y un vector b.")

    n = st.selectbox("Dimensión del sistema (n):", [2, 3, 4, 5], index=1)

    st.markdown("### Matriz A (simétrica)")
    cols_a = st.columns(n)
    A_input = np.zeros((n, n))

    defaults = {
        (0,0): 3, (0,1): -1, (0,2): -2,
        (1,0): -1, (1,1): 4, (1,2): -3,
        (2,0): -2, (2,1): -3, (2,2): 6
    }

    for i in range(n):
        for j in range(n):
            with cols_a[j]:
                default = defaults.get((i, j), (1.0 if i == j else 0.0))
                if j >= i:
                    val = st.number_input(
                        f"A[{i+1},{j+1}]", value=float(default),
                        key=f"a_{i}_{j}", step=0.5
                    )
                    A_input[i, j] = val
                    A_input[j, i] = val
                else:
                    A_input[i, j] = A_input[j, i]
                    st.text_input(
                        f"A[{i+1},{j+1}]", value=f"{A_input[i,j]:.1f}",
                        key=f"a_{i}_{j}", disabled=True
                    )

    st.markdown("### Vector b")
    cols_b = st.columns(n)
    b_input = np.zeros(n)
    b_defaults = [1, 0, 3, 0, 0]
    for i in range(n):
        with cols_b[i]:
            b_input[i] = st.number_input(
                f"b[{i+1}]", value=float(b_defaults[i] if i < len(b_defaults) else 0),
                key=f"b_{i}", step=1.0
            )

    tol = st.number_input("Tolerancia (ε)", value=1e-10, format="%.1e")

    if st.button("Resolver con Descenso del Gradiente Conjugado", use_container_width=True):
        eigvals = np.linalg.eigvalsh(A_input)
        if not np.all(eigvals > 0):
            st.error(f"La matriz NO es definida positiva. Valores propios: {eigvals.round(4)}")
        else:
            det = np.linalg.det(A_input)
            kappa = eigvals.max() / eigvals.min()

            st.info(f"det(A) = {det:.4f} · κ(A) = {kappa:.4f} · Valores propios: {eigvals.round(4)}")

            x_sol, hist = gradiente_conjugado(A_input, b_input, tol=tol)

            st.success(f"Convergencia en **{len(hist)-1} iteraciones**")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**Solución x:**")
                for i, v in enumerate(x_sol):
                    st.markdown(f"  x[{i+1}] = **{v:.6f}**")
            with col_r2:
                st.markdown("**Verificación Ax:**")
                Ax = A_input @ x_sol
                for i, v in enumerate(Ax):
                    st.markdown(f"  (Ax)[{i+1}] = {v:.6f}  ← b[{i+1}] = {b_input[i]}")

            st.markdown("### Iteraciones detalladas")
            import pandas as pd
            rows = []
            for h in hist:
                x_str = "[" + ", ".join(f"{v:.4f}" for v in h['x']) + "]"
                rows.append({
                    "k": h['k'],
                    "αₖ": f"{h['alpha']:.6f}" if h['alpha'] is not None else "—",
                    "xₖ": x_str,
                    "‖rₖ‖²": f"{h['rs']:.2e}",
                    "βₖ": f"{h['beta']:.6f}" if h['beta'] is not None else "—"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            residuos = [h['rs'] for h in hist]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(residuos))), y=residuos,
                mode='lines+markers',
                line=dict(color='#00d4aa', width=3),
                marker=dict(size=8, color='#00d4aa')
            ))
            fig.update_layout(
                title="Convergencia del Residuo",
                xaxis_title="Iteración", yaxis_title="‖rₖ‖²",
                yaxis_type="log",
                paper_bgcolor='#12121a', plot_bgcolor='#0a0a0f',
                font=dict(color='#e8e8f0'), height=350,
                xaxis=dict(gridcolor='#2a2a40', dtick=1),
                yaxis=dict(gridcolor='#2a2a40'),
            )
            st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555570; font-size:0.8rem;'>"
    "DAT-252 Métodos Numéricos II — UMSA Carrera de Informática — La Paz, Bolivia 2026<br>"
    "Método de Descenso del Gradiente Conjugado · Ian Ezequiel Salinas Condori"
    "</p>",
    unsafe_allow_html=True
)