from pathlib import Path
from io import BytesIO
import json

from PIL import Image

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="AULA | Alerta Temprana",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelo"

MODEL_PATH = MODEL_DIR / "modelo_final.joblib"
X_PATH = MODEL_DIR / "X_test.csv"
Y_PATH = MODEL_DIR / "y_test.csv"
INFO_PATH = MODEL_DIR / "informacion_modelo.json"

# Paleta (la misma en CSS y en matplotlib)
INK = "#20251F"      # carbón
VIOLET = "#526B50"   # verde salvia oscuro
CYAN = "#A7B99F"     # verde salvia claro
CORAL = "#A45F55"    # terracota sobrio para alertas
MINT = "#789477"     # verde de acento
PAPER = "#F7F7F1"
LINE = "#E6E8E0"
MUTED = "#73796F"

# =========================================================
# ESTILOS
# =========================================================

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {{
    --ink: {INK};
    --violet: {VIOLET};
    --cyan: {CYAN};
    --coral: {CORAL};
    --mint: {MINT};
    --paper: {PAPER};
    --line: {LINE};
    --muted: {MUTED};
}}

html, body, [class*="css"], .stMarkdown, p, li {{
    font-family: 'DM Sans', sans-serif;
}}

h1, h2, h3, h4 {{
    font-family: 'Manrope', sans-serif !important;
    color: var(--ink);
    letter-spacing: -0.01em;
}}

.stApp {{
    background: var(--paper);
}}

.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] {{
    background: #20251F;
    border-right: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{
    color: #E7E9FA !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.10);
}}
.brand {{
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: 0.02em;
}}
.brand span {{
    color: var(--cyan) !important;
}}
.side-chip {{
    display: inline-block;
    background: rgba(25,181,201,0.14);
    border: 1px solid rgba(25,181,201,0.45);
    border-radius: 999px;
    padding: 4px 12px;
    font-family: 'DM Sans', monospace;
    font-size: 0.78rem;
}}
.side-stats {{
    font-family: 'DM Sans', monospace;
    font-size: 0.8rem;
    line-height: 1.9;
    opacity: 0.9;
}}

/* ---------- Encabezado temático ---------- */
.hero {{
    background: linear-gradient(120deg, #20251F 0%, #344638 60%, #526B50 100%);
    border-radius: 22px;
    padding: 26px 34px;
    margin-bottom: 24px;
}}
.hero-text {{ max-width: 100%; }}
.hero-tag {{
    display: inline-block;
    font-family: 'DM Sans', monospace;
    font-size: 0.78rem;
    color: #DCE9D5;
    border: 1px solid rgba(159,232,241,0.35);
    border-radius: 6px;
    padding: 3px 10px;
    margin-bottom: 12px;
}}
.hero h1 {{
    color: #FFFFFF !important;
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 8px 0;
}}
.hero p {{
    color: #E8EDE3;
    font-size: 1rem;
    line-height: 1.55;
    margin: 0;
}}

/* ---------- Títulos de sección ---------- */
.section-title {{
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
    font-weight: 700;
    font-size: 1.35rem;
    margin: 6px 0 2px 0;
}}
.small-muted {{
    color: var(--muted);
    font-size: 0.9rem;
}}

/* ---------- Tarjetas de métricas ---------- */
.metric-card {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 4px solid var(--accent, var(--violet));
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 118px;
}}
.metric-label {{
    color: var(--muted);
    font-size: 0.86rem;
    font-weight: 500;
}}
.metric-value {{
    color: var(--ink);
    font-family: 'DM Sans', monospace;
    font-size: 1.95rem;
    font-weight: 600;
    margin-top: 6px;
}}
.metric-note {{
    color: #858B80;
    font-size: 0.78rem;
    margin-top: 2px;
}}

/* ---------- Contenedores con borde ---------- */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: #FFFFFF;
    border-color: var(--line) !important;
    border-radius: 14px !important;
}}

/* ---------- Métricas nativas ---------- */
[data-testid="stMetricValue"] {{
    font-family: 'DM Sans', monospace;
    color: var(--ink);
}}
[data-testid="stMetricLabel"] {{
    color: var(--muted);
}}

/* ---------- Resultado del modelo ---------- */
.veredicto {{
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    border: 1px solid;
}}
.veredicto.riesgo {{
    background: #FBEFEC; border-color: #EBC7C0; color: #873F3B;
}}
.veredicto.sin-riesgo {{
    background: #EAF1E7; border-color: #C9D8C3; color: #36543A;
}}
.veredicto h3 {{
    margin: 0 0 4px 0;
    font-size: 1.25rem;
    color: inherit !important;
}}
.veredicto p {{
    margin: 0;
    font-size: 0.92rem;
}}

/* Medidor de probabilidad */
.medidor-wrap {{
    margin: 10px 0 4px 0;
}}
.medidor-valor {{
    font-family: 'DM Sans', monospace;
    font-size: 2.3rem;
    font-weight: 600;
    color: var(--ink);
}}
.medidor-pista {{
    position: relative;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, {CYAN} 0%, {VIOLET} 55%, {CORAL} 100%);
    margin-top: 10px;
}}
.medidor-umbral {{
    position: absolute;
    top: -6px;
    left: 50%;
    width: 2px;
    height: 24px;
    background: var(--ink);
    opacity: 0.55;
}}
.medidor-punto {{
    position: absolute;
    top: -5px;
    width: 22px;
    height: 22px;
    margin-left: -11px;
    border-radius: 50%;
    background: #FFFFFF;
    border: 4px solid var(--ink);
}}
.medidor-escala {{
    display: flex;
    justify-content: space-between;
    font-family: 'DM Sans', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 12px;
}}

/* ---------- Botones ---------- */
div.stButton > button {{
    background: #526B50;
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.1rem;
}}
div.stButton > button:hover {{
    background: #40563F;
    color: white;
}}

.footer {{
    color: #858B80;
    font-size: 0.8rem;
    text-align: center;
    padding: 28px 0 4px 0;
}}

/* =====================================================
   CONTRASTE DEFINITIVO PARA STREAMLIT
   ===================================================== */

/* Fondo y color base del área principal */
:root {{ color-scheme: light !important; }}
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {{
    background: #F5F5F0 !important;
    color: #20251F !important;
}}

/* Streamlit renderiza gran parte del texto dentro de estos contenedores.
   Se aplica el color directamente a los elementos reales, no solo al padre. */
section.main [data-testid="stMarkdownContainer"],
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stMarkdownContainer"] strong,
section.main [data-testid="stMarkdownContainer"] em,
section.main [data-testid="stText"],
section.main [data-testid="stCaptionContainer"],
section.main [data-testid="stCaptionContainer"] p,
section.main label,
section.main .stRadio,
section.main .stCheckbox {{
    color: #343B32 !important;
    -webkit-text-fill-color: #343B32 !important;
}}

/* Encabezados markdown y títulos generados por Streamlit */
section.main h1,
section.main h2,
section.main h3,
section.main h4,
section.main h5,
section.main h6,
section.main [data-testid="stHeading"],
section.main [data-testid="stHeading"] *,
section.main .stHeading,
section.main .stHeading * {{
    color: #20251F !important;
    -webkit-text-fill-color: #20251F !important;
}}

/* Texto auxiliar y pies de texto */
section.main [data-testid="stCaptionContainer"],
section.main [data-testid="stCaptionContainer"] *,
section.main .small-muted,
section.main .metric-label,
section.main .metric-note,
section.main .medidor-escala,
section.main .footer {{
    color: #687064 !important;
    -webkit-text-fill-color: #687064 !important;
}}

/* Tarjetas */
section.main .metric-card,
section.main [data-testid="stVerticalBlockBorderWrapper"],
section.main [data-testid="stMetric"],
section.main [data-testid="stExpander"],
section.main [data-testid="stDataFrame"] {{
    background: #FFFFFF !important;
    color: #343B32 !important;
}}

/* Métricas */
section.main .metric-value,
section.main [data-testid="stMetricValue"],
section.main [data-testid="stMetricValue"] *,
section.main [data-testid="stMetricLabel"],
section.main [data-testid="stMetricLabel"] * {{
    color: #20251F !important;
    -webkit-text-fill-color: #20251F !important;
}}

/* Selectbox: evita el campo negro y la etiqueta clara */
section.main [data-testid="stSelectbox"] label,
section.main [data-testid="stSelectbox"] label * {{
    color: #343B32 !important;
    -webkit-text-fill-color: #343B32 !important;
}}
section.main [data-baseweb="select"] > div,
section.main [data-baseweb="select"] input,
section.main [data-baseweb="select"] span,
section.main [data-baseweb="select"] svg {{
    background-color: #FFFFFF !important;
    color: #293329 !important;
    fill: #293329 !important;
    -webkit-text-fill-color: #293329 !important;
}}
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
[role="option"] {{
    background-color: #FFFFFF !important;
    color: #293329 !important;
    -webkit-text-fill-color: #293329 !important;
}}

/* Inputs */
section.main input,
section.main textarea,
section.main [data-baseweb="input"] *,
section.main [data-baseweb="textarea"] * {{
    background: #FFFFFF !important;
    color: #293329 !important;
    -webkit-text-fill-color: #293329 !important;
}}

/* Dataframes y tablas */
section.main [data-testid="stDataFrame"] *,
section.main [data-testid="stTable"] * {{
    color: #293329 !important;
    -webkit-text-fill-color: #293329 !important;
}}

/* Alertas */
section.main [data-testid="stAlert"],
section.main [data-testid="stAlert"] p,
section.main [data-testid="stAlert"] span,
section.main [data-testid="stAlert"] div {{
    color: #343B32 !important;
    -webkit-text-fill-color: #343B32 !important;
}}

/* Botones */
section.main div.stButton > button,
section.main div.stButton > button * {{
    background: #526B50 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

/* Barra lateral: mantiene el contraste claro sobre azul oscuro */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {{
    color: #F4F5EF !important;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: #C7D0C1 !important;
}}

/* Excepciones deliberadas: portada y resultado */
section.main .hero,
section.main .hero * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
section.main .hero .hero-tag {{
    color: #DCE9D5 !important;
    -webkit-text-fill-color: #DCE9D5 !important;
}}
section.main .hero p {{
    color: #E8EDE3 !important;
    -webkit-text-fill-color: #E8EDE3 !important;
}}
section.main .veredicto.riesgo,
section.main .veredicto.riesgo * {{
    color: #873F3B !important;
    -webkit-text-fill-color: #873F3B !important;
}}
section.main .veredicto.sin-riesgo,
section.main .veredicto.sin-riesgo * {{
    color: #36543A !important;
    -webkit-text-fill-color: #36543A !important;
}}
section.main .medidor-valor {{
    color: #20251F !important;
    -webkit-text-fill-color: #20251F !important;
}}

/* ===================================================== */

/* ---------- AULA · refinamiento visual Keyvo ---------- */
:root {{
    color-scheme: light !important;
    --aula-forest: #344638;
    --aula-sage: #789477;
    --aula-sage-light: #DDE7D8;
    --aula-paper: #F7F7F1;
    --aula-ink: #20251F;
    --aula-line: #E6E8E0;
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
section.main {{ background: var(--aula-paper) !important; }}
.block-container {{ max-width: 1440px; padding-top: 1.8rem; }}
[data-testid="stSidebar"] {{ background: #26372D !important; }}
[data-testid="stSidebar"] * {{ color: #F3F5EF !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{ color: #C8D2C4 !important; }}
.brand {{ font-size: 1.85rem; letter-spacing: -.06em; }}
.side-chip {{ background: rgba(180,197,173,.14); border-color: rgba(180,197,173,.4); }}
.hero {{
    background: linear-gradient(115deg,#26372D 0%,#344638 65%,#526B55 100%);
    border-radius: 24px; padding: 30px 36px; box-shadow: 0 12px 32px rgba(38,55,45,.10);
}}
.hero-tag {{ border-radius: 99px; border-color: rgba(221,231,216,.35); letter-spacing: .08em; }}
.hero h1 {{ font-size: clamp(1.8rem,3vw,2.5rem); letter-spacing: -.035em; }}
.section-title {{ font-size: 1.5rem; letter-spacing: -.025em; font-weight: 800; }}
.metric-card {{
    border: 1px solid var(--aula-line); border-left: 0; border-radius: 17px;
    padding: 19px 21px; min-height: 120px; box-shadow: 0 5px 18px rgba(38,55,45,.035);
    position: relative; overflow: hidden;
}}
.metric-card:before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px; background: var(--accent,var(--aula-sage));
}}
.metric-value {{ font-family: 'Manrope',sans-serif; font-size: 2rem; font-weight: 800; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 18px !important; border-color: var(--aula-line) !important;
    box-shadow: 0 5px 18px rgba(38,55,45,.035);
}}
div.stButton > button {{
    background: #526B55; border: 1px solid #526B55; border-radius: 11px;
    font-weight: 700; padding: .65rem 1.15rem;
}}
div.stButton > button:hover {{ background: #344638; border-color: #344638; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    border-radius: 10px !important; border-color: var(--aula-line) !important;
}}
.stTabs [data-baseweb="tab"] {{ font-weight: 700; }}
.footer {{ padding-top: 34px; }}
section.main h1, section.main h2, section.main h3, section.main h4,
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stCaptionContainer"] {{ color: var(--aula-ink); }}
section.main .hero, section.main .hero * {{ color: #FFFFFF !important; }}
section.main .hero p {{ color: #E6ECE2 !important; }}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def html(contenido: str):
    """Renderiza HTML sin que Markdown lo interprete como bloque de código."""
    limpio = "\n".join(l.strip() for l in contenido.splitlines() if l.strip())
    st.markdown(limpio, unsafe_allow_html=True)


# =========================================================
# ELEMENTOS VISUALES
# =========================================================

def tarjeta_metrica(titulo, valor, nota, color):
    return f"""
    <div class="metric-card" style="--accent:{color}">
        <div class="metric-label">{titulo}</div>
        <div class="metric-value">{valor}</div>
        <div class="metric-note">{nota}</div>
    </div>
    """


def medidor(probabilidad: float) -> str:
    pos = min(max(probabilidad, 0.0), 1.0) * 100
    return f"""
    <div class="medidor-wrap">
        <div class="medidor-valor">{probabilidad:.1%}</div>
        <div class="small-muted">Probabilidad estimada de riesgo</div>
        <div class="medidor-pista">
            <div class="medidor-umbral"></div>
            <div class="medidor-punto" style="left:{pos:.1f}%"></div>
        </div>
        <div class="medidor-escala"><span>0%</span><span>umbral 50%</span><span>100%</span></div>
    </div>
    """


def estilo_ejes(fig, ax):
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


MAPA_CONFUSION = LinearSegmentedColormap.from_list(
    "aula", ["#FFFFFF", "#DDE7D8", "#789477", "#344638"]
)

# =========================================================
# CARGA DEL MODELO Y LOS DATOS
# =========================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_datos():
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).iloc[:, 0]
    return X, y


@st.cache_data
def cargar_informacion():
    with open(INFO_PATH, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


# =========================================================
# VALIDACIÓN DE ARCHIVOS
# =========================================================

archivos_necesarios = [MODEL_PATH, X_PATH, Y_PATH, INFO_PATH]
faltantes = [str(p.name) for p in archivos_necesarios if not p.exists()]

if faltantes:
    st.error("No se encontraron todos los archivos del modelo.")
    st.write("Archivos faltantes:", faltantes)
    st.info("Verifica que estén dentro de la carpeta modelo/.")
    st.stop()

try:
    modelo = cargar_modelo()
    X_test, y_test = cargar_datos()
    info = cargar_informacion()

    if list(X_test.columns) != list(modelo.feature_names_in_):
        st.warning(
            "El orden o los nombres de las variables de prueba "
            "no coinciden con los esperados por el modelo."
        )
        st.stop()

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

except Exception as error:
    st.error("No fue posible cargar o ejecutar el modelo.")
    st.exception(error)
    st.stop()

# =========================================================
# MÉTRICAS
# =========================================================

accuracy = accuracy_score(y_test, predicciones)
precision = precision_score(y_test, predicciones, zero_division=0)
recall = recall_score(y_test, predicciones, zero_division=0)
f1 = f1_score(y_test, predicciones, zero_division=0)
f2 = fbeta_score(y_test, predicciones, beta=2, zero_division=0)
auc = roc_auc_score(y_test, probabilidades)

matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
vn, fp, fn, vp = matriz.ravel()
especificidad = vn / (vn + fp) if (vn + fp) else 0.0

# =========================================================
# BARRA LATERAL
# =========================================================

with st.sidebar:
    html('<div class="brand">AULA<span>.</span></div>')
    st.caption("ACADEMIC INTELLIGENCE")
    st.divider()

    pagina = st.radio(
        "Navegación",
        [
            "Inicio",
            "Demostración del modelo",
            "Rendimiento y métricas",
            "Interpretabilidad SHAP",
            "Acerca del proyecto",
        ],
    )

    st.divider()

    st.markdown("**Modelo activo**")
    html('<span class="side-chip">HistGradientBoosting</span>')
    st.write("")
    html(
        f"""
        <div class="side-stats">
        variables&nbsp;&nbsp;&nbsp;&nbsp;{info['numero_variables']}<br>
        test&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{len(X_test):,} registros<br>
        semilla&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{info['semilla']}
        </div>
        """
    )

    st.divider()
    st.caption("Machine Learning II · Proyecto académico")

# =========================================================
# ENCABEZADO DE LA SECCIÓN ACTIVA
# =========================================================

encabezados = {
    "Inicio": (
        "AULA · ACADEMIC INTELLIGENCE",
        "Comprender el riesgo. Acompañar trayectorias.",
        "Una visión clara del desempeño y los hallazgos del modelo."
    ),
    "Demostración del modelo": (
        "AULA · STUDENT INSIGHTS",
        "Explora una predicción.",
        "Explora cómo el modelo clasifica registros del conjunto de prueba."
    ),
    "Rendimiento y métricas": (
        "AULA · MODEL PERFORMANCE",
        "El rendimiento, en perspectiva.",
        "Indicadores, errores y capacidad de discriminación del clasificador."
    ),
    "Interpretabilidad SHAP": (
        "AULA · MODEL EXPLAINABILITY",
        "Entender cómo decide el modelo.",
        "Importancia global y contribuciones de las variables al resultado del modelo."
    ),
    "Acerca del proyecto": (
        "AULA · RESEARCH",
        "Datos que orientan la prevención.",
        "Contexto, metodología, resultados y límites de esta investigación."
    ),
}

etiqueta, titulo_pagina, subtitulo_pagina = encabezados[pagina]
html(
    f"""
    <div class="hero" style="padding:26px 34px; margin-bottom:24px;">
        <div class="hero-text" style="max-width:100%;">
            <span class="hero-tag">{etiqueta}</span>
            <h1>{titulo_pagina}</h1>
            <p>{subtitulo_pagina}</p>
        </div>
    </div>
    """
)

# =========================================================
# PÁGINA: INICIO
# =========================================================

if pagina == "Inicio":

    html('<div class="section-title">Panel de seguimiento del modelo</div>')
    html('<p class="small-muted">Indicadores calculados sobre el conjunto de prueba.</p>')

    with st.container(border=True):
        st.markdown("### Detección temprana del riesgo de deserción universitaria")
        st.write(
            "AULA es un prototipo académico que explora el uso de Machine Learning "
            "para reconocer registros clasificados con riesgo y aportar información "
            "que pueda orientar acciones preventivas de acompañamiento."
        )
        contexto_a, contexto_b, contexto_c = st.columns(3)
        contexto_a.markdown("**Dataset**")
        contexto_a.caption("Risk of Desertion in Undergraduate UCC · Mendeley Data")
        contexto_b.markdown("**Objetivo**")
        contexto_b.caption("Clasificación binaria del nivel de riesgo asignado en la base")
        contexto_c.markdown("**Estrategia**")
        contexto_c.caption("Comparación de modelos, validación cruzada anidada y F2")

    c1, c2, c3, c4 = st.columns(4)

    metricas = [
        (c1, "Exactitud", f"{accuracy:.1%}", "Predicciones correctas", VIOLET),
        (c2, "Sensibilidad", f"{recall:.1%}", "Casos de riesgo detectados", CORAL),
        (c3, "Precisión", f"{precision:.1%}", "Alertas correctas", CYAN),
        (c4, "ROC-AUC", f"{auc:.3f}", "Capacidad de discriminación", MINT),
    ]

    for col, titulo, valor, nota, color in metricas:
        with col:
            html(tarjeta_metrica(titulo, valor, nota, color))

    st.write("")

    izquierda, derecha = st.columns([1.2, 0.8])

    with izquierda:
        with st.container(border=True):
            st.markdown("### ¿Qué hace AULA?")
            st.write(
                """
                AULA utiliza un modelo de aprendizaje automático para
                estimar la pertenencia de un registro a la categoría
                de riesgo de deserción.

                Su propósito es apoyar el análisis académico y facilitar
                la identificación de situaciones que podrían requerir
                acompañamiento institucional.
                """
            )
            st.markdown("**El sistema integra:**")
            st.write("• Procesamiento de variables académicas y contextuales.")
            st.write("• Un modelo HistGradientBoosting entrenado.")
            st.write("• Estimación de probabilidad de riesgo.")
            st.write("• Evaluación mediante métricas de clasificación.")

    with derecha:
        with st.container(border=True):
            st.markdown("### Modelo seleccionado")
            st.markdown("**Boosting (HistGB)**")
            st.caption("HistGradientBoostingClassifier")
            st.divider()
            st.metric("Variables de entrada", info["numero_variables"])
            st.metric("Registros de prueba", f"{len(X_test):,}")
            st.metric(
                "Registros clasificados como riesgo",
                f"{int((predicciones == 1).sum()):,}",
            )

    st.write("")
    st.info(
        "Importante: los indicadores corresponden al conjunto de prueba "
        "del proyecto. No representan resultados de una población "
        "universitaria en tiempo real."
    )

# =========================================================
# PÁGINA: DEMOSTRACIÓN
# =========================================================

elif pagina == "Demostración del modelo":

    html('<div class="section-title">Conoce y prueba AULA</div>')
    st.markdown(
        "Descubre de manera sencilla cómo funciona el modelo y explora "
        "cómo cambian sus estimaciones al modificar un perfil académico."
    )

    # ── 1. Demostración guiada ────────────────────────────────────────────
    st.markdown("## 01 · Demostración guiada")
    st.markdown("### ¿Cómo funciona AULA?")
    st.write(
        "AULA es una herramienta académica que utiliza Machine Learning para "
        "estimar si un registro presenta características asociadas con riesgo "
        "de deserción, según los patrones aprendidos de los datos."
    )

    pasos = st.columns(4, gap="medium")
    pasos_info = [
        ("01", "Información", "El sistema recibe información académica y de contexto."),
        ("02", "Aprendizaje", "El modelo aprende patrones a partir de ejemplos previamente clasificados."),
        ("03", "Estimación", "Analiza un nuevo perfil y calcula una probabilidad de pertenecer al grupo con riesgo."),
        ("04", "Resultado", "Presenta la estimación para apoyar la identificación temprana y el acompañamiento."),
    ]
    for columna, (numero, titulo, descripcion) in zip(pasos, pasos_info):
        with columna:
            with st.container(border=True):
                st.markdown(f"**{numero}**")
                st.markdown(f"### {titulo}")
                st.caption(descripcion)

    with st.container(border=True):
        st.markdown("### ¿Qué modelo utiliza?")
        st.markdown("**HistGradientBoosting**")
        st.write(
            "Es un algoritmo que combina varios árboles de decisión. Cada árbol "
            "contribuye al resultado y, en conjunto, el modelo aprende a reconocer "
            "patrones más complejos que los que podría identificar una sola regla."
        )
        st.caption(
            "AULA fue entrenado y evaluado con datos del proyecto. En la evaluación "
            "final obtuvo Recall de 0,971 y F2-score de 0,974."
        )

    with st.container(border=True):
        st.markdown("### ¿Cómo interpretar el resultado?")
        izquierda, derecha = st.columns(2)
        with izquierda:
            st.markdown("**Probabilidad inferior al 50 %**")
            st.success("Clasificación del modelo: sin riesgo")
            st.caption("El resultado queda por debajo del umbral de clasificación.")
        with derecha:
            st.markdown("**Probabilidad igual o superior al 50 %**")
            st.warning("Clasificación del modelo: con riesgo")
            st.caption("El resultado alcanza o supera el umbral de clasificación.")
        st.caption(
            "La probabilidad se refiere a la clase de riesgo definida en el dataset; "
            "no representa una certeza de deserción real ni sustituye la valoración "
            "de profesionales de la institución."
        )

    st.divider()

    # ── 2. Simulador interactivo ──────────────────────────────────────────
    st.markdown("## 02 · Simulador interactivo")
    st.markdown("### Construye un perfil académico")
    st.write(
        "Responde las preguntas y observa la estimación del modelo. Las preguntas "
        "se presentan en lenguaje natural y sus opciones corresponden a las "
        "categorías codificadas en el dataset."
    )

    definiciones_grupos = [
        {
            "titulo": "Experiencia académica",
            "items": [
                ("Autopercepción académica",
                 "¿En qué nivel te identificas con el perfil requerido para tu carrera?",
                 "autopercepcion_", "categorica"),
                ("Dificultades de tiempo",
                 "¿Con qué frecuencia experimentas dificultades para gestionar el tiempo de tus estudios?",
                 "dificultades_t_", "categorica"),
                ("Exigencias académicas",
                 "¿Qué efecto consideras que tiene el estrés académico en la deserción universitaria?",
                 "e_acad_", "categorica"),
                ("Horas semanales de estudio",
                 "¿Cuántas horas dedicas a estudiar semanalmente, sin contar las horas de clase?",
                 "h_semanal_", "categorica"),
            ],
        },
        {
            "titulo": "Acompañamiento y entorno universitario",
            "items": [
                ("Apoyo familiar",
                 "¿Qué tanto consideras que la falta de apoyo familiar podría afectar tu decisión de abandonar la universidad?",
                 "a_familiar_", "categorica"),
                ("Disposición de profesores",
                 "¿Cómo consideras el apoyo que tus profesores te brindan dentro y fuera del aula?",
                 "a_profesores_", "categorica"),
                ("Percepción del nivel académico de los profesores",
                 "¿Consideras que el nivel académico de los profesores es adecuado?",
                 "per_profesores_", "escala"),
                ("Recomendación de la universidad",
                 "¿Qué tan probable es que recomiendes la universidad a otras personas?",
                 "recomendación_u_", "escala"),
            ],
        },
    ]

    # Match each prefix exactly. This prevents labels such as "t_Siempre"
    # or "familiar_Nada" from leaking into the interface.
    grupos_disponibles = {}
    for grupo in definiciones_grupos:
        for titulo, pregunta, prefijo, tipo in grupo["items"]:
            columnas = [
                col for col in X_test.columns
                if col.startswith(prefijo)
            ]
            if columnas:
                grupos_disponibles[titulo] = {
                    "pregunta": pregunta,
                    "prefijo": prefijo,
                    "columnas": columnas,
                    "tipo": tipo,
                }

    # Use a real test record near the classification threshold as a neutral
    # reference instead of always selecting the first non-risk record.
    indice_base = int(np.argmin(np.abs(probabilidades - 0.5)))
    perfil_base = X_test.iloc[[indice_base]].copy()
    prob_base = float(probabilidades[indice_base])

    selecciones = {}
    for grupo in definiciones_grupos:
        items_visibles = [
            item for item in grupo["items"]
            if item[0] in grupos_disponibles
        ]
        if not items_visibles:
            continue
        st.markdown(f"### {grupo['titulo']}")
        columnas_form = st.columns(2, gap="large")
        for i, (titulo, pregunta, prefijo, tipo) in enumerate(items_visibles):
            definicion = grupos_disponibles[titulo]
            columnas = definicion["columnas"]
            with columnas_form[i % 2]:
                st.markdown(f"**{titulo}**")
                st.caption(pregunta)
                if tipo == "escala":
                    opciones = list(range(1, 11))
                    etiquetas = {
                        col: col[len(prefijo):].strip()
                        for col in columnas
                    }
                    valores = {
                        int(etiquetas[col]): col
                        for col in columnas
                        if etiquetas[col].isdigit()
                    }
                    valor_actual = next(
                        (
                            int(etiquetas[col])
                            for col in columnas
                            if etiquetas[col].isdigit()
                            and float(perfil_base.iloc[0][col]) == 1.0
                        ),
                        min(valores) if valores else 1
                    )
                    elegido_num = st.select_slider(
                        "Selecciona un valor de 1 a 10",
                        options=opciones,
                        value=valor_actual if valor_actual in opciones else 1,
                        key=f"aula_slider_{prefijo}"
                    )
                    elegido = valores.get(elegido_num)
                    if elegido is None:
                        st.info("Esta escala no está disponible completa en las columnas del modelo.")
                        continue
                else:
                    # Strip only the exact feature prefix; retain category text.
                    etiquetas = {
                        col: col[len(prefijo):].strip()
                        for col in columnas
                    }
                    actual = next(
                        (
                            col for col in columnas
                            if float(perfil_base.iloc[0][col]) == 1.0
                        ),
                        columnas[0]
                    )
                    elegido = st.selectbox(
                        "Selecciona una respuesta",
                        options=columnas,
                        index=columnas.index(actual),
                        format_func=lambda col, mapa=etiquetas: mapa[col],
                        key=f"aula_select_{prefijo}"
                    )
                selecciones[titulo] = (columnas, elegido)

    st.caption(
        "Para conservar el formato que espera el modelo, el simulador toma como "
        "referencia un registro real del conjunto de prueba cercano al umbral de "
        "clasificación. Las respuestas elegidas reemplazan las categorías de estos "
        "ocho grupos; las demás características del registro permanecen iguales."
    )

    if st.button("Calcular estimación", type="primary", key="aula_calcular"):
        escenario = perfil_base.copy()
        for columnas, elegido in selecciones.values():
            escenario.loc[:, columnas] = 0
            escenario.loc[:, elegido] = 1

        prob_nueva = float(modelo.predict_proba(escenario)[0, 1])
        st.session_state["aula_resultado_simulador"] = {
            "probabilidad": prob_nueva,
            "probabilidad_base": prob_base,
            "prediccion": int(prob_nueva >= 0.5),
            "grupos": len(selecciones),
        }

    if "aula_resultado_simulador" in st.session_state:
        resultado = st.session_state["aula_resultado_simulador"]
        st.divider()
        st.markdown("### Resultado de la estimación")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Probabilidad estimada", f"{resultado['probabilidad']:.1%}")
        with r2:
            st.metric(
                "Perfil de referencia",
                f"{resultado['probabilidad_base']:.1%}",
                delta=f"{resultado['probabilidad'] - resultado['probabilidad_base']:+.1%}"
            )

        st.progress(float(np.clip(resultado["probabilidad"], 0, 1)))
        if resultado["prediccion"] == 1:
            html(
                """<div class="veredicto riesgo"><h3>Probabilidad elevada de riesgo</h3>
                <p>La estimación alcanza o supera el umbral de clasificación del 50 %.
                Puede ser una señal para explorar opciones de acompañamiento.</p></div>"""
            )
        else:
            html(
                """<div class="veredicto sin-riesgo"><h3>Probabilidad por debajo del umbral</h3>
                <p>La estimación se encuentra por debajo del umbral de clasificación del 50 %.
                Esto no garantiza que no existan dificultades.</p></div>"""
            )
        st.caption(
            "Resultado generado por el modelo para este escenario. No es un diagnóstico "
            "ni una predicción cierta de deserción. Las variaciones no deben interpretarse "
            "como efectos causales de una respuesta aislada."
        )
    else:
        st.info("Completa las respuestas y selecciona «Calcular estimación» para ejecutar el modelo.")

    with st.expander("Alcance y uso responsable"):
        st.write(
            "El simulador conserva las características del perfil de referencia que no "
            "forman parte de las ocho preguntas visibles. Por ello, es una demostración "
            "interactiva del funcionamiento del modelo, no una evaluación individual "
            "completa. Para utilizarlo en un contexto institucional se necesitaría "
            "reconstruir y validar el formulario completo de entrada."
        )


# =========================================================
# PÁGINA: RENDIMIENTO
# =========================================================

elif pagina == "Rendimiento y métricas":

    html('<div class="section-title">Evaluación del clasificador</div>')

    st.write(
        "Resultados calculados directamente sobre el conjunto de prueba "
        "exportado desde el notebook."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, titulo, valor, color in [
        (c1, "Accuracy", accuracy, VIOLET),
        (c2, "Precision", precision, CYAN),
        (c3, "Recall", recall, CORAL),
        (c4, "F1-score", f1, MINT),
        (c5, "ROC-AUC", auc, INK),
    ]:
        with col:
            html(tarjeta_metrica(titulo, f"{valor:.3f}", "", color))

    st.write("")
    extra1, extra2 = st.columns(2)
    with extra1:
        html(tarjeta_metrica("F2-score", f"{f2:.3f}", "Mayor peso al recall", MINT))
    with extra2:
        html(tarjeta_metrica("Especificidad", f"{especificidad:.3f}", "Detección de casos sin riesgo", CYAN))

    izquierda, derecha = st.columns(2, gap="large")

    # ---- Matriz de confusión ----
    with izquierda:
        with st.container(border=True):
            st.markdown("### Matriz de confusión")

            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FFFFFF")
            imagen = ax.imshow(matriz, cmap=MAPA_CONFUSION)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Sin riesgo", "Riesgo"])
            ax.set_yticklabels(["Sin riesgo", "Riesgo"])
            ax.set_xlabel("Predicción del modelo")
            ax.set_ylabel("Etiqueta real")

            umbral_color = matriz.max() / 2
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j, i, f"{matriz[i, j]:,}",
                        ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if matriz[i, j] > umbral_color else INK,
                    )

            estilo_ejes(fig, ax)
            for lado in ("left", "bottom"):
                ax.spines[lado].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Aciertos y errores de clasificación.")

    # ---- Curva ROC ----
    with derecha:
        with st.container(border=True):
            st.markdown("### Curva ROC")

            fpr, tpr, _ = roc_curve(y_test, probabilidades)

            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FFFFFF")
            ax.plot([0, 1], [0, 1], linestyle="--", color=LINE, linewidth=1.5,
                    label="Azar")
            ax.fill_between(fpr, tpr, alpha=0.12, color=VIOLET)
            ax.plot(fpr, tpr, color=VIOLET, linewidth=2.4,
                    label=f"Modelo (AUC = {auc:.3f})")
            ax.set_xlabel("Tasa de falsos positivos")
            ax.set_ylabel("Tasa de verdaderos positivos")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(frameon=False, labelcolor=MUTED, loc="lower right")
            estilo_ejes(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Capacidad para distinguir ambas clases.")

    st.write("")

    izquierda2, derecha2 = st.columns(2, gap="large")

    # ---- Distribución de probabilidades ----
    with izquierda2:
        with st.container(border=True):
            st.markdown("### Probabilidades por clase real")

            y_arr = np.asarray(y_test)
            bins = np.linspace(0, 1, 26)

            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FFFFFF")
            ax.hist(probabilidades[y_arr == 0], bins=bins, alpha=0.78,
                    color=CYAN, edgecolor="#FFFFFF", linewidth=0.5,
                    label="Sin riesgo (real)")
            ax.hist(probabilidades[y_arr == 1], bins=bins, alpha=0.72,
                    color=VIOLET, edgecolor="#FFFFFF", linewidth=0.5,
                    label="Riesgo (real)")
            ax.axvline(0.5, color=INK, linewidth=1, linestyle="--", alpha=0.6)
            ax.set_xlabel("Probabilidad estimada de riesgo")
            ax.set_ylabel("Registros")
            ax.legend(frameon=False, labelcolor=MUTED)
            estilo_ejes(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption("Separación de probabilidades entre clases.")

    # ---- Resumen ----
    with derecha2:
        with st.container(border=True):
            st.markdown("### Resumen de clasificación")

            resumen = pd.DataFrame({
                "Indicador": [
                    "Verdaderos negativos",
                    "Falsos positivos",
                    "Falsos negativos",
                    "Verdaderos positivos",
                ],
                "Cantidad": [int(vn), int(fp), int(fn), int(vp)],
            })

            st.dataframe(resumen, use_container_width=True, hide_index=True)
            st.caption("Conteo de aciertos y equivocaciones.")

    st.write("")

    # ---- Distribución de las predicciones: tarjeta independiente ----
    with st.container(border=True):
        st.markdown("### Distribución de las predicciones")
        st.caption("Cantidad de registros por categoría predicha.")
        conteo_sin_riesgo = int((predicciones == 0).sum())
        conteo_riesgo = int((predicciones == 1).sum())
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        barras = ax.bar(
            ["Sin riesgo", "Riesgo"],
            [conteo_sin_riesgo, conteo_riesgo],
            color=[CYAN, VIOLET],
            width=0.42,
        )
        ax.bar_label(barras, padding=5, color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("Registros")
        estilo_ejes(fig, ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.write("")
    st.markdown("### Interpretación")

    st.write(
        f"""
        El modelo clasificó correctamente **{vp + vn:,} de los
        {len(y_test):,} registros** del conjunto de prueba.

        Identificó **{vp:,} casos de riesgo** y dejó sin detectar
        **{fn:,} casos de riesgo** según las etiquetas disponibles.
        Además, generó **{fp:,} falsas alarmas**.
        """
    )

    st.caption(
        "Las métricas describen el desempeño en este conjunto de prueba; "
        "no garantizan el mismo resultado en otras universidades o cohortes."
    )

# =========================================================
# PÁGINA: INTERPRETABILIDAD SHAP
# =========================================================

elif pagina == "Interpretabilidad SHAP":

    html('<div class="section-title">Interpretabilidad del modelo</div>')

    st.write(
        """
        SHAP permite analizar cómo contribuyen las variables
        a las predicciones del modelo. Aquí se presentan los resultados
        globales calculados sobre una muestra de registros del conjunto
        de prueba.
        """
    )

    st.info(
        "Estos resultados describen el comportamiento del modelo. "
        "No demuestran relaciones causales ni identifican factores "
        "que por sí solos produzcan deserción."
    )

    carpeta_shap = BASE_DIR / "imagenes"
    ruta_importancia = carpeta_shap / "shap_importancia.png"
    ruta_resumen = carpeta_shap / "shap_resumen.png"
    ruta_datos = carpeta_shap / "shap_importancia.csv"

    # Importancia global
    with st.container(border=True):
        st.markdown("### Variables más influyentes")
        st.caption("Magnitud media de las contribuciones SHAP.")
        if ruta_datos.exists():
            importancia_grafica = pd.read_csv(ruta_datos).head(15).copy()
            importancia_grafica = importancia_grafica.sort_values("importancia_shap_media")
            fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            barras = ax.barh(
                importancia_grafica["variable"],
                importancia_grafica["importancia_shap_media"],
                color=[
                    "#B4C5AD" if valor < importancia_grafica["importancia_shap_media"].max() * .65
                    else "#526B55"
                    for valor in importancia_grafica["importancia_shap_media"]
                ],
                height=.68
            )
            ax.bar_label(barras, fmt="%.3f", padding=4, fontsize=8, color=INK)
            ax.set_xlabel("Importancia SHAP media")
            ax.grid(axis="x", alpha=.14)
            estilo_ejes(fig, ax)
            for lado in ("top", "right", "left"):
                ax.spines[lado].set_visible(False)
            ax.tick_params(length=0, labelsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Gráfico regenerado desde el CSV original, con paleta salvia y bosque. "
                "Se conservan los valores de importancia calculados."
            )
        elif ruta_importancia.exists():
            st.image(str(ruta_importancia), use_container_width=True)
        else:
            st.warning(
                "No se encontró imagenes/shap_importancia.csv ni shap_importancia.png. "
                "Ejecuta primero: python interpretabilidad.py"
            )

    st.write("")

    # Distribución de contribuciones
    with st.container(border=True):
        st.markdown("### Distribución de las contribuciones")
        st.caption(
            "Cada punto representa una contribución SHAP de una variable "
            "en uno de los registros explicados. Se adapta únicamente la paleta "
            "visual a los tonos salvia y bosque de AULA."
        )
        if ruta_resumen.exists():
            # Recolor the existing SHAP beeswarm without changing point positions,
            # SHAP values, feature order, or scale.
            imagen_shap = Image.open(ruta_resumen).convert("RGBA")
            pixeles = np.array(imagen_shap).copy()
            rgb = pixeles[..., :3].astype(np.float32) / 255.0
            hsv = rgb_to_hsv(rgb)
            tono = hsv[..., 0]
            saturacion = hsv[..., 1]

            # Toda la escala cromática original se convierte a verdes:
            # azul/cian (valores bajos) -> salvia claro;
            # violeta/magenta/rojo (valores altos) -> verde bosque.
            colores = saturacion > 0.20
            bajos = colores & (tono >= 0.45) & (tono < 0.70)
            altos = colores & (
                ((tono >= 0.70) & (tono <= 1.0)) |
                (tono <= 0.08)
            )

            pixeles[bajos, :3] = np.array([167, 185, 159], dtype=np.uint8)
            pixeles[altos, :3] = np.array([82, 107, 85], dtype=np.uint8)

            # Neutralize the pale lavender plot background to AULA's warm white.
            fondo = (saturacion < 0.16) & (pixeles[..., :3].mean(axis=2) > 215)
            pixeles[fondo, :3] = np.array([247, 247, 241], dtype=np.uint8)

            imagen_aula = Image.fromarray(pixeles, mode="RGBA")
            st.image(imagen_aula, use_container_width=True)
        else:
            st.warning(
                "No se encontró imagenes/shap_resumen.png. "
                "Ejecuta primero: python interpretabilidad.py"
            )

    st.write("")

    # Tabla del ranking
    with st.container(border=True):
        st.markdown("### Ranking de variables")
        st.caption("Las 15 variables con mayor importancia SHAP media.")
        if ruta_datos.exists():
            importancia_shap = pd.read_csv(ruta_datos)
            st.dataframe(
                importancia_shap.head(15),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "No se encontró imagenes/shap_importancia.csv. "
                "Ejecuta primero: python interpretabilidad.py"
            )

    st.caption(
        "Análisis global realizado con 313 variables, "
        "30 registros de referencia y 20 registros explicados. "
        "Las explicaciones corresponden a la muestra analizada."
    )

# =========================================================
# PÁGINA: ACERCA DEL PROYECTO
# =========================================================

elif pagina == "Acerca del proyecto":

    html('<div class="section-title">Sobre la investigación</div>')
    st.caption(
        "Contexto, datos, metodología, resultados e implicaciones del prototipo académico."
    )

    with st.container(border=True):
        st.markdown("### Detección temprana del riesgo de deserción universitaria")
        st.write(
            "La deserción universitaria es un fenómeno que puede afectar "
            "la continuidad académica y los proyectos personales de los estudiantes. "
            "Este proyecto explora el uso de aprendizaje automático para identificar "
            "patrones asociados con niveles de riesgo y aportar información que "
            "pueda orientar acciones preventivas de acompañamiento."
        )
        st.write(
            "AULA es un prototipo académico de apoyo analítico. Su resultado "
            "corresponde a una clasificación estadística construida a partir de "
            "la variable objetivo del conjunto de datos; no determina el futuro "
            "de una persona ni reemplaza la valoración de profesionales o equipos "
            "de bienestar universitario."
        )

    st.write("")
    st.markdown("### Datos utilizados")

    d1, d2, d3, d4 = st.columns(4)
    datos_resumen = [
        (d1, "Registros", "5.000", "Base consolidada"),
        (d2, "Registros reales", "4.000", "Perfiles de origen"),
        (d3, "Registros sintéticos", "1.000", "Datos derivados"),
        (d4, "Variables predictoras", str(info["numero_variables"]), "Tras la selección"),
    ]
    for col, nombre, valor, nota in datos_resumen:
        with col:
            html(tarjeta_metrica(nombre, valor, nota, VIOLET))

    st.write("")
    with st.container(border=True):
        st.markdown("### Composición de la base consolidada")
        st.progress(0.8, text="80 % registros reales · 20 % registros sintéticos")
        st.caption(
            "Los 4.000 registros reales representan 834 perfiles únicos; "
            "se reportaron repeticiones de perfiles en la base."
        )

    st.write("")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("### Origen y estructura")
            st.write(
                "El conjunto de datos utilizado se denomina "
                "**Risk of Desertion in Undergraduate UCC**, disponible en "
                "Mendeley Data y asociado a la Universidad Cooperativa de Colombia. "
                "La base consolidada del proyecto reúne 4.000 registros reales "
                "y 1.000 registros sintéticos."
            )
            st.write(
                "Los registros reales corresponden a 834 perfiles únicos, "
                "con repeticiones de perfiles en la base. Los registros sintéticos "
                "se generaron combinando información de perfiles reales de la "
                "misma categoría de riesgo. Por ello, su presencia debe tenerse "
                "en cuenta al interpretar la evaluación y la capacidad de "
                "generalización del modelo."
            )

    with c2:
        with st.container(border=True):
            st.markdown("### Variable objetivo")
            st.write(
                "La variable original `nivel_motivacion` contiene cinco niveles "
                "de riesgo de deserción. Para el modelado se transformó en una "
                "clasificación binaria:"
            )
            st.markdown(
                "- **Sin riesgo (0):** riesgo muy bajo y riesgo leve.\n"
                "- **Con riesgo (1):** riesgo medio, riesgo alto y deserción casi segura."
            )
            st.write(
                "Esta transformación permite evaluar el problema como una tarea "
                "de clasificación binaria. La etiqueta representa el nivel de "
                "riesgo asignado en la base, no la confirmación de una deserción real."
            )

    st.write("")
    st.markdown("### Metodología de aprendizaje automático")

    with st.container(border=True):
        st.write(
            "El trabajo parte de la preparación y auditoría de los datos, "
            "incluyendo la revisión de duplicados y la separación entre "
            "entrenamiento y prueba. Para reducir el riesgo de que perfiles "
            "repetidos aparezcan en ambos conjuntos, se empleó una partición "
            "agrupada y estratificada."
        )
        st.write(
            "Se compararon una línea base y distintos clasificadores: "
            "Regresión Logística, SVM con kernel RBF, Random Forest y "
            "HistGradientBoosting, además de una variante de Random Forest "
            "con selección de variables. La selección del modelo se realizó "
            "mediante validación cruzada anidada, utilizando F2 como criterio "
            "principal para dar mayor peso a la sensibilidad en la detección "
            "de casos clasificados con riesgo."
        )
        st.write(
            "El modelo seleccionado fue **Boosting (HistGB)**, implementado "
            "con HistGradientBoosting. La configuración final documentada "
            "incluye `learning_rate=0.1`, `max_depth=6`, `max_iter=200`, "
            "ponderación de clases balanceada y semilla 42."
        )

    st.write("")
    st.markdown("### Resultados en el conjunto de prueba")

    r1, r2, r3, r4, r5 = st.columns(5)
    resultados = [
        (r1, "Exactitud", f"{accuracy:.1%}"),
        (r2, "Precisión", f"{precision:.1%}"),
        (r3, "Sensibilidad", f"{recall:.1%}"),
        (r4, "F1", f"{f1:.3f}"),
        (r5, "ROC-AUC", f"{auc:.3f}"),
    ]
    for col, nombre, valor in resultados:
        with col:
            html(tarjeta_metrica(nombre, valor, "Conjunto de prueba", CYAN))

    extra_r1, extra_r2 = st.columns(2)
    with extra_r1:
        html(tarjeta_metrica("F2-score", f"{f2:.3f}", "Énfasis en sensibilidad", MINT))
    with extra_r2:
        html(tarjeta_metrica("Especificidad", f"{especificidad:.3f}", "Clase sin riesgo", CYAN))

    with st.container(border=True):
        st.markdown("### Lectura de los resultados")
        st.write(
            f"El conjunto de prueba contiene **{len(y_test):,} registros**. "
            f"La matriz de confusión muestra **{int(vp):,} verdaderos positivos**, "
            f"**{int(vn):,} verdaderos negativos**, **{int(fp):,} falsos positivos** "
            f"y **{int(fn):,} falsos negativos**. En términos prácticos, el modelo "
            f"identificó {int(vp):,} registros etiquetados con riesgo y no detectó "
            f"{int(fn):,} de ellos en este conjunto de evaluación."
        )
        st.write(
            f"El valor ROC-AUC de **{auc:.3f}** resume la capacidad de ordenar "
            "registros con riesgo por encima de registros sin riesgo a través "
            "de distintos umbrales. Las métricas describen únicamente este "
            "conjunto de prueba y no garantizan el mismo rendimiento en otras "
            "cohortes o instituciones."
        )

    st.write("")
    st.markdown("### Consideraciones y limitaciones")

    with st.container(border=True):
        st.markdown(
            "- **Datos sintéticos:** complementan la base, pero no equivalen "
            "a observaciones independientes de nuevos estudiantes.\n"
            "- **Perfiles repetidos:** la cantidad de registros no representa "
            "la misma cantidad de estudiantes únicos.\n"
            "- **Linaje sintético:** no se conservaron identificadores de los perfiles "
            "padres de cada registro sintético; por ello, la partición agrupada no "
            "permite asegurar que todos los descendientes sintéticos estén separados "
            "de sus perfiles de origen.\\n"
            "- **Una partición de prueba:** la evaluación final se basa en un único "
            "hold-out agrupado y requiere validación adicional.\\n"
            "- **Sin dimensión temporal:** la base estática no permite determinar "
            "con cuánta anticipación se podría detectar un caso.\\n"
            "- **Alcance de la etiqueta:** el modelo predice la categoría de "
            "riesgo definida en los datos, no una deserción observada.\n"
            "- **Generalización:** los datos proceden de un contexto institucional "
            "específico y requieren validación externa.\n"
            "- **Uso responsable:** una predicción no debe ser la única base "
            "para decisiones académicas sobre una persona."
        )
        st.info(
            "AULA es una demostración académica. Antes de un uso institucional "
            "se necesitarían más datos reales, validación temporal y externa, "
            "calibración del umbral y revisión humana de cada caso."
        )

# =========================================================
# PIE DE PÁGINA
# =========================================================

html(
    """
    <div class="footer">
        AULA · Sistema académico de alerta temprana<br>
        Proyecto de Machine Learning II · Universidad Externado de Colombia
    </div>
    """
)