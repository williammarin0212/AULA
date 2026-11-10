# ==========================================================
# INTERPRETABILIDAD DEL MODELO DE DESERCIÓN - AULA
# SHAP: importancia global y explicación de predicciones
# ==========================================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1. RUTAS Y CONFIGURACIÓN
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelo"
OUTPUT_DIR = BASE_DIR / "imagenes"

MODEL_PATH = MODEL_DIR / "modelo_final.joblib"
X_PATH = MODEL_DIR / "X_test.csv"
Y_PATH = MODEL_DIR / "y_test.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BACKGROUND = 30
N_EXPLICAR = 20
MAX_EVALS = None  # Se calcula según el número de variables

# Paleta visual AULA
INK = "#12163A"
VIOLET = "#5B4BDB"
CYAN = "#19B5C9"
CORAL = "#F0506E"
PAPER = "#F5F6FB"
MUTED = "#555D7D"
LINE = "#E3E6F3"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": LINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

# ----------------------------------------------------------
# 2. CARGA Y VALIDACIÓN
# ----------------------------------------------------------

def cargar_recursos():
    archivos = [MODEL_PATH, X_PATH, Y_PATH]
    faltantes = [p.name for p in archivos if not p.exists()]

    if faltantes:
        raise FileNotFoundError(
            "No se encontraron estos archivos en la carpeta modelo/: "
            + ", ".join(faltantes)
        )

    modelo = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_PATH)
    y_test = pd.read_csv(Y_PATH).iloc[:, 0]

    if len(X_test) != len(y_test):
        raise ValueError("X_test e y_test tienen diferente número de filas.")

    if hasattr(modelo, "feature_names_in_"):
        esperadas = list(modelo.feature_names_in_)
        if list(X_test.columns) != esperadas:
            raise ValueError(
                "Las columnas de X_test no coinciden con las del modelo."
            )

    return modelo, X_test, y_test


# ----------------------------------------------------------
# 3. FUNCIÓN DE PREDICCIÓN
# ----------------------------------------------------------

def probabilidad_riesgo(datos):
    """Devuelve la probabilidad de la clase positiva (riesgo)."""
    probabilidades = modelo.predict_proba(datos)

    if probabilidades.ndim != 2 or probabilidades.shape[1] < 2:
        raise ValueError("El modelo no devolvió dos probabilidades de clase.")

    return probabilidades[:, 1]


# ----------------------------------------------------------
# 4. EXPLICACIÓN SHAP
# ----------------------------------------------------------

def calcular_shap():
    print("\n" + "=" * 60)
    print("AULA | INTERPRETABILIDAD SHAP")
    print("=" * 60)

    n = len(X_test)
    if n < 2:
        raise ValueError("Se necesitan al menos dos registros de prueba.")

    rng = np.random.default_rng(RANDOM_STATE)

    n_background = min(N_BACKGROUND, n)
    n_explicar = min(N_EXPLICAR, n)

    indices = rng.permutation(n)
    background_idx = indices[:n_background]
    explain_idx = indices[n_background:n_background + n_explicar]

    if len(explain_idx) == 0:
        explain_idx = indices[:n_explicar]

    background = X_test.iloc[background_idx].copy()
    X_explicar = X_test.iloc[explain_idx].copy()

    print(f"Variables: {X_test.shape[1]}")
    print(f"Registros de referencia: {len(background)}")
    print(f"Registros explicados: {len(X_explicar)}")
    print("Calculando SHAP. Puede tardar unos minutos...")

    masker = shap.maskers.Independent(background)

    # PermutationExplainer funciona con modelos que exponen predict_proba,
    # aunque el estimador no tenga un TreeExplainer compatible.
    explainer = shap.Explainer(
        probabilidad_riesgo,
        masker,
        algorithm="permutation",
        feature_names=list(X_test.columns)
    )

    max_evals = MAX_EVALS or (2 * X_test.shape[1] + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer(
            X_explicar,
            max_evals=max_evals,
            batch_size=50
        )

    values = np.asarray(shap_values.values)

    # Asegurar matriz (registros, variables)
    if values.ndim == 3:
        values = values[:, :, -1]

    if values.ndim != 2:
        raise ValueError(
            f"Dimensión SHAP inesperada: {values.shape}"
        )

    if values.shape[1] != X_explicar.shape[1]:
        raise ValueError(
            "El número de valores SHAP no coincide con las variables."
        )

    return shap_values, values, X_explicar


# ----------------------------------------------------------
# 5. GRÁFICAS Y TABLA DE IMPORTANCIA
# ----------------------------------------------------------

def guardar_resultados(shap_values, values, X_explicar):
    importancia = np.abs(values).mean(axis=0)

    tabla = pd.DataFrame({
        "variable": X_explicar.columns,
        "importancia_shap_media": importancia
    }).sort_values(
        "importancia_shap_media",
        ascending=False
    )

    tabla.to_csv(
        OUTPUT_DIR / "shap_importancia.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top = tabla.head(15).sort_values(
        "importancia_shap_media",
        ascending=True
    )

    # Gráfica de importancia global
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top["variable"],
        top["importancia_shap_media"],
        color=VIOLET,
        edgecolor="none"
    )
    ax.set_title("SHAP | Variables con mayor influencia")
    ax.set_xlabel("Impacto medio absoluto en la predicción")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "shap_importancia.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=PAPER
    )
    plt.close(fig)

    # Resumen SHAP tipo beeswarm
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        values,
        X_explicar,
        feature_names=list(X_explicar.columns),
        max_display=15,
        show=False,
        plot_size=(11, 8),
        color=VIOLET
    )
    fig = plt.gcf()
    fig.set_facecolor(PAPER)
    ax = plt.gca()
    ax.set_facecolor("#FFFFFF")
    ax.set_title(
        "SHAP | Distribución de impactos por variable",
        color=INK,
        pad=14,
        fontweight="bold"
    )
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "shap_resumen.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=PAPER
    )
    plt.close(fig)

    # Valores SHAP por registro y variable
    shap_df = pd.DataFrame(
        values,
        columns=X_explicar.columns,
        index=X_explicar.index
    )
    shap_df.to_csv(
        OUTPUT_DIR / "shap_valores_registros.csv",
        encoding="utf-8-sig"
    )

    resumen = {
        "metodo": "SHAP PermutationExplainer",
        "clase_explicada": "Riesgo de deserción (clase 1)",
        "registros_referencia": int(min(N_BACKGROUND, len(X_test))),
        "registros_explicados": int(len(X_explicar)),
        "numero_variables": int(X_explicar.shape[1]),
        "max_evals": int(MAX_EVALS or (2 * X_test.shape[1] + 1)),
        "top_15_variables": top.sort_values(
            "importancia_shap_media",
            ascending=False
        )["variable"].tolist()
    }

    with open(
        OUTPUT_DIR / "shap_informacion.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    print("\nSHAP finalizado correctamente.")
    print("\nLas 15 variables más influyentes:")
    print(
        tabla.head(15).to_string(
            index=False,
            formatters={
                "importancia_shap_media": "{:.6f}".format
            }
        )
    )
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


# ----------------------------------------------------------
# 6. EJECUCIÓN
# ----------------------------------------------------------

if __name__ == "__main__":
    try:
        modelo, X_test, y_test = cargar_recursos()
        shap_values, values, X_explicar = calcular_shap()
        guardar_resultados(shap_values, values, X_explicar)

    except Exception as error:
        print("\nERROR DURANTE EL ANÁLISIS SHAP")
        print(f"{type(error).__name__}: {error}")
        raise
