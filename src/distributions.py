from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fitter import Fitter
from scipy import stats


# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "distributions"
FIGURES_DIR = RESULTS_DIR / "figures"

INTERARRIVALS_PATH = PROCESSED_DIR / "interarrivals.csv"
CONSULTAS_PATH = PROCESSED_DIR / "consultas.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# CONFIGURACIÓN
# =========================================================

# Distribuciones candidatas para variables continuas, positivas
# y potencialmente asimétricas.
CANDIDATE_DISTRIBUTIONS = [
    "expon",
    "gamma",
    "weibull_min",
    "lognorm",
]

# Métrica principal para ordenar los ajustes.
# Fitter también calcula BIC, SSE, KS, etc.
FIT_METHOD = "aic"

# Cantidad de bins para histogramas / Fitter.
BINS = 80


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def convertir_a_python(obj):
    """
    Convierte tipos numpy a tipos nativos de Python
    para poder guardarlos en JSON.
    """
    if isinstance(obj, dict):
        return {k: convertir_a_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [convertir_a_python(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def guardar_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            convertir_a_python(obj),
            f,
            ensure_ascii=False,
            indent=4,
        )


def estadisticas_descriptivas(serie, nombre):
    """
    Devuelve estadísticas descriptivas útiles para el TP.
    """
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) == 0:
        raise ValueError(f"No hay datos válidos para {nombre}.")

    resumen = {
        "variable": nombre,
        "n": int(len(x)),
        "media": float(x.mean()),
        "desvio": float(x.std()),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "mediana": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
        "p99": float(x.quantile(0.99)),
        "max": float(x.max()),
        "asimetria": float(x.skew()),
    }

    return resumen


def imprimir_resumen(resumen):
    print("\n" + "=" * 60)
    print(resumen["variable"].upper())
    print("=" * 60)

    for clave, valor in resumen.items():
        if clave == "variable":
            continue

        if isinstance(valor, float):
            print(f"{clave:12s}: {valor:.4f}")
        else:
            print(f"{clave:12s}: {valor}")


# =========================================================
# AJUSTE DE DISTRIBUCIONES
# =========================================================

def ajustar_distribuciones(
    datos,
    nombre_variable,
    nombre_archivo,
    distributions=None,
    bins=BINS,
):
    """
    Ajusta varias distribuciones con Fitter.

    Guarda:
    - tabla completa de métricas
    - top de distribuciones
    - parámetros de la mejor distribución
    - gráfico histograma + PDFs ajustadas

    La selección principal se ordena por AIC, pero se guardan
    también SSE, BIC y métricas KS para poder justificar la elección.
    """

    if distributions is None:
        distributions = CANDIDATE_DISTRIBUTIONS

    x = pd.to_numeric(
        pd.Series(datos),
        errors="coerce",
    ).dropna()

    # Las distribuciones candidatas son para variables positivas.
    x = x[x > 0]

    if len(x) < 100:
        raise ValueError(
            f"{nombre_variable}: hay muy pocos datos válidos "
            f"para hacer un ajuste confiable ({len(x)})."
        )

    print("\n" + "=" * 60)
    print(f"AJUSTANDO: {nombre_variable}")
    print("=" * 60)
    print(f"Observaciones utilizadas: {len(x):,}")
    print("Distribuciones:", distributions)

    fitter = Fitter(
        x.to_numpy(),
        bins=bins,
        distributions=distributions,
        timeout=60,
        density=True,
        verbose=False,
    )

    # Un solo worker + threads evita problemas de multiprocessing
    # al ejecutar el script en Windows.
    fitter.fit(
        progress=True,
        max_workers=1,
        prefer="threads",
    )

    # Tabla completa de resultados
    resultados = fitter.df_errors.copy()

    if FIT_METHOD not in resultados.columns:
        raise ValueError(
            f"Fitter no devolvió la métrica '{FIT_METHOD}'. "
            f"Columnas disponibles: {resultados.columns.tolist()}"
        )

    resultados = resultados.sort_values(
        FIT_METHOD,
        ascending=True,
    )

    resultados.to_csv(
        RESULTS_DIR / f"{nombre_archivo}_fit_results.csv"
    )

    top = resultados.head(len(distributions))

    print("\nRanking por AIC:")
    columnas_mostrar = [
        col
        for col in [
            "sumsquare_error",
            "aic",
            "bic",
            "ks_statistic",
            "ks_pvalue",
        ]
        if col in top.columns
    ]

    print(top[columnas_mostrar].to_string())

    mejor = fitter.get_best(method=FIT_METHOD)

    for dist_name, parametros in mejor.items():
        if "loc" in parametros and parametros["loc"] <= 0:
            parametros["loc"] = 0.00001

    print("\nMejor ajuste según AIC:")
    print(mejor)

    guardar_json(
        mejor,
        RESULTS_DIR / f"{nombre_archivo}_best_fit.json",
    )

    # Gráfico del ajuste
    plt.figure(figsize=(10, 6))
    fitter.hist()
    fitter.plot_pdf(
        Nbest=min(4, len(distributions)),
        method=FIT_METHOD,
    )
    plt.title(f"Ajuste de distribuciones - {nombre_variable}")
    plt.xlabel(nombre_variable)
    plt.ylabel("Densidad")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"{nombre_archivo}_fit.png",
        dpi=150,
    )
    plt.close()

    return fitter, resultados, mejor


# =========================================================
# ANÁLISIS DE LLEGADAS
# =========================================================

def analizar_interarribos(interarrivals):
    print("\n\n############################################################")
    print("1. ANÁLISIS DE TIEMPOS ENTRE LLEGADAS")
    print("############################################################")

    df = interarrivals.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["interarrival_seconds"] = pd.to_numeric(
        df["interarrival_seconds"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp", "interarrival_seconds"]
    )

    # El preprocesamiento puede conservar ceros si existiesen.
    # Para el ajuste continuo usamos estrictamente valores > 0.
    datos = df.loc[
        df["interarrival_seconds"] > 0,
        "interarrival_seconds",
    ]

    resumen = estadisticas_descriptivas(
        datos,
        "Tiempo entre llegadas (segundos)",
    )

    imprimir_resumen(resumen)

    guardar_json(
        resumen,
        RESULTS_DIR / "interarrival_descriptive_stats.json",
    )

    # -----------------------------------------------------
    # Revisión de gaps largos
    # -----------------------------------------------------

    limites = [300, 900, 1800, 3600]

    print("\nGaps largos:")
    gaps = []

    for limite in limites:
        cantidad = int((datos > limite).sum())
        porcentaje = float(cantidad / len(datos) * 100)

        gaps.append({
            "mayor_a_segundos": limite,
            "cantidad": cantidad,
            "porcentaje": porcentaje,
        })

        print(
            f"> {limite:4d} s: "
            f"{cantidad:5d} ({porcentaje:.2f}%)"
        )

    pd.DataFrame(gaps).to_csv(
        RESULTS_DIR / "interarrival_long_gaps.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Actividad según hora del día
    # -----------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour

    por_hora_dia = (
        df.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("cantidad")
        .reset_index()
    )

    por_hora_dia["porcentaje"] = (
        por_hora_dia["cantidad"]
        / por_hora_dia["cantidad"].sum()
        * 100
    )

    por_hora_dia.to_csv(
        RESULTS_DIR / "arrivals_by_hour_of_day.csv",
        index=False,
    )

    print("\nLlegadas según hora del día:")
    print(por_hora_dia.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(
        por_hora_dia["hour"],
        por_hora_dia["cantidad"],
    )
    plt.xlabel("Hora del día (UTC)")
    plt.ylabel("Cantidad de consultas")
    plt.title("Distribución de consultas por hora del día")
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "arrivals_by_hour_of_day.png",
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # Cantidad de llegadas por hora calendario
    # -----------------------------------------------------

    df["calendar_hour"] = df["timestamp"].dt.floor("h")

    por_hora_calendario = (
        df.groupby("calendar_hour")
        .size()
        .rename("cantidad")
        .reset_index()
    )

    por_hora_calendario.to_csv(
        RESULTS_DIR / "arrivals_by_calendar_hour.csv",
        index=False,
    )

    print("\nCantidad de consultas por hora calendario:")
    print(
        por_hora_calendario["cantidad"]
        .describe()
        .to_string()
    )

    # -----------------------------------------------------
    # Histograma simple
    # -----------------------------------------------------

    plt.figure(figsize=(10, 5))
    plt.hist(datos, bins=80, density=True)
    plt.xlabel("Segundos entre llegadas")
    plt.ylabel("Densidad")
    plt.title("Histograma de tiempos entre llegadas")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "interarrival_histogram.png",
        dpi=150,
    )
    plt.close()

    # -----------------------------------------------------
    # Ajuste completo
    # -----------------------------------------------------

    fitter, resultados, mejor = ajustar_distribuciones(
        datos,
        nombre_variable="Tiempo entre llegadas (segundos)",
        nombre_archivo="interarrival",
    )

    # -----------------------------------------------------
    # Sensibilidad sin el 1% superior
    # -----------------------------------------------------
    #
    # NO se usa automáticamente para la simulación.
    # Sirve para comparar cuánto influyen los gaps extremos.
    # -----------------------------------------------------

    p99 = datos.quantile(0.99)

    datos_p99 = datos[
        datos <= p99
    ]

    print(
        "\nAjuste de sensibilidad excluyendo solamente "
        f"el 1% superior (> {p99:.2f} s)."
    )

    _, resultados_p99, mejor_p99 = ajustar_distribuciones(
        datos_p99,
        nombre_variable=(
            "Tiempo entre llegadas - sensibilidad hasta P99"
        ),
        nombre_archivo="interarrival_p99_sensitivity",
    )

    return {
        "resumen": resumen,
        "fit": mejor,
        "fit_p99_sensibilidad": mejor_p99,
        "por_hora_dia": por_hora_dia,
        "resultados_fit": resultados,
        "resultados_fit_p99": resultados_p99,
    }


# =========================================================
# HARDNESS
# =========================================================

def analizar_hardness(consultas):
    print("\n\n############################################################")
    print("2. DISTRIBUCIÓN EMPÍRICA DE HARDNESS")
    print("############################################################")

    hardness = pd.to_numeric(
        consultas["hardness"],
        errors="coerce",
    ).dropna()

    resumen = estadisticas_descriptivas(
        hardness,
        "Hardness",
    )

    imprimir_resumen(resumen)

    frecuencias = (
        hardness.value_counts()
        .sort_index()
        .rename_axis("hardness")
        .reset_index(name="frecuencia")
    )

    frecuencias["probabilidad"] = (
        frecuencias["frecuencia"]
        / frecuencias["frecuencia"].sum()
    )

    frecuencias["porcentaje"] = (
        frecuencias["probabilidad"] * 100
    )

    frecuencias.to_csv(
        RESULTS_DIR / "hardness_empirical_distribution.csv",
        index=False,
    )

    guardar_json(
        resumen,
        RESULTS_DIR / "hardness_descriptive_stats.json",
    )

    print("\nDistribución empírica:")
    print(frecuencias.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.bar(
        frecuencias["hardness"].astype(str),
        frecuencias["porcentaje"],
    )
    plt.xlabel("Hardness")
    plt.ylabel("Porcentaje de consultas")
    plt.title("Distribución empírica de hardness")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "hardness_distribution.png",
        dpi=150,
    )
    plt.close()

    return frecuencias


# =========================================================
# OUTPUT TOKENS
# =========================================================

def analizar_output_tokens(consultas):
    print("\n\n############################################################")
    print("3. ANÁLISIS DE OUTPUT TOKENS")
    print("############################################################")

    output_tokens = pd.to_numeric(
        consultas["output_tokens"],
        errors="coerce",
    ).dropna()

    output_tokens = output_tokens[
        output_tokens > 0
    ]

    resumen = estadisticas_descriptivas(
        output_tokens,
        "Output tokens",
    )

    imprimir_resumen(resumen)

    guardar_json(
        resumen,
        RESULTS_DIR / "output_tokens_descriptive_stats.json",
    )

    plt.figure(figsize=(10, 5))
    plt.hist(
        output_tokens,
        bins=80,
        density=True,
    )
    plt.xlabel("Output tokens")
    plt.ylabel("Densidad")
    plt.title("Histograma de longitud de las respuestas")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "output_tokens_histogram.png",
        dpi=150,
    )
    plt.close()

    fitter, resultados, mejor = ajustar_distribuciones(
        output_tokens,
        nombre_variable="Output tokens",
        nombre_archivo="output_tokens",
    )

    return {
        "resumen": resumen,
        "fit": mejor,
        "resultados_fit": resultados,
    }


# =========================================================
# RELACIÓN HARDNESS / TOKENS
# =========================================================

def analizar_relaciones(consultas):
    print("\n\n############################################################")
    print("4. RELACIÓN ENTRE HARDNESS Y TOKENS")
    print("############################################################")

    df = consultas[
        [
            "hardness",
            "input_tokens",
            "output_tokens",
        ]
    ].copy()

    for columna in df.columns:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    df = df.dropna()

    pearson_output = df["hardness"].corr(
        df["output_tokens"],
        method="pearson",
    )

    spearman_output = df["hardness"].corr(
        df["output_tokens"],
        method="spearman",
    )

    pearson_input = df["hardness"].corr(
        df["input_tokens"],
        method="pearson",
    )

    spearman_input = df["hardness"].corr(
        df["input_tokens"],
        method="spearman",
    )

    correlaciones = {
        "hardness_vs_output_tokens": {
            "pearson": float(pearson_output),
            "spearman": float(spearman_output),
        },
        "hardness_vs_input_tokens": {
            "pearson": float(pearson_input),
            "spearman": float(spearman_input),
        },
    }

    guardar_json(
        correlaciones,
        RESULTS_DIR / "correlations.json",
    )

    print("\nHardness vs output tokens")
    print(f"Pearson : {pearson_output:.4f}")
    print(f"Spearman: {spearman_output:.4f}")

    print("\nHardness vs input tokens")
    print(f"Pearson : {pearson_input:.4f}")
    print(f"Spearman: {spearman_input:.4f}")

    # Resumen por valor de hardness
    por_hardness = (
        df.groupby("hardness")
        .agg(
            cantidad=("output_tokens", "size"),
            output_mean=("output_tokens", "mean"),
            output_median=("output_tokens", "median"),
            output_std=("output_tokens", "std"),
            input_mean=("input_tokens", "mean"),
            input_median=("input_tokens", "median"),
        )
        .reset_index()
    )

    por_hardness.to_csv(
        RESULTS_DIR / "tokens_by_hardness.csv",
        index=False,
    )

    print("\nTokens por nivel de hardness:")
    print(por_hardness.to_string(index=False))

    # Scatter
    plt.figure(figsize=(9, 5))
    plt.scatter(
        df["hardness"],
        df["output_tokens"],
        alpha=0.15,
        s=10,
    )
    plt.xlabel("Hardness")
    plt.ylabel("Output tokens")
    plt.title("Hardness vs output tokens")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "hardness_vs_output_tokens.png",
        dpi=150,
    )
    plt.close()

    return correlaciones, por_hardness


# =========================================================
# FUNCIONES PARA LA SIMULACIÓN FUTURA
# =========================================================

def generar_desde_ajuste(best_fit, size=1, random_state=None):
    """
    Genera valores aleatorios a partir de un ajuste guardado
    por Fitter.

    Ejemplo de best_fit:
        {
            "lognorm": {
                "s": ...,
                "loc": ...,
                "scale": ...
            }
        }
    """

    if len(best_fit) != 1:
        raise ValueError(
            "best_fit debe contener exactamente una distribución."
        )

    nombre, parametros = next(iter(best_fit.items()))


    # Modificando best fit para que el loc no sea negativo
    if "loc" in parametros and parametros["loc"] <= 0: 
        parametros["loc"] = 0.00001

    if not hasattr(stats, nombre):
        raise ValueError(
            f"SciPy no reconoce la distribución '{nombre}'."
        )

    distribucion = getattr(stats, nombre)

    valores = distribucion.rvs(
        size=size,
        random_state=random_state,
        **parametros,
    )

    return np.asarray(valores)


def calcular_tiempo_servicio(
    output_tokens,
    ttft_segundos,
    tokens_por_segundo,
):
    """
    Transforma una longitud de respuesta en tiempo de atención.

    T_servicio = TTFT + output_tokens / velocidad

    Esta función NO fija todavía los parámetros de los tres modelos.
    Esos parámetros deberían quedar en models.py cuando se definan
    y documenten las fuentes definitivas.
    """

    if tokens_por_segundo <= 0:
        raise ValueError(
            "tokens_por_segundo debe ser mayor que cero."
        )

    return (
        float(ttft_segundos)
        + float(output_tokens) / float(tokens_por_segundo)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("Cargando datasets procesados...")

    if not INTERARRIVALS_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {INTERARRIVALS_PATH}\n"
            "Ejecutá preprocessing.py primero."
        )

    if not CONSULTAS_PATH.exists():
        raise FileNotFoundError(
            f"No existe: {CONSULTAS_PATH}\n"
            "Ejecutá preprocessing.py primero."
        )

    interarrivals = pd.read_csv(
        INTERARRIVALS_PATH
    )

    consultas = pd.read_csv(
        CONSULTAS_PATH
    )

    print(
        f"Interarribos cargados: {len(interarrivals):,}"
    )

    print(
        f"Consultas cargadas: {len(consultas):,}"
    )

    # Validación mínima
    requeridas_interarrivals = {
        "timestamp",
        "interarrival_seconds",
    }

    requeridas_consultas = {
        "hardness",
        "input_tokens",
        "output_tokens",
    }

    faltan_interarrivals = (
        requeridas_interarrivals
        - set(interarrivals.columns)
    )

    faltan_consultas = (
        requeridas_consultas
        - set(consultas.columns)
    )

    if faltan_interarrivals:
        raise ValueError(
            "Faltan columnas en interarrivals.csv: "
            + ", ".join(sorted(faltan_interarrivals))
        )

    if faltan_consultas:
        raise ValueError(
            "Faltan columnas en consultas.csv: "
            + ", ".join(sorted(faltan_consultas))
        )

    resultados_interarribos = analizar_interarribos(
        interarrivals
    )

    distribucion_hardness = analizar_hardness(
        consultas
    )

    resultados_output = analizar_output_tokens(
        consultas
    )

    correlaciones, tokens_por_hardness = (
        analizar_relaciones(consultas)
    )

    # Resumen principal de parámetros seleccionados
    resumen_modelo = {
        "interarrival_best_fit": (
            resultados_interarribos["fit"]
        ),
        "output_tokens_best_fit": (
            resultados_output["fit"]
        ),
        "hardness_empirical_distribution": (
            distribucion_hardness[
                [
                    "hardness",
                    "probabilidad",
                ]
            ].to_dict(orient="records")
        ),
        "correlations": correlaciones,
    }

    guardar_json(
        resumen_modelo,
        RESULTS_DIR / "model_distribution_summary.json",
    )

    print("\n" + "=" * 60)
    print("ANÁLISIS TERMINADO")
    print("=" * 60)

    print("\nResultados guardados en:")
    print(RESULTS_DIR)

    print("\nArchivos principales:")
    print("- interarrival_fit_results.csv")
    print("- interarrival_best_fit.json")
    print("- interarrival_p99_sensitivity_fit_results.csv")
    print("- output_tokens_fit_results.csv")
    print("- output_tokens_best_fit.json")
    print("- hardness_empirical_distribution.csv")
    print("- correlations.json")
    print("- model_distribution_summary.json")

    print("\nGráficos:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
