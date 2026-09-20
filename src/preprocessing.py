from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken


# =========================================================
# RUTAS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset 1: Chatbot Arena descargado localmente en formato Parquet
ARENA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "chatbot_arena"
    / "train-00000-of-00001-cced8514c7ed782a.parquet"
)

# Dataset 2: hardness descargado localmente y SIN comprimir
HARDNESS_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "hardness"
    / "chatbot-arena-gpt3-scores.jsonl"
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# TOKENIZER
# =========================================================

encoding = tiktoken.get_encoding("o200k_base")


def contar_tokens(texto):
    """Devuelve la cantidad de tokens de un texto."""
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return np.nan

    if not isinstance(texto, str):
        texto = str(texto)

    return len(encoding.encode(texto))


# =========================================================
# FUNCIONES AUXILIARES PARA CONVERSACIONES
# =========================================================

def _normalizar_conversacion(conversacion):
    """
    Convierte distintos formatos posibles de conversation_a / conversation_b
    a una lista de mensajes.
    """
    if conversacion is None:
        return []

    if isinstance(conversacion, np.ndarray):
        conversacion = conversacion.tolist()

    if isinstance(conversacion, tuple):
        conversacion = list(conversacion)

    if isinstance(conversacion, list):
        return conversacion

    return []


def extraer_primera_consulta(conversacion):
    """
    Extrae el primer mensaje del usuario/human.
    Soporta estructuras con:
    - role/content
    - from/value
    """
    conversacion = _normalizar_conversacion(conversacion)

    for mensaje in conversacion:
        if not isinstance(mensaje, dict):
            continue

        role = mensaje.get("role", mensaje.get("from"))

        if role in {"user", "human"}:
            return mensaje.get("content", mensaje.get("value"))

    return None


def extraer_ultima_respuesta(conversacion):
    """
    Extrae la última respuesta del assistant/gpt.
    Soporta estructuras con:
    - role/content
    - from/value
    """
    conversacion = _normalizar_conversacion(conversacion)

    respuestas = []

    for mensaje in conversacion:
        if not isinstance(mensaje, dict):
            continue

        role = mensaje.get("role", mensaje.get("from"))

        if role in {"assistant", "gpt"}:
            contenido = mensaje.get("content", mensaje.get("value"))

            if contenido:
                respuestas.append(contenido)

    return respuestas[-1] if respuestas else None


# =========================================================
# CARGA DATASET 1: CHATBOT ARENA PARQUET LOCAL
# =========================================================

print("Cargando Chatbot Arena desde Parquet local...")

arena = pd.read_parquet(ARENA_PATH)

print(f"Registros Chatbot Arena: {len(arena):,}")
print("Columnas:", arena.columns.tolist())


columnas_requeridas_arena = {
    "question_id",
    "tstamp",
    "conversation_a",
    "conversation_b",
}

faltantes_arena = columnas_requeridas_arena - set(arena.columns)

if faltantes_arena:
    raise ValueError(
        "Faltan columnas requeridas en Chatbot Arena: "
        + ", ".join(sorted(faltantes_arena))
    )


# =========================================================
# 1. TIEMPOS ENTRE LLEGADAS
# =========================================================
#
# IMPORTANTE:
# Los interarribos se calculan ANTES del merge con hardness.
# Así usamos todas las llegadas disponibles en Chatbot Arena
# y no deformamos la distribución al perder consultas.
# =========================================================

llegadas = arena[["question_id", "tstamp"]].copy()

llegadas["timestamp"] = pd.to_datetime(
    llegadas["tstamp"],
    unit="s",
    utc=True,
    errors="coerce",
)

llegadas = (
    llegadas
    .dropna(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

llegadas["interarrival_seconds"] = (
    llegadas["timestamp"]
    .diff()
    .dt.total_seconds()
)

# La primera observación no tiene interarribo
llegadas = llegadas.dropna(subset=["interarrival_seconds"])

# Conservamos por ahora los ceros:
# pueden representar consultas ocurridas en el mismo segundo.
# Los outliers y gaps largos se analizarán luego en distributions.py.

interarrivals_output = llegadas[
    [
        "question_id",
        "timestamp",
        "interarrival_seconds",
    ]
].copy()

interarrivals_output.to_csv(
    PROCESSED_DIR / "interarrivals.csv",
    index=False,
)

print("\nInterarribos generados:")
print(interarrivals_output["interarrival_seconds"].describe())


# =========================================================
# 2. CARGA DATASET 2: HARDNESS JSONL LOCAL SIN COMPRIMIR
# =========================================================

print("\nCargando dataset de hardness desde JSONL local...")

hardness = pd.read_json(
    HARDNESS_PATH,
    lines=True,
)

print(f"Registros hardness: {len(hardness):,}")
print("Columnas:", hardness.columns.tolist())


columnas_requeridas_hardness = {
    "question_id",
    "prompt",
    "score_value_1",
    "score_value_2",
    "score_value_3",
}

faltantes_hardness = columnas_requeridas_hardness - set(hardness.columns)

if faltantes_hardness:
    raise ValueError(
        "Faltan columnas requeridas en hardness: "
        + ", ".join(sorted(faltantes_hardness))
    )


# =========================================================
# 3. HARDNESS FINAL
# =========================================================

score_columns = [
    "score_value_1",
    "score_value_2",
    "score_value_3",
]

# Convertimos los scores a numéricos.
# Si hay algún valor inválido, se transforma en NaN.
for columna in score_columns:
    hardness[columna] = pd.to_numeric(
        hardness[columna],
        errors="coerce"
    )

# Nos quedamos SOLO con filas que tengan los 3 scores válidos
hardness = hardness.dropna(
    subset=score_columns
).copy()

# Calculamos la mediana de los 3 scores
hardness["hardness"] = (
    hardness[score_columns]
    .median(axis=1)
    .astype(int)
)

# Dejamos solamente hardness válidos entre 1 y 10
hardness = hardness[
    hardness["hardness"].between(1, 10)
].copy()

# =========================================================
# 4. INPUT TOKENS
# =========================================================
#
# El dataset de hardness trae el prompt, pero no los tokens.
# Los calculamos nosotros.
# =========================================================

hardness["input_tokens"] = (
    hardness["prompt"]
    .apply(contar_tokens)
)


# =========================================================
# 5. OUTPUT TOKENS DESDE CHATBOT ARENA
# =========================================================

print("\nExtrayendo respuestas y calculando output tokens...")

arena["response_a"] = (
    arena["conversation_a"]
    .apply(extraer_ultima_respuesta)
)

arena["response_b"] = (
    arena["conversation_b"]
    .apply(extraer_ultima_respuesta)
)

arena["output_tokens_a"] = (
    arena["response_a"]
    .apply(contar_tokens)
)

arena["output_tokens_b"] = (
    arena["response_b"]
    .apply(contar_tokens)
)

# Tomamos el promedio de longitud de las dos respuestas como
# longitud representativa de salida para esa consulta.
arena["output_tokens"] = (
    arena[
        [
            "output_tokens_a",
            "output_tokens_b",
        ]
    ]
    .mean(axis=1)
)


# =========================================================
# 6. MERGE POR question_id
# =========================================================

consultas = hardness.merge(
    arena[
        [
            "question_id",
            "tstamp",
            "output_tokens_a",
            "output_tokens_b",
            "output_tokens",
        ]
    ],
    on="question_id",
    how="inner",
)


consultas["timestamp"] = pd.to_datetime(
    consultas["tstamp"],
    unit="s",
    utc=True,
    errors="coerce",
)


# =========================================================
# 7. LIMPIEZA FINAL
# =========================================================

consultas = consultas.dropna(
    subset=[
        "question_id",
        "timestamp",
        "hardness",
        "input_tokens",
        "output_tokens",
    ]
)

consultas = consultas[
    (consultas["input_tokens"] > 0)
    & (consultas["output_tokens"] > 0)
].copy()

# Por consistencia, mantenemos output_tokens como valor numérico.
consultas["output_tokens"] = pd.to_numeric(
    consultas["output_tokens"],
    errors="coerce",
)


# =========================================================
# 8. DATASET PROCESADO FINAL
# =========================================================

columnas_finales = [
    "question_id",
    "timestamp",
    "prompt",
    "hardness",
    "input_tokens",
    "output_tokens_a",
    "output_tokens_b",
    "output_tokens",
]

consultas = consultas[columnas_finales].copy()

consultas.to_csv(
    PROCESSED_DIR / "consultas.csv",
    index=False,
)


# =========================================================
# 9. CONTROLES
# =========================================================

print("\n" + "=" * 50)
print("PREPROCESAMIENTO TERMINADO")
print("=" * 50)

print(f"\nInterarribos: {len(interarrivals_output):,}")
print(f"Consultas cruzadas: {len(consultas):,}")

print("\nPrimeras filas de consultas.csv:")
print(consultas.head())

print("\nValores faltantes:")
print(consultas.isna().sum())

print("\nHardness:")
print(consultas["hardness"].describe())

print("\nInput tokens:")
print(consultas["input_tokens"].describe())

print("\nOutput tokens:")
print(consultas["output_tokens"].describe())

print("\nArchivos generados:")
print(PROCESSED_DIR / "interarrivals.csv")
print(PROCESSED_DIR / "consultas.csv")
