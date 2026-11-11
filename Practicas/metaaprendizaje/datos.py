"""Carga de los archivos originales y recuperación de los pliegues del curso."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


BANK_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
BANK_FICHA = "https://archive.ics.uci.edu/dataset/222/bank+marketing"


def _directorio_practicas(base: Path) -> Path:
    """Acepta la raíz del repositorio o el directorio Practicas."""
    base = Path(base)
    if (base / "Practicas" / "datos").is_dir():
        return base / "Practicas"
    if (base / "datos").is_dir():
        return base
    raise FileNotFoundError(f"No existe datos/ ni Practicas/datos/ bajo {base}.")


def cargar_desercion(base: Path):
    """Devuelve X, y, cinco pares de índices e IDs, en el orden original de P3.

    Las huellas de los dos CSV se contrastan con los metadatos archivados.
    Se recupera la asignación por ID, sin volver a sortear las particiones.
    """
    practica = _directorio_practicas(base)
    datos = practica / "datos"
    resultados = practica / "resultados_practica_03"
    metadata = json.loads((resultados / "metadatos.json").read_text(encoding="utf-8"))
    for nombre in ("desercion-entrenamiento.csv", "desercion-prueba.csv"):
        ruta = datos / nombre
        esperado = metadata["huellas_datos_sha256"][nombre]
        if sha256(ruta.read_bytes()).hexdigest() != esperado:
            raise ValueError(f"{nombre} no coincide con los datos utilizados en P3.")

    train = pd.read_csv(datos / "desercion-entrenamiento.csv", dtype={"id": "string"},
                        encoding="utf-8-sig")
    test = pd.read_csv(datos / "desercion-prueba.csv", dtype={"id": "string"},
                       encoding="utf-8-sig")
    if train.shape != (3539, 38) or test.shape != (885, 31):
        raise ValueError("Las dimensiones de los CSV de deserción no son las de P3.")
    if not train.columns.is_unique or not test.columns.is_unique:
        raise ValueError("Hay nombres de columnas duplicados en deserción.")
    for df in (train, test):
        if df["id"].isna().any() or not df["id"].is_unique:
            raise ValueError("Los IDs de deserción deben ser únicos y no nulos.")
    if not set(train["id"]).isdisjoint(test["id"]):
        raise ValueError("Hay IDs compartidos entre entrenamiento y prueba.")

    columnas = [c for c in test if c != "id"]
    if columnas != metadata["columnas_primer_semestre"]:
        raise ValueError("El orden de los atributos no coincide con el registrado en P3.")
    X = train[columnas].apply(pd.to_numeric, errors="raise")
    y = train["Target"].astype("string").str.strip().str.lower().astype(str)
    if set(y) != {"dropout", "enrolled", "graduate"}:
        raise ValueError("Las clases de deserción son inesperadas.")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Hay valores infinitos en los atributos de deserción.")

    asignacion = pd.read_csv(resultados / "pliegues.csv", dtype={"id": "string"})
    if (list(asignacion) != ["id", "pliegue_externo"]
            or not asignacion["id"].is_unique
            or asignacion["id"].isna().any()
            or set(asignacion["id"]) != set(train["id"])):
        raise ValueError("La asignación de pliegues no contiene exactamente los IDs de P3.")
    pliegue = asignacion.set_index("id")["pliegue_externo"].reindex(train["id"]).to_numpy()
    if set(pliegue) != {1, 2, 3, 4, 5}:
        raise ValueError("Se esperaban los cinco pliegues originales de P3.")
    cv = [(np.flatnonzero(pliegue != i), np.flatnonzero(pliegue == i))
          for i in range(1, 6)]
    return X, y, cv, train["id"].copy()


def cargar_bank(base: Path):
    """Carga las 41 188 filas originales; X excluye únicamente duration e y.

    Las categorías, incluidos los valores 'unknown', permanecen intactas.
    Toda transformación aprendida corresponde a la Pipeline del modelo.
    """
    ruta = _directorio_practicas(base) / "datos" / "bank-additional-full.csv"
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Falta {ruta}. Ejecuta descargar_bank(base) para obtener el archivo oficial."
        )
    df = pd.read_csv(ruta, sep=";")
    if df.shape != (41188, 21) or not df.columns.is_unique:
        raise ValueError("Se requiere bank-additional-full.csv original de 41188 × 21.")
    esperadas = {
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays",
        "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
        "euribor3m", "nr.employed", "y",
    }
    if set(df) != esperadas or set(df["y"]) != {"yes", "no"}:
        raise ValueError("El esquema o las clases de Bank Marketing son inesperados.")
    X = df.drop(columns=["duration", "y"])
    y = df["y"].map({"yes": 1, "no": 0}).astype("int64")
    return X, y


def descargar_bank(base: Path) -> Path:
    """Descarga de UCI solo el CSV completo y su descripción, si faltan.

    Lee miembros explícitos del ZIP en memoria, sin extraer directorios ni
    sustituir archivos existentes. Una copia ya presente se valida al cargarla.
    """
    carpeta = _directorio_practicas(base) / "datos"
    destino = carpeta / "bank-additional-full.csv"
    descripcion = carpeta / "bank-additional-names.txt"
    if destino.is_file() and descripcion.is_file():
        cargar_bank(base)
        return destino
    solicitud = Request(BANK_URL, headers={"User-Agent": "Practica4-metaaprendizaje/1.0"})
    with urlopen(solicitud, timeout=60) as respuesta:
        archivo = respuesta.read()
    with ZipFile(BytesIO(archivo)) as exterior:
        # El paquete oficial de UCI contiene el ZIP específico de esta variante.
        if "bank-additional.zip" in exterior.namelist():
            with ZipFile(BytesIO(exterior.read("bank-additional.zip"))) as interior:
                csv = interior.read("bank-additional/bank-additional-full.csv")
                texto = interior.read("bank-additional/bank-additional-names.txt")
        else:
            csv = exterior.read("bank-additional/bank-additional-full.csv")
            texto = exterior.read("bank-additional/bank-additional-names.txt")
    # Validar el miembro antes de escribirlo: evita guardar una variante incorrecta.
    comprobacion = pd.read_csv(BytesIO(csv), sep=";")
    if comprobacion.shape != (41188, 21) or not {"duration", "y"} <= set(comprobacion):
        raise ValueError("El miembro descargado no es bank-additional-full.csv.")
    if not destino.exists():
        destino.write_bytes(csv)
    if not descripcion.exists():
        descripcion.write_bytes(texto)
    cargar_bank(base)
    return destino

