import streamlit as st
import pandas as pd


"""
DATA.PY

Obtener los dataframes de la carpeta con .csv
y tenerlos en Cache
"""
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ARREGLAR LAS RUTAS CUANDO SEA DEPLOYMENT A STREAMLIT


@st.cache_data
def load_returns(tickers):
    """
    Carga todos.csv, filtra por tickers y guarda el resultado en cache.
    Si se vuelve a llamar con los mismos tickers NO se ejecuta nuevamente.
    """

    # Normalizamos para evitar que cambie el cache por el orden de tickers
    tickers = sorted(list(tickers))

    df = pd.read_csv("MarketData/todos.csv")

    # Asegurar que la columna Date esté siempre
    if "Date" not in df.columns:
        raise ValueError("El archivo todos.csv necesita una columna Date")

    # Filtrar columnas
    columnas = ["Date"] + tickers
    df = df[columnas]

    # Convertir fechas
    df["Date"] = pd.to_datetime(df["Date"])

    return df

@st.cache_data
def load_ETF_bechmark(strategy: str):
    """
    Carga el dataset adecuado según la estrategia.
    - 'by Regions' → SPY.csv
    - 'by Sectors' → ACWI.csv
    
    El resultado queda cacheado y NO se vuelve a ejecutar
    si se usa la misma estrategia.
    """

    base_path = "MarketData/"

    if strategy == "by Regions":
        file = base_path + "SPY.csv"

    elif strategy == "by Sectors":
        file = base_path + "ACWI.csv"

    else:
        raise ValueError(f"Estrategia no válida: {strategy}")

    df = pd.read_csv(file)

    # Convertir columna fecha en caso de necesitarlo
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df

@st.cache_data
def cargar_cortes_fechas():
    df_fechas_load = pd.read_csv("MarketDATA/todos.csv", parse_dates=["Date"])
    return df_fechas_load

@st.cache_data
def load_data_bounds(date):
    return date
    
