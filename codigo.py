
import yfinance as yf
import pandas as pd
from functools import lru_cache
from pathlib import Path


# ETFs de Regiones y Sectores
REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]
SECTORES = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI",
            "XLB","XLRE","XLK","XLU"]


# Carpeta de salida
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)  # Crear carpeta si no existe


# Descargar precios ajustados
@lru_cache(maxsize=10)
def _download_prices(tickers: tuple, start="2020-01-01") -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"Error: No se pudieron descargar datos para {tickers}")

    # MultiIndex (varios tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"].copy() if "Close" in raw.columns.levels[0] else raw.iloc[:, :len(tickers)].copy()
    else:
        df = raw.iloc[:, 0].to_frame(name=tickers[0])

    return df


# Limpiar datos
def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="any")
    return df


# Cargar precios
def load_prices(universe="regiones", start="2020-01-01") -> pd.DataFrame:
    if universe == "regiones":
        tickers = tuple(REGIONES)
    elif universe == "sectores":
        tickers = tuple(SECTORES)
    else:
        if isinstance(universe, str):
            tickers = (universe,)
        elif isinstance(universe, list):
            tickers = tuple(universe)
        else:
            raise ValueError("Universe inválido. Usa 'regiones', 'sectores' o lista de tickers.")

    df = _download_prices(tickers, start=start)
    df = _clean_data(df)
    return df


# Cargar rendimientos
def load_returns(universe="regiones", start="2020-01-01") -> pd.DataFrame:
    prices = load_prices(universe, start=start)
    returns = prices.pct_change().dropna()
    return returns


# Guardar cada ETF en CSV individual
def save_returns_to_csv(returns_df: pd.DataFrame, universe: str):
    # Carpeta para este universo
    dir_path = OUTPUT_DIR / universe
    dir_path.mkdir(exist_ok=True)

    for col in returns_df.columns:
        file_path = dir_path / f"{col}_returns.csv"
        returns_df[[col]].to_csv(file_path, index=True)
        print(f"✅ Guardado: {file_path}")


# Verificación rápida de funcionalidad
if __name__ == "__main__":
    for universe in ["regiones", "sectores"]:
        returns_df = load_returns(universe)
        print(f"\nPrimeras filas de rendimientos ({universe}):")
        print(returns_df.head())

        save_returns_to_csv(returns_df, universe)