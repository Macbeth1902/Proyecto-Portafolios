
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
            raise ValueError("Universo inválido. Usa 'regiones', 'sectores' o una lista de tickers.")

    df = _download_prices(tickers, start=start)
    df = _clean_data(df)
    return df


# Cargar rendimientos
def load_returns(universe="regiones", start="2020-01-01") -> pd.DataFrame:
    prices = load_prices(universe, start=start)
    returns = prices.pct_change().dropna()
    return returns


# Guardar cada ETF en CSV, tanto individual como consolidado
def save_all_csvs():
    # Regiones
    regiones_ret = load_returns("regiones")
    regiones_ret.to_csv(OUTPUT_DIR / "regiones.csv")
    print("Guardado: regiones.csv")

    # Sectores
    sectores_ret = load_returns("sectores")
    sectores_ret.to_csv(OUTPUT_DIR / "sectores.csv")
    print("Guardado: sectores.csv")

    # Consolidado
    todos_ret = pd.concat([regiones_ret, sectores_ret], axis=1)
    todos_ret.to_csv(OUTPUT_DIR / "todos.csv")
    print("Guardado: todos.csv")

    # CSV individual para cada ETF
    for universe, df in [("regiones", regiones_ret), ("sectores", sectores_ret)]:
        folder = OUTPUT_DIR / universe
        folder.mkdir(exist_ok=True)

        for ticker in df.columns:
            df[[ticker]].to_csv(folder / f"{ticker}.csv")
            print(f"Guardado CSV individual: {ticker} en /{universe}/")

## PRUEBA
if __name__ == "__main__":
    save_all_csvs()
    print("\n CSV exportados correctamente.")