import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import backend.variables as va
from backend.sidebar import show_sidebar
from backend.metricas import Metricas

"""
OPTIMIZED.PY

Muestra la hoja del portafolio optimizado
"""

st.set_page_config(
    page_title="Portfolio Allocation Analysis",
    page_icon="Fotos/portafolio.png",
    layout="wide",)

show_sidebar()

# ============================================      TITULO       ============================================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.write("# :material/work: Portfolio Allocation Analysis")
    st.write("##### Easily compare portfolio returns and its risk measures.")
    ""  # Add some space.
    ""  # Add some space.
st.write(f"### Analysis Type: :violet[Optimized] with {st.session_state.opcion_2}, {st.session_state.horizon} time horizon.")


#  ===========================  SAFETY CHECK =========================== 
def show_warning_with_image(msg: str):
    st.warning(msg)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Fotos/perrolaptop.GIF", use_container_width=True)
    st.stop()

# 1. Validar tickers_input
tickers = st.session_state.get("tickers_input", None)

if not tickers or (isinstance(tickers, str) and tickers.strip() == ""):
    show_warning_with_image("⚠️ No tickers selected, please :red[**input tickers in sidebar**].")

# 2. Validar existencia de df_tickers
if "df_tickers" not in st.session_state or st.session_state.df_tickers is None:
    show_warning_with_image("❌ Missing data: Click :green[Load data].")

# 3. Validar existencia de df_ETF_bechmark
if "df_ETF_bechmark" not in st.session_state or st.session_state.df_ETF_bechmark is None:
    show_warning_with_image("❌ Missing data: Click :green[Load data].")

# =======================================================================

# Load Session state variables
tickers = st.session_state.tickers_input
strategy = st.session_state.opcion_2
df_tickers_now= st.session_state.df_tickers
df_ETF_bechmark_now = st.session_state.df_ETF_bechmark

# Convertir las fechas del date_input a pandas Timestamp
fecha_inicio_var = pd.to_datetime(st.session_state.fecha_inicio_widget)
fecha_fin_var    = pd.to_datetime(st.session_state.fecha_corte_widget)

# Asegurar que la columna también es datetime64
df_tickers_now["Date"] = pd.to_datetime(df_tickers_now["Date"])
df_ETF_bechmark_now["Date"] = pd.to_datetime(df_ETF_bechmark_now["Date"])

# Cortar DataFrames
df_ticker_cortado_fechas = df_tickers_now[(df_tickers_now["Date"] >= fecha_inicio_var) & (df_tickers_now["Date"] <= fecha_fin_var)]
df_ETF_cortado_fechas = df_ETF_bechmark_now[(df_ETF_bechmark_now["Date"] >= fecha_inicio_var) & (df_ETF_bechmark_now["Date"] <= fecha_fin_var)]



#  ===========================    DEBUG   ==================
st.write(str(tickers))
st.write(str(strategy))
st.write(str(fecha_inicio_var))
st.write(str(fecha_fin_var))
#  ===========================    DEBUG   ==================

    
# Estado inicial
if "camino" not in st.session_state:
    st.session_state.camino = None


def boton_Variance():
    seleccionado_1 = st.session_state.camino == "Varianza"

    if st.button(":red[**Minimize Variance**]", key="Varianza"):
        st.session_state.camino = "Varianza"
        st.write("Varianza") 
        st.write(tickers)
        
def boton_Sharp():
    seleccionado_2 = st.session_state.camino == "Sharp"

    if st.button(":blue[**Maximize Sharp**]", key="Sharp"):
        st.session_state.camino = "Sharp"
        st.write("sharp") 
        st.write(tickers)
        
def boton_Markowitz():
    seleccionado_3 = st.session_state.camino == "Marko"

    if st.button(":violet[**Markowitz Optimization**]", key="Marko"):
        st.session_state.camino = "Marko"
        st.write("markowitz") 
        st.write(tickers)

# Layout en 3 columnas
col1, col2, col3 = st.columns(3)
with col1:
    boton_Variance()
with col2:
    boton_Sharp()
with col3:
    boton_Markowitz()

# Mostrar selección
if st.session_state.camino:
    st.write("### Seleccionaste:", st.session_state.camino)


def Mostrar_Optimization_Output():
    pesos_output = {"AAPL": 0.25, "MSFT": 0.35, "VOO": 0.40}                                #--------------- OUTPUT DE PESOS LUEGO DE LA OPTIMIZACION -------------------------
    df_pesos = pd.DataFrame.from_dict(pesos_output, orient="index", columns=["Weight"])
    df_pesos = df_pesos.sort_values("Weight", ascending=False)

    st.subheader("Portfolio Weights")
    st.dataframe(df_pesos.style.format("{:.2%}"))

Mostrar_Optimization_Output()
    
col1, col2 = st.columns([0.9, 1])

with col2:
    with st.container():
        Metricas()


with col1:
    with st.container():
        st.subheader("Portfolio Chart")

        ticker = "TICKER"
               # -------------------------
        # BENCHMARK: crecer desde $1
        # -------------------------
        df_bench = df_ETF_bechmark_now.copy()
        df_bench["Date"] = pd.to_datetime(df_bench["Date"])
        df_bench = df_bench.sort_values("Date")

        # Suponemos que los rendimientos son diarios en decimal (ej: 0.0012)
        df_bench["cum_growth"] = (1 + df_bench[df_bench.columns[1]]).cumprod()

        # -------------------------
        # SIMULACIONES RANDOM
        # -------------------------
        precio_inicial = 1       # para que compare con el benchmark
        mu = 0.10 / 365
        sigma = 0.20 / np.sqrt(365)
        dias = len(df_bench)     # misma longitud que el benchmark

        ruido1 = np.random.normal(mu, sigma, dias)
        precios1 = precio_inicial * np.exp(np.cumsum(ruido1))

        ruido2 = np.random.normal(mu, sigma, dias)
        precios2 = precio_inicial * np.exp(np.cumsum(ruido2))

        # -------------------------
        # PLOT
        # -------------------------
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.set_facecolor("#22304A")
        fig.patch.set_facecolor("#22304A")

        # Líneas
        ax.plot(df_bench["Date"], df_bench["cum_growth"], color="yellow", linewidth=2,
                label="Benchmark ($1 inicial)")

        ax.plot(df_bench["Date"], precios1, color="red", label="Simulation 1")
        ax.plot(df_bench["Date"], precios2, color="gray", alpha=0.7, label="Simulation 2")

        # Títulos y estilo
        ax.set_title("Benchmark vs Random Simulations", color="white")
        
        ax.tick_params(colors="white")
        for side in ["bottom", "top", "left", "right"]:
            ax.spines[side].set_color("white")

        ax.legend(facecolor="#1C273B", labelcolor="white")

        st.pyplot(fig)
        
#  ===========================    DEBUG   ==================
st.write(df_ticker_cortado_fechas.head(5))
st.write(df_ETF_cortado_fechas.head(5))
st.write("----------------")
st.write(df_ticker_cortado_fechas.tail(5))
st.write(df_ETF_cortado_fechas.tail(5))
