import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import variables as va
from sidebar import show_sidebar
from metricas import Metricas

"""
CUSTOM.PY

Muestra la hoja del portafolio custom
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
st.write(f"### Analysis Type: :blue[Custom] with {st.session_state.opcion_2}, {st.session_state.horizon} time horizon.")


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

def Estado_1_CUSTOM_2():

    opcion_21 = strategy        # estrategia: Regions/Sectors
    tickers_now = st.session_state.tickers_input  # lista actual de tickers

    # Diccionario según la estrategia del sidebar
    if opcion_21 == "by Regions":
        desc_dict = va.ETF_regions_desc
    else:
        desc_dict = va.ETF_sectors_desc

    # --- 1) Inicializar df_custom si no existe aún --- #
    if "df_custom" not in st.session_state:

        n = len(tickers_now)
        df_init = pd.DataFrame({
            "ETFs": tickers_now,
            "Weights (%)": [100/n] * n
        })

        df_init["ETF that tracks the"] = df_init["ETFs"].map(desc_dict)

        st.session_state.df_custom = df_init.copy()
        st.session_state.last_strategy = opcion_21
        st.session_state.last_tickers = tickers_now.copy()

    # --- 2) Detectar cambios en los tickers --- #
    if st.session_state.last_tickers != tickers_now:

        old_df = st.session_state.df_custom.copy()

        # Nuevo DF base
        n = len(tickers_now)

        new_df = pd.DataFrame({
            "ETFs": tickers_now,
            "Weights (%)": [100/n] * n
        })

        # Actualizar descripciones
        new_df["ETF that tracks the"] = new_df["ETFs"].map(desc_dict)

        # Guardar actualización
        st.session_state.df_custom = new_df.copy()
        st.session_state.last_tickers = tickers_now.copy()

    # --- 3) Detectar cambio de estrategia (Regions/Sectors) --- #
    if st.session_state.last_strategy != opcion_21:

        new_desc = va.ETF_regions_desc if opcion_21 == "by Regions" else va.ETF_sectors_desc

        st.session_state.df_custom["ETF that tracks the"] = \
            st.session_state.df_custom["ETFs"].map(new_desc)

        st.session_state.last_strategy = opcion_21


    st.write("### Configuración de Pesos (Custom)")

    # -------- FORM PARA EVITAR AUTOUPDATE -------- #
    with st.form("form_pesos"):

        edited_df = st.data_editor(
            st.session_state.df_custom,
            num_rows="fixed",
            column_config={
                "Weights (%)": st.column_config.NumberColumn(
                    "Weights (%)",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f %%"
                )
            },
            key="editor_custom"
        )
        # PARA PODER VER CUANTO LLEVAMOS
        preview_sum = edited_df["Weights (%)"].sum()

        if abs(preview_sum - 100) < 0.01:
            st.success(f"✔ La suma actual es: **{preview_sum:.2f}%**")
        elif preview_sum < 100:
            st.warning(f"⚠ La suma es **{preview_sum:.2f}%** → faltan {100 - preview_sum:.2f}%")
        else:
            st.error(f"❌ La suma es **{preview_sum:.2f}%** → excede {preview_sum - 100:.2f}%")

        submit = st.form_submit_button("Aplicar Cambios")


    if submit:
        total = edited_df["Weights (%)"].sum()

        # Validar que la suma sea 100 (con tolerancia por flotantes)
        if abs(total - 100) > 1e-6:
            st.error(f"❌ Los pesos deben sumar **100%**. La suma actual es: **{total:.2f} %**")
        else:
            st.session_state.df_custom = edited_df.copy()
            st.success("✔️ Cambios aplicados correctamente.")
            # CONTINUAR---------------

    # -------- Mostrar valores finales -------- #
    st.write("### Pesos finales (formato decimal):")
    weights_decimal = st.session_state.df_custom["Weights (%)"] / 100
    st.write(weights_decimal)
    
Estado_1_CUSTOM_2()



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