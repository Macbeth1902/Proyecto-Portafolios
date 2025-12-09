import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import backend.variables as va
from backend.sidebar import show_sidebar
from backend.metricas import Metricas
from backend.portfolio_analyzer import PortfolioAnalyzer
from backend.data import load_returns,load_ETF_bechmark,cargar_cortes_fechas,load_data_bounds,compare_values

### CUSTOM.PY
### Muestra la hoja del portafolio custom

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

fecha_inicio_titulo = pd.to_datetime(st.session_state.fecha_inicio_widget)
fecha_fin_titulo    = pd.to_datetime(st.session_state.fecha_corte_widget)

# Formato: Mes YYYY  (por ejemplo: Enero 2023)
fecha_inicio_fmt = fecha_inicio_titulo.strftime("%B %Y")
fecha_fin_fmt    = fecha_fin_titulo.strftime("%B %Y")
st.write(f"### Analysis Type: :blue[Custom]  {st.session_state.opcion_2}, with {fecha_inicio_fmt} - {fecha_fin_fmt} time horizon.")
st.write("---")

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
    
    
if "df_todos" not in st.session_state:
    st.session_state.df_todos = load_returns([
    "SPLG", "EWC", "IEUR", "EEM", "EWJ",
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"])            
# # =========================== LOAD DATA ===========================

# Load Session state variables
tickers = st.session_state.tickers_input
strategy = st.session_state.opcion_2
df_tickers_now= st.session_state.df_tickers
df_ETF_bechmark_now = st.session_state.df_ETF_bechmark
df_Todos = st.session_state.df_todos

# Convertir las fechas del date_input a pandas Timestamp
fecha_inicio_var = pd.to_datetime(st.session_state.fecha_inicio_widget)
fecha_fin_var    = pd.to_datetime(st.session_state.fecha_corte_widget)

# Asegurar que la columna también es datetime64
df_tickers_now["Date"] = pd.to_datetime(df_tickers_now["Date"])
df_ETF_bechmark_now["Date"] = pd.to_datetime(df_ETF_bechmark_now["Date"])
df_Todos["Date"] = pd.to_datetime(df_ETF_bechmark_now["Date"])

# Cortar DataFrames
df_ticker_cortado_fechas = df_tickers_now[(df_tickers_now["Date"] >= fecha_inicio_var) & (df_tickers_now["Date"] <= fecha_fin_var)]
df_ETF_cortado_fechas = df_ETF_bechmark_now[(df_ETF_bechmark_now["Date"] >= fecha_inicio_var) & (df_ETF_bechmark_now["Date"] <= fecha_fin_var)]
df_Todos_cortado_fechas = df_Todos[(df_Todos["Date"] >= fecha_inicio_var) & (df_Todos["Date"] <= fecha_fin_var)]

#  ===========================    DEBUG   ==================
#st.write(str(tickers))
#st.write(str(strategy))
#st.write(str(fecha_inicio_var))
#st.write(str(fecha_fin_var))


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

        # old_df = st.session_state.df_custom.copy()

        # Nuevo DF base
        n = len(tickers_now)

        new_df = pd.DataFrame({
            "ETFs": tickers_now,
            "Weights (%)": [round(100/n, 2)] * n
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


   
    col1,col2,col3 = st.columns([0.3,2,0.3])
    with col2:
        st.write("### :blue[Custom] Weight Input")
        # -------- FORM PARA EVITAR AUTOUPDATE -------- #
        form_key = f"form_pesos_{strategy}_{'_'.join(tickers_now)}"
        with st.form(form_key):
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
        raw = edited_df["Weights (%)"].sum()
        total_warning = raw * 100 if raw <= 1.001 else raw
        # Validar que la suma sea 100 (con tolerancia por flotantes)
        if abs(total_warning - 100) > 1e-6:
            st.error(f"❌ Los pesos deben sumar **100%**. La suma actual es: **{total:.2f} %**")
        else:
            st.session_state.df_custom = edited_df.copy()
            st.success("✔️ Cambios aplicados correctamente.")
            # CONTINUAR---------------

    # -------- Mostrar valores finales -------- #
    weights_decimal = st.session_state.df_custom["Weights (%)"] / 100
    weights_decimal.index = st.session_state.df_custom["ETFs"]
    # Alinear DataFrame con los tickers de los pesos
    df_returns_aligned = df_ticker_cortado_fechas.copy()

    # Mantener 'Date' y solo los tickers que tenemos en weights_decimal
    tickers_in_df = [ticker for ticker in weights_decimal.index if ticker in df_returns_aligned.columns]

    df_returns_aligned = df_returns_aligned[["Date"] + tickers_in_df]
    
    # Reindexar pesos para coincidir con columnas del DataFrame
    weights_aligned = weights_decimal.loc[tickers_in_df]
    st.session_state.weights_aligned_custom = weights_aligned


    #st.write(f"{type(df_returns_aligned)} - {type(weights_aligned)} - {type(st.session_state.Tasa_libre_de_riesgo_widget)})")
    analyzer = PortfolioAnalyzer(df_returns_aligned, weights_aligned, st.session_state.Tasa_libre_de_riesgo_widget)
    
    metrics = analyzer.analyze()
    st.session_state.metrics = metrics
    Metricas(metrics)

    return analyzer

# ========================== METRICAS ==========================
analyzer = Estado_1_CUSTOM_2()

if analyzer:

    # CÁLCULO DE MÉTRICAS CUSTOM
    metrics = st.session_state.metrics

    # ================= Benchmark =================
    # Benchmark según estrategia
    st.subheader("Comparizon :blue[Custom] vs :gray[Benchmark] ")

    if strategy == "by Regions":
        benchmark_weights = { # Escribimos el "portafolio predefinido"
            "SPLG": 0.7062, "EWC": 0.0323,
            "IEUR": 0.1176, "EEM": 0.0902, "EWJ": 0.0537
        }
    else:
        benchmark_weights = {
            "XLC": 0.0999, "XLY": 0.1025, "XLP": 0.0482, "XLE": 0.0295,
            "XLF": 0.1307, "XLV": 0.0958, "XLI": 0.0809, "XLB": 0.0166,
            "XLRE": 0.0187, "XLK": 0.3535, "XLU": 0.0237
        }

    tickers_bench = [t for t in benchmark_weights if t in df_Todos_cortado_fechas.columns]
    df_bench_aligned = df_bench_aligned = df_Todos_cortado_fechas[["Date"] + tickers_bench]
    weights_bench_aligned = pd.Series({t: benchmark_weights[t] for t in tickers_bench})

    benchmark_analyzer = PortfolioAnalyzer(
        df_bench_aligned,
        weights_bench_aligned,
        st.session_state.Tasa_libre_de_riesgo_widget,
        analyzer.portfolio_returns
    )

    benchmark_metrics = benchmark_analyzer.analyze()
    
    col1,col2 = st.columns(2)
    with col1:
        # ================= TABLA COMPARATIVA =================
        df_compare = pd.DataFrame({
            "Custom": metrics, ## Portafolio definido por el usuario
            "Benchmark": benchmark_metrics ## Portafolio base o predefinido
        }).round(4)
        
        df_compare["Winner"] = df_compare.apply(compare_values, axis=1)
        st.dataframe(df_compare)

    # ================= GRÁFICA PRINCIPAL =================

    cum_custom = (1 + analyzer.portfolio_returns).cumprod()
    cum_bench = (1 + benchmark_analyzer.portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(cum_custom, label="Custom", linewidth=2.5, color="#58a6ff")
    ax.plot(cum_bench, label="Benchmark", linewidth=2.5, color="#859d8e")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#c9d1d9")
    

    ax.set_facecolor("#22304A")
    fig.patch.set_facecolor("#22304A")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
   
    ax.legend(facecolor="#22304A", labelcolor="white")
    with col2:
        st.pyplot(fig)
        

    # ================= VISUALIZACIONES =================
    st.write("### :blue[Custom] Portfolio Drawdown & Return Visualization")
    analyzer.plot_portfolio_analysis()
    
    st.write(f"**Benchmark weights:** {weights_bench_aligned}")
    st.write(f"**Custom weights:** {st.session_state.weights_aligned_custom}")
    