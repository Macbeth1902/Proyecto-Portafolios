import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import io
import scipy.optimize as op

import backend.variables as va
from backend.sidebar import show_sidebar
from backend.metricas import Metricas
from backend.portfolio_analyzer import PortfolioAnalyzer
from backend.data import load_returns,load_ETF_bechmark,cargar_cortes_fechas,load_data_bounds,compare_values
from backend.Black_Litterman import black_litterman_portfolio

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
st.write(f"### Analysis Type: :violet[Optimized]  {st.session_state.opcion_2}, with {fecha_inicio_fmt} - {fecha_fin_fmt} time horizon.")


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
df_Todos = st.session_state.df_todos


# Convertir las fechas del date_input a pandas Timestamp
fecha_inicio_var = pd.to_datetime(st.session_state.fecha_inicio_widget)
fecha_fin_var    = pd.to_datetime(st.session_state.fecha_corte_widget)

# Asegurar que la columna también es datetime64
df_tickers_now["Date"] = pd.to_datetime(df_tickers_now["Date"])
df_ETF_bechmark_now["Date"] = pd.to_datetime(df_ETF_bechmark_now["Date"])
df_Todos["Date"] = pd.to_datetime(df_Todos["Date"])


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

    
# Estado inicial
if "camino" not in st.session_state:
    st.session_state.camino = None


def matrices_activos():
    """
    Carga los rendimientos diarios de una lista de tickers usando la función `load_returns`,
    procesa los datos, y calcula la matriz de varianza-covarianza, la matriz de correlación
    y el vector de mínima varianza.

    Retorna:
    - df_filtrado_5_años (pd.DataFrame): DataFrame con los rendimientos procesados, con una columna 'date'.
    - retornos_sin_fecha (pd.DataFrame): DataFrame con los rendimientos procesados, con el índice de fecha.
    - mtx_var_covar (np.ndarray): Matriz de varianza-covarianza de los retornos.
    - mtx_correl (pd.DataFrame): Matriz de correlación de los retornos.
    - min_var_vector (np.ndarray): Vector de pesos para la cartera de mínima varianza.
    """

    # Usar la función load_returns para obtener los rendimientos.
    # Se asume que load_returns ya entrega los datos limpios y en el rango de tiempo deseado (últimos 5 años).
    retornos_sin_fecha = df_ticker_cortado_fechas.set_index("Date")
    
    # ================== REDUNCIANCIA CHECK ======================== 
    #st.write(retornos_sin_fecha.head(5))
    if retornos_sin_fecha.empty:
        raise ValueError("No hay datos suficientes para los tickers proporcionados.")
    
    # Número real de activos con datos
    tickers_validos = retornos_sin_fecha.columns.tolist()
    n_activos = len(tickers_validos)

    if n_activos != len(tickers):
        show_warning_with_image("⚠️ Tickers selected changed, please Click :green[**Load data**] in sidebar.")
    # ========================================================================
    
    # SOLO TOMAMOS LOS QUE TIENEN RENDIMIENTOS   
    retornos_sin_fecha = retornos_sin_fecha.select_dtypes(include=["float","int"])
    
    # 'df_filtrado_5_años' debe tener una columna 'date'
    df_filtrado_5_años = retornos_sin_fecha.reset_index().rename(columns={'Date': 'date'})

    # Calcular la matriz de varianza-covarianza y la matriz de correlación
    mtx_var_covar = retornos_sin_fecha.cov().values
    mtx_correl = retornos_sin_fecha.corr()

    # Mínima varianza con descomposición de autovalores
    eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_covar)
    min_var_vector = eigenvectors[:, 0] # El primer autovector corresponde al autovalor más pequeño (mínima varianza)

    # Impresión y visualización

    df_var_covar = pd.DataFrame(mtx_var_covar, index=tickers, columns=tickers)
    series_min_var = pd.Series(min_var_vector, index=tickers)

    # Los guardamos en session state
    st.session_state.df_filtrado_5_años_session =  df_filtrado_5_años
    st.session_state.retornos_sin_fecha_session = retornos_sin_fecha
    st.session_state.df_var_cov_session = df_var_covar
    st.session_state.mtx_correl_session = mtx_correl
    st.session_state.series_min_var_session = series_min_var
    
    return df_filtrado_5_años, retornos_sin_fecha, mtx_var_covar, mtx_correl, min_var_vector
    
def Heat_map():
    # HEAT MAP
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#22304A")      # Fondo de toda la figura
    ax.set_facecolor("#22304A")             # Fondo del área del plot    
    ax.set_title(f"Correlation Matrix", color="white")

    sns.heatmap(st.session_state.mtx_correl_session, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax ,cbar=False)
    ax.tick_params(colors="white")
    # Mostrar en Streamlit
    col1,col2,col3 = st.columns([1,1.5,1])
    with col2:
        st.pyplot(fig)

    #return df_filtrado_5_años, retornos_sin_fecha, mtx_var_covar, mtx_correl, min_var_vector

matrices_activos()
st.write("---")
""

# ==================PORTAFOLIOS ==================

def display_portfolio_analysis():

    # Construcción de matriz de retornos y parámetros de Markowitz
    # =====================================================================
    returns = st.session_state.retornos_sin_fecha_session
    #st.write(returns.head(5))
    mean_returns_local = returns.mean() * 252   # Retorno promedio anual de cada activo
    cov_matrix_local = st.session_state.df_var_cov_session  * 252      # Matriz de covarianza anualizada
    n_local = len(returns.columns) 
    
    # DEBUG
    #st.write("**Retornos diarios:**", returns)
    #st.write("**Retornos anuales promedio:**", mean_returns_local)
    #st.write("**Covarianza anualizada:**", cov_matrix_local)
    #st.write("**Número de activos:**", n_local)

    # --- Funciones anidadas para la optimización de Markowitz ---

    # Función rendimiento y riesgo del portafolio (anidada)
    def portafolio_rendimiento(weights):
        ret = np.dot(weights, mean_returns_local)                # w^T μ
        vol = np.sqrt(weights @ cov_matrix_local @ weights.T)    # sqrt(w^T Σ w)
        return ret, vol

    # Función para pesos de mínima volatilidad (anidada)
    def minima_volatilidad():
        x0 = np.ones(n_local)/n_local      # Condición inicial: pesos iguales
        bounds = tuple((0,1) for _ in range(n_local))  # Pesos entre 0 y 1 (no short)
        restriccion = ({'type':'eq','fun':lambda w: np.sum(w)-1})  # Suma de pesos = 1

        # Minimizamos solo la volatilidad:
        # minimize( σp(w) )
        result = op.minimize(
            lambda w: portafolio_rendimiento(w)[1],
            x0, constraints=restriccion, bounds=bounds
        )
        return result.x, portafolio_rendimiento(result.x)


    # Función para los pesos de máximo retorno (anidada)
    def maximo_retorno():
        x0 = np.ones(n_local)/n_local
        bounds = tuple((0,1) for _ in range(n_local))
        restriccion = ({'type':'eq','fun':lambda w: np.sum(w)-1})

        # maximize(rp)  equivale a minimize(-rp)
        result = op.minimize(
            lambda w: -portafolio_rendimiento(w)[0],
            x0, constraints=restriccion, bounds=bounds
        )
        return result.x, portafolio_rendimiento(result.x)



    # Función para los pesos de máximo sharpe (anidada)
    def maximo_sharpe():           #risk_free=0.0
        """
        Sharpe ratio: S = (rp - rf) / σp
        maximize(S)  <--> minimize( -S )
        """
        risk_free = st.session_state.Tasa_libre_de_riesgo_widget/100
        x0 = np.ones(n_local)/n_local
        bounds = tuple((0,1) for _ in range(n_local))
        restriccion = ({'type':'eq','fun':lambda w: np.sum(w)-1})

        def neg_sharpe(w):
            r, vol = portafolio_rendimiento(w)
            return -(r - risk_free) / vol    # Negativo porque minimize() debe maximizar el Sharpe
        

        result = op.minimize(
            neg_sharpe, x0, constraints=restriccion, bounds=bounds
        )
        return result.x, portafolio_rendimiento(result.x)

    # Función para el portafolio de mínima varianza para un retorno objetivo (anidada)
    def min_varianza_retorno():          #retorno_deseado
        
        retorno_deseado = st.session_state.Rendimiento_objetivo_widget/100

        # Punto inicial: pesos iguales
        x0 = np.ones(n_local) / n_local

        # Restricciones: suma de pesos = 1  y retorno objetivo
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},                     # suma = 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns_local) - retorno_deseado}  # retorno deseado
        )

        # No permitir posiciones cortas (0 a 1)
        bounds = tuple((0, 1) for _ in range(n_local))

        # Función objetivo: varianza
        def variance(w):
            return w @ cov_matrix_local @ w.T

        # Optimización
        result = op.minimize(
            variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Retornar pesos + rendimiento y riesgo resultante
        w = result.x
        ret, vol = portafolio_rendimiento(w)

        return w, ret, vol
    
    
    # ===========================
    # EJECUTAR TODAS LAS OPTIMIZACIONES
    # ===========================

    w_min_vol, (ret_min_vol, vol_min_vol) = minima_volatilidad()
    w_max_ret, (ret_max_ret, vol_max_ret) = maximo_retorno()
    w_max_sharpe, (ret_max_sharpe, vol_max_sharpe) = maximo_sharpe()
    w_min_mark, ret_min_mark, vol_min_mark = min_varianza_retorno()


    # ===========================
    # MOSTRAR RESULTADOS EN STREAMLIT
    # ===========================
    tickers_visualizar = returns.columns
    
    col1 , col2, col3, col4= st.columns([1,1,1,1])
    with col1:
        st.subheader("Min Variance Portfolio")
        df_visualizar1 = pd.DataFrame({"ETFs": tickers_visualizar,"Weight": w_min_vol.round(4)}) # MIN VARIANZE
        st.write("Weights:",df_visualizar1 )
        st.session_state.df_min_var = df_visualizar1

        col1_1, col2_2 = st.columns(2)
        col1_1.metric(
            label="Return",
            value=f"{ret_min_vol*100:.2f}%",border=True, width="content")
        col2_2.metric(
            label="Volatility",
            value=f"{vol_min_vol*100:.2f}%",border=True,width="content")
        

    with col3:
        st.subheader("Max Sharp Portfolio")
        df_visualizar3 = pd.DataFrame({"ETFs": tickers_visualizar,"Weight": w_max_sharpe.round(4)}) # MAX SHARP
        st.write("Weights:",df_visualizar3 )
        st.session_state.df_max_sharp_3 = df_visualizar3
        #st.write(f"Retorno: {ret_max_sharpe:.4f}, Volatilidad: {vol_max_sharpe:.4f}")
        col1_1, col2_2 = st.columns(2)
        col1_1.metric(
            label="Return",
            value=f"{ret_max_sharpe*100:.2f}%",border=True, width="content")
        col2_2.metric(
            label="Volatility",
            value=f"{vol_max_sharpe*100:.2f}%",border=True,width="content")
    
    with col4:
        st.subheader(f"Markowitz Opti Portfolio") #return: {str(st.session_state.Rendimiento_objetivo_widget)} \% Portfolio
        df_visualizar4 = pd.DataFrame({"ETFs": tickers_visualizar,"Weight": w_min_mark.round(4)}) # MARKOWITZ
        st.write("Weights:",df_visualizar4 )
        st.session_state.df_marko_4 = df_visualizar4
        #st.write(f"Retorno: {ret_min_mark:.4f}, Volatilidad: {vol_min_mark:.4f}")
        col1_1, col2_2 = st.columns(2)
        col1_1.metric(
            label="Return",
            value=f"{ret_min_mark*100:.2f}%",border=True, width="content")

        col2_2.metric(
            label="Volatility",
            value=f"{vol_min_mark*100:.2f}%",border=True,width="content")
    
    with col2:
        st.subheader("Max Return Portfolio")
        df_visualizar2 = pd.DataFrame({"ETFs": tickers_visualizar,"Weight": w_max_ret.round(4)}) # Max ret
        st.write("Weights:",df_visualizar2 )
        st.session_state.df_max_ret_4 = df_visualizar2
        #st.write(f"Retorno: {ret_max_ret:.4f}, Volatilidad: {vol_max_ret:.4f}")
        col1_1, col2_2 = st.columns(2)
        col1_1.metric(
            label="Return",
            value=f"{ret_max_ret*100:.2f}%",border=True, width="content")
        col2_2.metric(
            label="Volatility",
            value=f"{vol_max_ret*100:.2f}%",border=True,width="content")
    
    st.session_state.pesos_min_var = w_min_vol
    st.session_state.pesos_max_ret = w_max_ret
    st.session_state.pesos_max_shap = w_max_sharpe
    st.session_state.pesos_mark = w_min_mark
    

    

# ----------------------------------------

display_portfolio_analysis()

# -------------------------------------------------------------------------
def analisis_optimized(pesos, df_opt):
    
    st.session_state.df_optimized = df_opt
    
    weights_decimal = pd.DataFrame(pesos, columns=["value"])

    weights_decimal.index = st.session_state.df_optimized["ETFs"]
    
    # Alinear DataFrame con los tickers de los pesos
    df_returns_aligned = df_ticker_cortado_fechas.copy()


    # Mantener 'Date' y solo los tickers que tenemos en weights_decimal
    tickers_in_df = [ticker for ticker in weights_decimal.index if ticker in df_returns_aligned.columns]
    
    df_returns_aligned = df_returns_aligned[["Date"] + tickers_in_df]
    
    # Reindexar pesos para coincidir con columnas del DataFrame
    weights_aligned = weights_decimal.loc[tickers_in_df]
    weights_aligned = weights_aligned["value"].rename("Weights (%)")

    st.session_state.weights_aligned_custom = weights_aligned

    #st.write(f"{type(df_returns_aligned)} - {type(weights_aligned)} - {type(st.session_state.Tasa_libre_de_riesgo_widget)})")
    analyzer_op = PortfolioAnalyzer(df_returns_aligned, weights_aligned, st.session_state.Tasa_libre_de_riesgo_widget)

    metrics = analyzer_op.analyze()

    st.session_state.metrics = metrics
    Metricas(metrics)

    return analyzer_op
# -------------------------------------------------------------------------

# ===========================    BOTONES =====================================



# ===== SELECTOR DE MÉTODO DE OPTIMIZACIÓN =====

opciones = {
    "Minimize Variance (σ)": ("Varianza", st.session_state.pesos_min_var, st.session_state.df_min_var),
    "Maximize Return (r_max)": ("Max Return", st.session_state.pesos_max_ret, st.session_state.df_max_ret_4),
    "Maximize Sharpe (S_max)": ("Sharp", st.session_state.pesos_max_shap, st.session_state.df_max_sharp_3),
    "Markowitz Optimization": ("Marko", st.session_state.pesos_mark, st.session_state.df_marko_4),
}

st.write("# Select Optimization:")

# Selector
seleccion = st.selectbox(
    "",
    list(opciones.keys())
)

# Guardamos el tipo en session_state
st.session_state.camino = opciones[seleccion][0]

# Extraemos los pesos correspondientes
pesos_elegidos = opciones[seleccion][1]
df_elegido    = opciones[seleccion][2]

# Ejecutamos análisis
#analisis_optimized(pesos_elegidos)

analyzer_op = analisis_optimized(pesos_elegidos, df_elegido)

if analyzer_op:

    # CÁLCULO DE MÉTRICAS CUSTOM
    metrics = st.session_state.metrics

    # ================= Benchmark =================
    # Benchmark según estrategia
    st.subheader("Comparizon :violet[Optimized] vs :gray[Benchmark] ")

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
    df_bench_aligned = df_Todos_cortado_fechas[["Date"] + tickers_bench]
    weights_bench_aligned = pd.Series({t: benchmark_weights[t] for t in tickers_bench})

    benchmark_analyzer = PortfolioAnalyzer(
        df_bench_aligned,
        weights_bench_aligned,
        st.session_state.Tasa_libre_de_riesgo_widget,
        analyzer_op.portfolio_returns
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

    cum_custom = (1 + analyzer_op.portfolio_returns).cumprod()
    cum_bench = (1 + benchmark_analyzer.portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(cum_custom, label="Custom", linewidth=2.5, color="#bf40bf")
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
    st.write("### :violet[Optimized] Portfolio Drawdown & Return Visualization")
    analyzer_op.plot_portfolio_analysis()
    
    st.write(f"**Benchmark weights:** {weights_bench_aligned}")
    st.write(f"**Custom weights:** {st.session_state.weights_aligned_custom}")


Heat_map()


# =====================================================
# BLACK–LITTERMAN SECTION
# =====================================================

st.write("---")
st.write("# Black-Litterman Optimization")
st.caption(
    "Incorporate subjective market views into portfolio optimization. "
    "All inputs are annual. Ω is diagonal. Risk-free rate is ignored.")

returns_bl = st.session_state.retornos_sin_fecha_session
assets = list(returns_bl.columns)
n_assets = len(assets)

# =====================
# Parámetros BL
# =====================
col1, col2, col3 = st.columns([1,1,1])
with col1:
    with st.container(border=True):
        col1_1, col2_2 = st.columns(2)
        with col1_1:
            st.write("### Tau (uncertainty)")
        with col2_2:
            tau = st.slider("", 0.01, 0.1, 0.025, step=0.005)
with col2:
    with st.container(border=True):
        col1_1, col2_2 = st.columns(2)
        with col1_1:
            st.write("### Risk Aversion (λ)")
        with col2_2:
            lam = st.slider("", 0.5, 5.0, 3.0, step=0.1)

with col3:
    with st.container(border=True):
        col1_1, col2_2 = st.columns(2)
        with col1_1:
            st.markdown("### Market Views")
        with col2_2:
            n_views = st.number_input("Number of Views", 1, 5, 1)

# =====================
# Views
# =====================

P = np.zeros((n_views, n_assets))
Q = np.zeros(n_views)

invalid_views = False

for i in range(n_views):
        
    st.markdown(f"### View {i+1}")

    col1, col2, col3 = st.columns(3)

    # -------- Columna 1: tipo de view
    with col1:
        view_type = st.selectbox(
            "View type",
            ["Absolute", "Relative"],
            key=f"view_type_{i}")

    # -------- Columna 2: activos
    with col2:
        if view_type == "Absolute":
            asset = st.selectbox("Asset",assets,key=f"abs_asset_{i}")
            
        else:
            col1_1, col2_2 = st.columns(2)
            with col1_1:
                a1 = st.selectbox( "Outperform", assets,key=f"a1_{i}")
                
            with col2_2:
                a2 = st.selectbox("Underperform",assets,key=f"a2_{i}")
                
            if a1 == a2:
                st.warning("⚠️ Select :red[different] ticker for Relative comparizon. ⚠️")
                invalid_views = True            # Flag


    # -------- Columna 3: magnitud de la view
    with col3:
        if view_type == "Absolute":
            q = st.number_input("Expected return (annual %)",value=5.0,step=0.5,key=f"q_abs_{i}") / 100

            P[i, assets.index(asset)] = 1
            Q[i] = q

        else:
            q = st.number_input("Outperformance (annual %)",value=2.0,step=0.5,key=f"q_rel_{i}") / 100

            P[i, assets.index(a1)] = 1
            P[i, assets.index(a2)] = -1
            Q[i] = q
            
#st.write(f"P ---- {P}")
w_m = np.ones(n_assets) / n_assets # Prior weights (equilibrium)

# =====================
# Ejecutar BL
# =====================
if st.button("🔥 :green[Run Black-Litterman] 🔥"):
    if invalid_views == True:
        st.warning("⚠️ Fix views before running. ⚠️")
        st.stop()
    
    w_bl, post_ret = black_litterman_portfolio(
        df_returns=returns_bl,
        tau=tau,
        risk_free_rate=None,
        P=P,
        Q=Q,
        lam=lam,
        w_m=w_m,
        sum_constraint=True
    )
    #st.write(returns_bl)
    #st.write(assets)

    st.session_state.w_bl = w_bl
    st.session_state.assets_bl = assets

    # ========= Mostramos pesos del portafolio =========
    st.markdown("### Black-Litterman Optimal Weights (Ordered by weight)")

    df_bl = pd.DataFrame({
        "Asset": assets,
        "Weight": w_bl
    }).sort_values("Weight", ascending=False)

    st.dataframe(
        df_bl.style.format({"Weight": "{:.2%}"}),
        use_container_width=True
    )

# =====================================================
# BLACK–LITTERMAN vs OPTIMIZED
# =====================================================

def normalize_metric(name: str) -> str:
## Normaliza el nombre de la métrica.
    return name.lower().strip()

## Definir listas de métricas

HIGHER_IS_BETTER = [ ## Métricas donde un valor más alto es mejor
    "return",
    "mean",
    "sharpe",
    "sortino",
    "calmar",
    "skew"
]

LOWER_IS_BETTER = [ ## Métricas donde un valor más bajo es mejor
    "volatility",
    "std",
    "drawdown",
    "var",
    "cvar"
]

ABS_CLOSE_TO_ZERO = [ "kurtosis"] ## Métricas donde el valor cercano a cero es mejor

if "w_bl" in st.session_state:

    st.subheader(
        f"Comparizon :green[Black-Litterman] vs "
        f":violet[Optimized – {seleccion}]"
    )

    # ---------- Construir DataFrame de BL ----------
    df_bl_returns = df_ticker_cortado_fechas.copy()

    tickers_bl = [
        t for t in st.session_state.assets_bl
        if t in df_bl_returns.columns
    ]

    df_bl_returns = df_bl_returns[["Date"] + tickers_bl]

    weights_bl = pd.Series(
        st.session_state.w_bl,
        index=st.session_state.assets_bl
    ).loc[tickers_bl]

    # ---------- Analyzer Black–Litterman ----------
    analyzer_bl = PortfolioAnalyzer(
        df_bl_returns,
        weights_bl,
        st.session_state.Tasa_libre_de_riesgo_widget,
        analyzer_op.portfolio_returns  # Mismo periodo para comparación justa
    )

    metrics_bl = analyzer_bl.analyze()

    # ---------- Tabla comparativa ----------
    df_compare_bl = pd.DataFrame({
        "Optimized": st.session_state.metrics,
        "Black-Litterman": metrics_bl
    }).round(4)

    def compare_bl_vs_opt(row):
        metric_raw = row.name
        metric = normalize_metric(metric_raw)

        opt = row["Optimized"]
        bl  = row["Black-Litterman"]

        # Evitar errores si hay NaN
        if pd.isna(opt) or pd.isna(bl):
            return "⬜"

        if any(k in metric for k in HIGHER_IS_BETTER):
            return "🟩 Black-Litterman" if bl > opt else "🟪 Optimized"

        if any(k in metric for k in LOWER_IS_BETTER):
            return "🟩 Black-Litterman" if bl < opt else "🟪 Optimized"
        
        if any(k in metric for k in ABS_CLOSE_TO_ZERO):
            return "🟩 Black-Litterman" if abs(bl) < abs(opt) else "🟪 Optimized"
            
        return "⬜"

    df_compare_bl["Winner"] = df_compare_bl.apply(compare_bl_vs_opt, axis=1)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.dataframe(df_compare_bl, use_container_width=True)

    # ---------- Gráfica acumulada ----------
    cum_opt = (1 + analyzer_op.portfolio_returns).cumprod()
    cum_bl  = (1 + analyzer_bl.portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.plot(
        cum_opt,
        label=f"Optimized – {seleccion}",
        linewidth=2.5,
        color="#bf40bf"
    )

    ax.plot(
        cum_bl,
        label="Black-Litterman",
        linewidth=2.5,
        color="#3cb371"
    )

    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#c9d1d9")

    ax.set_facecolor("#22304A")
    fig.patch.set_facecolor("#22304A")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(facecolor="#22304A", labelcolor="white")

    with col2:
        st.pyplot(fig)

    # ---------- Visualizaciones ----------
    st.write("### :green[Black-Litterman] Portfolio Visualization")
    analyzer_bl.plot_portfolio_analysis()

    # ================= PESOS CONSOLIDADOS =================

    st.write("### Final Portfolio Weights Comparison")

    # Optimized weights (Series)
    w_opt = st.session_state.weights_aligned_custom.copy()
    w_opt.name = "Optimized"

    # Black-Litterman weights (Series)
    w_bl_named = weights_bl.copy()
    w_bl_named.name = "Black-Litterman"

    # Unir en un solo DataFrame
    df_weights_compare = pd.concat([w_opt, w_bl_named], axis=1).fillna(0)

    # Formato en %
    df_weights_compare = df_weights_compare.applymap(lambda x: f"{x*100:.2f}%")

    st.dataframe(
        df_weights_compare,
        use_container_width=True
    )