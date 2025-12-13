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
def analisis_optimized(pesos):
    
    st.session_state.df_optimized = st.session_state.df_min_var
    
    
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
    "Minimize Variance (σ)": ("Varianza", st.session_state.pesos_min_var),
    "Maximize Return (r_max)": ("Max Return", st.session_state.pesos_max_ret),
    "Maximize Sharpe (S_max)": ("Sharp", st.session_state.pesos_max_shap),
    "Markowitz Optimization": ("Marko", st.session_state.pesos_mark),
}
st.write("# Select Optimization:")

# Selector
seleccion = st.selectbox(
    "📊",
    list(opciones.keys())
)

# Guardamos el tipo en session_state
st.session_state.camino = opciones[seleccion][0]

# Extraemos los pesos correspondientes
pesos_elegidos = opciones[seleccion][1]

# Ejecutamos análisis
#analisis_optimized(pesos_elegidos)

analyzer_op = analisis_optimized(pesos_elegidos)

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
    df_bench_aligned = df_bench_aligned = df_Todos_cortado_fechas[["Date"] + tickers_bench]
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








""
""
st.write("# :green[=================== DEBUG ==================]")
st.write("### ---------------------------------------- INPUTS ----------------------------------------")
st.write(f"**Custom weights** {st.session_state.weights_aligned_custom}")

st.write("##### Dataframe con retunrs")
st.write(df_ticker_cortado_fechas.head(5))

trading_days = 252

Tau_widget = 0.025
risk_free_rate_widget = 0.10/trading_days  # 10% anual en retorno diario
Lambda_widget = 3.0 

# Test con 4 activos EEM EWC EWJ IEUR

# Orden de assets: ['EEM','EWC','EWJ','IEUR','splg´]
P_widget = np.array([
    [ 1,  0,  0,  0, 0],   # View 1: EEM = Q[0]
    [ 0,  1,  0, -1, 0]    # View 2: EWC - IEUR = Q[1]
])

Q_widget = np.array([ 0.02 , 0.06 ])
w_m_widget = np.array([0.2,0.2,0.2,0.2,0.2])

st.write(f"**P**:  {P_widget} , forma: {P_widget.shape}")
st.write(f"**Q**:  {Q_widget} , forma: {Q_widget.shape}")

st.write(f"Tau:                 {Tau_widget}.")
st.write(f"Risk free rate:      {risk_free_rate_widget}.")
st.write(f"Lambda:               {Lambda_widget}.")




st.write("### ---------------------------------------- BLACK LITTERMAN OUTPUT ----------------------------------------")
#  black_litterman_portfolio(df_returns, tau, risk_free_rate, P, Q, lam, w_m ,sum_constraint=True):
black_litterman_portfolio(df_ticker_cortado_fechas,Tau_widget,risk_free_rate_widget,P_widget,Q_widget,Lambda_widget,w_m_widget)


#st.write(list(st.session_state.retornos_sin_fecha_session.columns))
        
#  ===========================    DEBUG   ==================
#st.write(df_ticker_cortado_fechas.head(5))
#st.write(df_ETF_cortado_fechas.head(5))
#st.write("----------------")
#st.write(df_ticker_cortado_fechas.tail(5))
#st.write(df_ETF_cortado_fechas.tail(5))

#st.write(returns.head(5))
#st.write()