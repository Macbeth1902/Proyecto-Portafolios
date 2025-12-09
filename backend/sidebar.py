import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import backend.variables as va
from backend.metricas import Metricas
from backend.data import load_returns,load_ETF_bechmark,cargar_cortes_fechas,load_data_bounds


def init_state():
    if "tickers_input" not in st.session_state:
        st.session_state.tickers_input = va.ETF_regiones[:]   # copia por defecto
        
"""
SIDERBAR.PY

Genera los widgets , elementos visuales y outputs
de datos del Sidebar
"""


def show_sidebar():
    with st.sidebar:
        init_state()

        col1,col2,col3 = st.columns([0.5,2,0.7])
        with col2:
            st.title("⚙️ Settings")
    
        #  ============================== (1) Estrategia Inversion =============================
        st.selectbox(
            "Investment strategy:",
            ["by Regions", "by Sectors"],
            key="opcion_2")
        
        #  ============================== (2) Ticker Input  =============================
        DEFAULT_ETFS = va.ETF_regiones   

        def etfs_to_str(tickers):
            return ",".join(tickers)

        if "tickers_input" not in st.session_state:
            st.session_state.tickers_input = st.query_params.get(
                "etfs", etfs_to_str(DEFAULT_ETFS)
            ).split(",")

        def update_query_param():
            if st.session_state.tickers_input:
                st.query_params["etfs"] = etfs_to_str(st.session_state.tickers_input)
            else:
                st.query_params.pop("etfs", None)


        def ETF_select():
            if st.session_state.opcion_2 == "by Regions":
                return va.ETF_regiones
            else:
                return va.ETF_sectores

        tickers = st.multiselect(
            "💼 Portfolio Composition:",
            options=sorted(set(ETF_select()) | set(st.session_state.tickers_input)),
            placeholder="Choose ETFs to compare. Example: XLK",
            accept_new_options=True,
            key="tickers_input",
            on_change=update_query_param,)

        #  ============================== (3) Horizonte de Inversion ============================= QUITAR ???
        #horizon = st.pills(
        #    "Time horizon",
        #    options=list(va.horizon_map.keys()),
        #    default="6 Months",key="horizon"
        #)

        
        # ======== (2) UPDATE TICKERS SELECCIONADOS ======================
         # Update query param when text input changes
        tickers = [t.upper() for t in tickers]
        if tickers:
            st.query_params["stocks"] = etfs_to_str(tickers)
        else:
            # Clear the param if input is empty
            st.query_params.pop("stocks", None)
            
 

        # ============================== (4) Botones Load Data =============================
        col1, col2 = st.columns(2)
        with col1:
            boton_aceptar = st.button(":green[Load data]")

        with col2:
            boton_cancelar = st.button(":red[Cancelar]")


        # 4.1) Load Data
        if boton_aceptar:
            
            #if horizon == None: 
            #    st.sidebar.write(" :warning: **:red[Add time horizon]** :warning: ")
                #st.stop()
            if len(tickers) == 0: # NO HAY ETFs
                st.sidebar.write(" :warning: **:red[Add tickers]** :warning: ")
                
            if len(tickers) != 0:
                st.session_state.df_tickers = load_returns(st.session_state.tickers_input)
                st.session_state.df_ETF_bechmark = load_ETF_bechmark(st.session_state.opcion_2)
            
                
                st.session_state.fecha_inicio_actual = load_data_bounds(st.session_state.fecha_inicio)
                st.session_state.fecha_fin_actual = load_data_bounds(st.session_state.fecha_corte)
            
            if "df_todos" not in st.session_state:
                st.session_state.df_todos = load_returns([
                "SPLG", "EWC", "IEUR", "EEM", "EWJ",
                "XLC", "XLY", "XLP", "XLE", "XLF",
                "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"])
   
                with col1:
                    st.success("✅ Data loaded")
                
                
        # 4.2) CANCELAR     
        if boton_cancelar:
            pass
        
        st.markdown("---")

        # ============================== (5) Fecha Slicer =============================
        
        #  5.1. Cargar DataFrame solo una vez
        if "df_cortes_fechas" not in st.session_state:
            st.session_state.df_cortes_fechas = cargar_cortes_fechas()

        df_fechas_sidebar = st.session_state.df_cortes_fechas
        
        fechas_disponibles = df_fechas_sidebar["Date"].sort_values().unique()
        st.session_state.fechas_disponibles = fechas_disponibles

            
        # Inicializar solo si no existen
        if "fecha_inicio" not in st.session_state:
            st.session_state.fecha_inicio = fechas_disponibles[0]

        if "fecha_corte" not in st.session_state:
            st.session_state.fecha_corte = fechas_disponibles[-1]



        #  5.2. Widgets que PERSISTEN

        fechas_disponibles = [pd.to_datetime(f).date() for f in st.session_state.fechas_disponibles]

        # Aseguramos que el estado también tenga tipo date ================================================= VERIFICAR QUE SE SIGA MANEJANDO DATETIME
        if isinstance(st.session_state.fecha_inicio, (str, np.datetime64, pd.Timestamp)):
            st.session_state.fecha_inicio = pd.to_datetime(st.session_state.fecha_inicio).date()

        if isinstance(st.session_state.fecha_corte, (str, np.datetime64, pd.Timestamp)):
            st.session_state.fecha_corte = pd.to_datetime(st.session_state.fecha_corte).date()

        col1, col2 = st.columns(2)
        with col1:
        # =========================
        #  WIDGETS DATE_INPUT
        # =========================
            fecha_inicio = st.date_input(
                "Start date:",
                value=st.session_state.fecha_inicio,
                min_value=min(fechas_disponibles),
                max_value=max(fechas_disponibles),
                key="fecha_inicio_widget"   # Persistencia automática
            )
        with col2:
            fecha_corte = st.date_input(
                "End date:",
                value=st.session_state.fecha_corte,
                min_value=min(fechas_disponibles),
                max_value=max(fechas_disponibles),
                key="fecha_corte_widget"    # Persistencia automática
            )
        
         # ============================== (6) SLICERS RETORNO Y TASA LIBRE DE RIESGO =============================(hay que pasarlo a porcentaje)
        
        if "Rendimiento_objetivo" not in st.session_state:
            st.session_state.Rendimiento_objetivo = 10   # valor inicial (10%)

        if "Tasa_libre_de_riesgo" not in st.session_state:
            st.session_state.Tasa_libre_de_riesgo = 5   # valor inicial (5%)
                
        
        col1,col2 =st.columns(2)
        with col1:
            
            slider_50 = st.slider(
                "Objective Return",
                min_value=5.0,
                max_value=20.0,
                value=10.0,      
                step=0.1,           # valor inicial
                format="%.2f%%",  # muestra como %
                key="Rendimiento_objetivo_widget"
            )
        with col2:
            # Slider de 0% a 30%
            slider_30 = st.slider(
                "Risk Free Rate",
                min_value=0.0,
                max_value=30.0,
                value=5.0,      # valor inicial
                step=0.1,
                format="%.2f%%",
                key="Tasa_libre_de_riesgo_widget"
            )
            
        #st.write(st.session_state.Rendimiento_objetivo_widget)
        #st.write(st.session_state.Tasa_libre_de_riesgo_widget)


