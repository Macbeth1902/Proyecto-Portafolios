import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import variables as va
from data import load_returns,load_ETF_bechmark


def init_state():
    if "tickers_input" not in st.session_state:
        st.session_state.tickers_input = va.ETF_regiones[:]   # copia por defecto



def show_sidebar():
    with st.sidebar:
        init_state()

        st.title("⚙️ Settings")
    
        st.selectbox(
            "Investment strategy:",
            ["by Regions", "by Sectors"],
            key="opcion_2"
        )
        
        DEFAULT_ETFS = va.ETF_regiones   # -------------------------

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
            on_change=update_query_param,
        )

        # Buttons for picking time horizon
        horizon = st.pills(
            "Time horizon",
            options=list(va.horizon_map.keys()),
            default="6 Months",key="horizon"
        )

        tickers = [t.upper() for t in tickers]
        #--------------------------------------------------
        # Update query param when text input changes
        if tickers:
            st.query_params["stocks"] = etfs_to_str(tickers)
        else:
            # Clear the param if input is empty
            st.query_params.pop("stocks", None)
            
        #--------------------------------------------------


        col1, col2 = st.columns(2)

        with col1:
            boton_aceptar = st.button(":green[Load data]")

        with col2:
            boton_cancelar = st.button(":red[Cancelar]")



        if boton_aceptar:
            
            if horizon == None: 
                st.sidebar.write(" :warning: **:red[Add time horizon]** :warning: ")
                #st.stop()
            if len(tickers) == 0: # NO HAY ETFs
                st.sidebar.write(" :warning: **:red[Add tickers]** :warning: ")
                
            if len(tickers) != 0:
                st.session_state.df_tickers = load_returns(st.session_state.tickers_input)
                st.session_state.df_ETF_bechmark = load_ETF_bechmark(st.session_state.opcion_2)
                with col1:
                    st.success("✅ Data loaded")
                
            # Corregir logica
                
                    
        if boton_cancelar:
            pass
        
        
        
        
        


