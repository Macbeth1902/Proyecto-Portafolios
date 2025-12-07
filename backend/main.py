import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import variables as va
from sidebar import show_sidebar
from metricas import Metricas
import yfinance as yf


st.set_page_config(
    page_title="Portfolio Allocation Analysis",
    page_icon="Fotos/portafolio.png",
    layout="wide",)

show_sidebar()

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.write("# :material/work: Portfolio Allocation Analysis")
    st.write("##### Easily compare portfolio returns and its risk measures.")
    ""  # Add some space.
    ""  # Add some space.
    st.image("Fotos/Portada.jpg",  use_container_width=True)
    st.write("### 1.- Select your :green[**assets**] and navigate using the menu on the :green[**top left**].")
    st.write("### 2.- Click :green[**Load data**] to update your current dataframe before running any analysis.")
    ""  # Add some space.
    ""  # Add some space.
    ""  # Add some space.
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.write("#### TEAM COMPOSITION")
        st.write("- Coria Arreguin Kevin Aaron")
        st.write("- Cerros García David")
        st.write("- Pérez Hernández Rafael Diego")
        st.write("- Tilihuit Ortigoza Rebeca Macbeth")

tickers = st.session_state.tickers_input
strategy = st.session_state.opcion_2

st.write(str(tickers))
st.write(str(strategy))
