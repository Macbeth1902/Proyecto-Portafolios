import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import backend.variables as va


"""
METRICAS.PY

Formato visual HTML de la sección de
reportar métricas.
"""

def Metricas(metrics_custom: dict):
    st.markdown("""
    <div style='background:#0d1117; padding:20px; border-radius:12px;'>
        <h2 style='color:#58a6ff; margin-top:0;'>📈 Portfolio Metrics</h2>
        <p style='color:#c9d1d9; text-align: justify;'>
            Overview of the performance and risk profile of your custom portfolio.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)

    for (metric, value), col in zip(metrics_custom.items(), cols * 4):
        col.markdown(f"""
        <div style="
            width:100%;
            padding:10px;
            border-radius:12px;
            background:#161b22;
            border:1px solid #30363d;
            text-align:center;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            aspect-ratio:1/1;
            min-height:80px;
            max-height:120px;
        ">
            <h4 style="color:#58a6ff; font-size:20px; margin:0;">{metric}</h4>
            <p style="color:#3fb950; font-size:20px; margin:0;">
                {value:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)