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



def black_litterman_portfolio(df_returns, tau, risk_free_rate, P, Q, lam,w_m ,sum_constraint=True):
    """
    Calcula la asignación óptima de pesos usando Black-Litterman.
    
    Parámetros:(Filas, Columnas)
    ----------------------------------------------
    df_returns : pandas dataframe con (n_returns, n_assets)
        Rendimientos históricos diarios de los activos
        
    tau : float
        Parámetro de escala de la incertidumbre (generalmente entre 0.01 y 0.05)
        Pequeño -> Poca Incertidumbre
        Grande -> Mucha Incertidumbre
        preset = 0.025
        
    risk_free_rate : float  (generalmente entre 0.01 y 0.20) Anualizada
        Tasa libre de riesgo diaria
        preset = 0.10
        
    P : array-like (n_views, n_assets)
        Matriz de selección de views
        2 tipos
        
    Q : array-like (n_views,)
        Vector de valores esperados de las views
        (column vector)
        
    lam : float (generalmente entre 0.5 y 5.0) preset = 3.0
        Coeficiente de aversión al riesgo
        
    sum_constraint : bool
        Si True, los pesos suman 1; si False, no hay restricción de suma
        
    ----- Output: -----
    
    w_optimal : numpy array
        Pesos óptimos del portafolio
        
    posterior_returns : numpy array
        Retornos esperados posteriores de Black-Litterman
    """
    
    if "Date" in df_returns.columns:
        df_returns = df_returns.drop(columns=["Date"])

    df_returns = df_returns.apply(pd.to_numeric, errors='coerce')
    df_returns = df_returns.dropna(axis=1, how="all")  # quitar columnas vacías


    # Detectar filas con strings, None o NaN
    bad_rows = (
        df_returns.isna().any(axis=1) |
        df_returns.applymap(lambda x: isinstance(x, str)).any(axis=1))

    st.write("# Filas sin datos:")
    st.write(df_returns[bad_rows])

    # Eliminar filas malas
    df_returns = df_returns[~bad_rows]

    # Convertir limpio a numpy
    returns = df_returns.to_numpy(dtype=float)


    # convertir a numpy (n_assets, n_days)
    st.write("# Dataframe Head y Tail:")
    st.write(df_returns.head(5))
    st.write(df_returns.tail(5))
    #st.write(df_returns.dtypes)

    n_assets = returns.shape[1]
    st.write(n_assets)
    
    # Asegurar que returns sea un array numpy
    st.write(f"# Returns numpy: ")
    st.write(returns)
    st.write(returns.dtype)
    
    st.write(f"Retunrs Shape: {returns.shape}")
    st.write(f"df Retunrs Shape: {df_returns.shape}")

    # ===============================================================================================
    # 1. Reverse Optimization --- Calcular retornos medios históricos (Π) y matriz de covarianza (Σ)
    
    # Calcular matriz de covarianza Σ
    Sigma = np.cov(returns, rowvar=False)
    st.write(f"# 1.1. SIGMA , sigma shape: {Sigma.shape} ")
    st.write(Sigma)
    
    # Calcular Π = λΣw_M
    Pi = lam * (Sigma @ w_m)
    st.write(f"# 1.2. PI , pi shape: {Pi.shape} , vector columna")
    st.write(Pi)
    
    # ===============================================================================================
    # 2. Calcular Ω (matriz de covarianza de las views)
    # Ω = P * (τΣ) * P^T
    Omega = P @ (tau * Sigma) @ P.T
    # Asegurar que Ω sea invertible (añadir pequeña diagonal si es necesario)
    if np.linalg.matrix_rank(Omega) < Omega.shape[0]:
        Omega += np.eye(Omega.shape[0]) * 1e-6
    Omega = np.diag(np.diag(Omega))
    
    st.write(f"# 2. OMEGA , omega shape: {Omega.shape} una una fila por view y una columna por view.")
    st.write("Ω = P * (τΣ) * P^T")
    st.write(Omega)
    
    # ===============================================================================================
    # 3. Calcular el inverso de (τΣ)
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    st.write(f"# 3. Tau sigma Inverso , shape: {tau_Sigma_inv.shape}.")
    st.write(tau_Sigma_inv)
    
    # ===============================================================================================
    # 4. Calcular retornos posteriores esperados (fórmula de Black-Litterman)
    # R_posterior = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1Π + P'Ω^-1Q]
    st.write("# 4. Black Litterman Retorno posterior:")
    st.write("### R_post = [(τΣ)^-1 + P'Ω^-:red[1P]]^-1 * [(τΣ)^-1:red[Π] + P'Ω^-:red[1Q]]")
    P_T = P.T
    
    # Primera parte: [(τΣ)^-1 + P'Ω^-1P]
    Omega_inv = np.linalg.inv(Omega)
    first_part = tau_Sigma_inv + P_T @ Omega_inv @ P
    
    st.write(f"### 4.1. [(τΣ)^-1 + P'Ω^-:red[1P]] , shape: {first_part.shape}.")
    st.write(first_part)
    
    # Segunda parte: [(τΣ)^-1Π + P'Ω^-1Q]
    second_part = tau_Sigma_inv @ Pi + P_T @ Omega_inv @ Q
    st.write(f"### 4.2. [(τΣ)^-1:red[Π] + P'Ω^-:red[1Q]] , shape: {second_part.shape}.")
    st.write(second_part)
    
    # Calcular retornos posteriores
    posterior_returns = np.linalg.inv(first_part) @ second_part
    st.write("# 4.3 Retorno posterior: 4.1^-1 * 4.2")
    st.write(posterior_returns)
    
    
    # ===============================================================================================
    # 5. Calcular asignación óptima de activos
    if not sum_constraint:
        # Sin restricción de suma (fórmula 42)
        w_optimal = (1/lam) * np.linalg.inv(Sigma) @ posterior_returns
        st.write("# 5. w_optimal ")
        st.write(w_optimal)
        
    else:
        # Con restricción de que los pesos sumen 1
        # Usamos optimización como en el ejemplo de Markowitz
        # Función objetivo: maximizar utility = w'*R - (λ/2)*w'Σw
        # ===============================================================================================
        def objective(w):
            utility = w @ posterior_returns - (lam/2) * (w @ Sigma @ w.T)
            return -utility  # Negativo porque minimize() minimiza
        
        # Condiciones iniciales (pesos iguales)
        x0 = np.ones(n_assets) / n_assets
        st.write(f"### Estado 0: ")
        st.write(x0)
        
        # Restricciones
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        st.write(f"### Constraints")
        st.write(constraints)
        
        # Límites (sin ventas cortas)
        bounds = tuple((0, 1) for _ in range(n_assets))
        # Optimización
        result = op.minimize(objective, x0, constraints=constraints, 
                         bounds=bounds, method='trust-constr',tol=1e-25)
        
        w_optimal = result.x
        st.write("# Optimal: Black Litterman Output ")
        st.write("#### Pesos finales del portafolio que maximizan la utilidad del inversionista, dados los retornos posteriores. ")
        st.write(w_optimal)
        
        st.write("# Posterior Results: ")
        st.write("#### Retornos esperados “posteriores” de cada activo después de incorporar tus views. ")
        st.write(posterior_returns)
        
        
    
    return w_optimal, posterior_returns
    

