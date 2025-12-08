import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import streamlit as st

class PortfolioAnalyzer:

    def __init__(self, returns_df, weights, risk_free_rate=0, benchmark_returns=None):
        # Convertir índice
        if "Date" in returns_df.columns:
            returns_df = returns_df.set_index("Date")

        # Eliminar filas con NaNs
        returns_df = returns_df[weights.index].dropna()

        self.returns_df = returns_df
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = 252 # Asumiendo datos diarios: AÑO BURSÁTIL

        self.portfolio_returns = self._calculate_portfolio_returns() # Retornos del portafolio
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None

    def _calculate_portfolio_returns(self):
        return self.returns_df.dot(self.weights)

    # ================= Métricas =================
    def calculate_mean(self):
        # """Media de retornos del portafolio"""
        return self.portfolio_returns.mean() * self.periods_per_year

    def calculate_volatility(self):
        # """Volatilidad (desviación estándar) del portafolio"""
        return self.portfolio_returns.std() * np.sqrt(self.periods_per_year)

    def calculate_max_drawdown(self):
        # """Maximum Drawdown (MDD) del portafolio"""
        cum = (1 + self.portfolio_returns).cumprod()
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        return abs(dd.min())

    def calculate_var(self, conf=0.95):
        # """Value at Risk (VaR) del portafolio"""
        return -np.percentile(self.portfolio_returns, (1 - conf) * 100)

    def calculate_cvar(self, conf=0.95):
        # """Conditional Value at Risk (CVaR) del portafolio"""
        var = self.calculate_var(conf)
        tail = self.portfolio_returns[self.portfolio_returns <= -var]
        return -tail.mean() if len(tail) > 0 else var

    def calculate_sharpe(self):
        # """Sharpe Ratio del portafolio"""
        excess = self.portfolio_returns - self.risk_free_rate / self.periods_per_year
        if excess.std() == 0:
            return np.nan
        return np.sqrt(self.periods_per_year) * excess.mean() / excess.std()

    def calculate_sortino(self):
        # """Sortino Ratio del portafolio"""
        downside = self.portfolio_returns[self.portfolio_returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.nan
        excess = self.portfolio_returns - self.risk_free_rate / self.periods_per_year
        return np.sqrt(self.periods_per_year) * excess.mean() / downside.std()

    def calculate_skewness(self):
        # """Sesgo de la distribución de retornos"""
        return skew(self.portfolio_returns)

    def calculate_kurtosis(self):
        #"""Curtosis de la distribución de retornos"""
        return kurtosis(self.portfolio_returns, fisher=True)

    def calculate_calmar_ratio(self):
        # """Calmar Ratio del portafolio"""
        mdd = self.calculate_max_drawdown()
        return self.calculate_mean() / mdd if mdd != 0 else np.nan

    def analyze(self):
        return {
            "Mean (Annual)": self.calculate_mean(),
            "Volatility (Annual)": self.calculate_volatility(),
            "Max Drawdown": self.calculate_max_drawdown(),
            "VaR 95%": self.calculate_var(),
            "CVaR 95%": self.calculate_cvar(),
            "Sharpe": self.calculate_sharpe(),
            "Sortino": self.calculate_sortino(),
            "Skewness": self.calculate_skewness(),
            "Kurtosis": self.calculate_kurtosis(),
            "Calmar Ratio": self.calculate_calmar_ratio()
        }

    def plot_portfolio_analysis(self):
        DARK_BG = "#0d1117"
        FG = "#c9d1d9"
        ACCENT1 = "#58a6ff"
        ACCENT2 = "#3fb950"

        # ================= Cumulative Return =================
        # Gráfica de retorno acumulado: Evolución del portafolio
        cum = (1 + self.portfolio_returns).cumprod()

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(cum, linewidth=2, label="Cumulative Return", color=ACCENT1)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=FG)
        ax.set_title("Cumulative Return", color=FG)
        ax.legend(facecolor=DARK_BG, labelcolor=FG)
        st.pyplot(fig)

        # ================= Drawdown =================
        cummax = cum.cummax()
        dd = (cum - cummax) / cummax

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.fill_between(dd.index, dd, color="#d73a49")
        fig.patch.set_facecolor(DARK_BG)
        ax.set_title("Drawdown", color=FG)
        ax.tick_params(colors=FG)
        st.pyplot(fig)

        # ================= Histogram =================
        # Histograma con la distribución de retornos diarios
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(self.portfolio_returns, bins=40, alpha=0.75, color=ACCENT2)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_title("Daily Return Distribution", color=FG)
        ax.tick_params(colors=FG)
        st.pyplot(fig)