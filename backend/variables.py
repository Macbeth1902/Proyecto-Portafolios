"""
VARIABLES.PY

Guarda listas y dataframes para no hacer 
un relajo en los demás archivos
"""


ETF_regiones = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]
ETF_sectores = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

ETF_regions_desc = {
    "SPLG": "🇺🇸 S&P 500 Index; exposure to the 500 largest U.S. companies.",
    "EWC": "🇨🇦 Canadian equity market; heavy exposure to energy, materials, and financials.",
    "IEUR": "🇪🇺 Developed European equities; includes companies from the UK, Germany, France, Switzerland, and others.",
    "EEM": "🌏 Emerging markets; large exposure to China, Taiwan, India, Brazil, and South Korea.",
    "EWJ": "🇯🇵 Japanese equity market; exposure to industrials, technology, and manufacturing sectors."
}
ETF_sectors_desc = {
    "XLC": "📡 Communication Services; telecom, media, entertainment, and digital platforms.",
    "XLY": "🛍️ Consumer Discretionary; retail, automobiles, apparel, and non-essential goods.",
    "XLP": "🥫 Consumer Staples; food, beverages, household essentials, and personal care.",
    "XLE": "⚡ Energy; oil, gas, exploration, and energy services companies.",
    "XLF": "💰 Financials; banks, insurance, capital markets, and financial services.",
    "XLV": "🩺 Health Care; pharmaceuticals, biotechnology, medical equipment, and health services.",
    "XLI": "🏭 Industrials; aerospace, transportation, machinery, and industrial services.",
    "XLB": "🧱 Materials; chemicals, metals, mining, construction materials, and packaging.",
    "XLRE": "🏢 Real Estate (REITs); commercial, residential, and industrial real estate.",
    "XLK": "💻 Technology; software, hardware, semiconductors, and IT services.",
    "XLU": "🔌 Utilities; electricity, gas, water, and regulated utilities."
}

horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y"
    }

# Métricas donde un valor MÁS ALTO es mejor
higher_is_better = [
    "Mean (Annual)",
    "Sharpe",
    "Sortino",
    "Calmar Ratio",
    "Skewness"
]

# Métricas donde un valor MÁS BAJO es mejor
lower_is_better = [
    "Volatility (Annual)",
    "Max Drawdown",
    "VaR 95%",
    "CVaR 95%",
    "Kurtosis"  
]



