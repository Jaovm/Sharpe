import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Otimização de Carteira - Sharpe", layout="wide")
st.title("Otimização de Carteira usando Simulação de Monte Carlo para Melhor Sharpe")

# Entradas do usuário
st.sidebar.header("Parâmetros da Carteira")
tickers = st.sidebar.text_area("Tickers separados por vírgula", value="AGRO3.SA,BBAS3.SA,BBSE3.SA,BPAC11.SA,EGIE3.SA,ITUB3.SA,PRIO3.SA,PSSA3.SA,SAPR3.SA,SBSP3.SA,VIVT3.SA,WEGE3.SA,TOTS3.SA,B3SA3.SA,TAEE3.SA")
tickers = [t.strip().upper() for t in tickers.split(",")]

anos = st.sidebar.slider("Período (anos)", 1, 10, 7)

st.sidebar.subheader("Alocação Inicial e Restrições")
pesos_iniciais = []
limites_min = []
limites_max = []

for ticker in tickers:
    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    with col1:
        peso = st.number_input(f"{ticker} (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key=f"peso_{ticker}") / 100
    with col2:
        min_val = st.number_input(f"Min {ticker} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"min_{ticker}") / 100
    with col3:
        max_val = st.number_input(f"Max {ticker} (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key=f"max_{ticker}")
        max_val /= 100
    pesos_iniciais.append(peso)
    limites_min.append(min_val)
    limites_max.append(max_val)

# Baixar dados de preços
raw_data = yf.download(tickers, period=f"{anos}y", group_by="ticker")
dados = pd.DataFrame({ticker: raw_data[ticker]['Adj Close'] for ticker in tickers})
dados = dados.dropna()

# Retornos esperados e matriz de covariância
mu = expected_returns.mean_historical_return(dados)
S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

# Simulação de Monte Carlo
def simulate_portfolios(num_portfolios=10000):
    results = []
    for _ in range(num_portfolios):
        pesos = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()
        pesos = np.clip(pesos, limites_min, limites_max)
        pesos /= pesos.sum()

        retorno_esp = np.dot(pesos, mu)
        risco = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
        sharpe = retorno_esp / risco

        results.append((sharpe, retorno_esp, risco, pesos))

    return sorted(results, key=lambda x: x[0], reverse=True)[0]  # Melhor Sharpe

best_sharpe, best_return, best_risk, best_weights = simulate_portfolios()

# Resultados
st.subheader("Melhor Carteira - Índice de Sharpe Máximo")
st.markdown(f"**Sharpe:** {best_sharpe:.2f} | **Retorno Esperado:** {best_return:.2%} | **Risco (Vol):** {best_risk:.2%}")

result_df = pd.DataFrame({
    'Ticker': tickers,
    'Peso (%)': np.round(best_weights * 100, 2)
})
st.dataframe(result_df.set_index('Ticker'))

# Gráfico de pizza
fig, ax = plt.subplots()
ax.pie(best_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)
