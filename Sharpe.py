import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt import risk_models, expected_returns
from sklearn.covariance import LedoitWolf

# Ações e alocações informadas
acoes = {
    'AGRO3.SA': 10,
    'BBAS3.SA': 1.2,
    'BBSE3.SA': 6.5,
    'BPAC11.SA': 10.6,
    'EGIE3.SA': 5,
    'ITUB3.SA': 0.5,
    'PRIO3.SA': 15,
    'PSSA3.SA': 15,
    'SAPR3.SA': 6.7,
    'SBSP3.SA': 4,
    'VIVT3.SA': 6.4,
    'WEGE3.SA': 15,
    'TOTS3.SA': 1,
    'B3SA3.SA': 0.1,
    'TAEE3.SA': 3
}

# Parâmetros
anos = 7
simulacoes = 1_000_000
tickers = list(acoes.keys())
pesos_informados = np.array(list(acoes.values())) / 100

# Coleta de dados
raw_data = yf.download(tickers, period=f"{anos}y", auto_adjust=True)["Close"]
raw_data = raw_data.dropna()
retornos = raw_data.pct_change().dropna()

# Cálculo estatístico
media_anual = expected_returns.mean_historical_return(raw_data, frequency=252)
cov_anual = risk_models.CovarianceShrinkage(retornos).ledoit_wolf()

# Funções
def simular_monte_carlo(n_sim, retornos_esperados, covariancia, min_bounds, max_bounds):
    num_ativos = len(retornos_esperados)
    resultados = {
        'Retorno': [],
        'Volatilidade': [],
        'Sharpe': [],
        'Pesos': []
    }

    for _ in range(n_sim):
        pesos = np.random.dirichlet(np.ones(num_ativos), 1).flatten()
        if np.all(pesos >= min_bounds) and np.all(pesos <= max_bounds):
            ret_esp = np.dot(pesos, retornos_esperados)
            vol = np.sqrt(np.dot(pesos.T, np.dot(covariancia, pesos)))
            sharpe = ret_esp / vol if vol > 0 else 0
            resultados['Retorno'].append(ret_esp)
            resultados['Volatilidade'].append(vol)
            resultados['Sharpe'].append(sharpe)
            resultados['Pesos'].append(pesos)
    return resultados

# Limites de alocação
min_bounds = np.zeros(len(tickers))
max_bounds = np.ones(len(tickers))

# Simulações
resultados = simular_monte_carlo(simulacoes, media_anual.values, cov_anual.values, min_bounds, max_bounds)
df_resultados = pd.DataFrame(resultados)

# Carteiras ótimas
idx_sharpe = df_resultados['Sharpe'].idxmax()
idx_retorno = df_resultados['Retorno'].idxmax()

pesos_sharpe = df_resultados.loc[idx_sharpe, 'Pesos']
pesos_retorno = df_resultados.loc[idx_retorno, 'Pesos']

# Função de métricas
def calcular_metricas(pesos, retornos_esperados, cov):
    retorno = np.dot(pesos, retornos_esperados)
    volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
    sharpe = retorno / volatilidade if volatilidade > 0 else 0
    cagr = (1 + retorno) ** anos - 1
    return retorno * 100, volatilidade * 100, sharpe, cagr * 100

# Resultados
ret_inf, vol_inf, sharpe_inf, cagr_inf = calcular_metricas(pesos_informados, media_anual.values, cov_anual.values)
ret_sh, vol_sh, sharpe_sh, cagr_sh = calcular_metricas(pesos_sharpe, media_anual.values, cov_anual.values)
ret_ret, vol_ret, sharpe_ret, cagr_ret = calcular_metricas(pesos_retorno, media_anual.values, cov_anual.values)

# Streamlit
st.title("Otimização de Carteira - Ações Brasileiras")

st.subheader("Carteira Informada")
st.dataframe(pd.DataFrame({
    'Ticker': tickers,
    'Alocação (%)': (pesos_informados * 100).round(2)
}))
st.markdown(f"**Retorno Esperado:** {ret_inf:.2f}%  \n"
            f"**Volatilidade Anual:** {vol_inf:.2f}%  \n"
            f"**Sharpe:** {sharpe_inf:.2f}  \n"
            f"**CAGR:** {cagr_inf:.2f}%")

st.subheader("Carteira com Maior Sharpe")
st.dataframe(pd.DataFrame({
    'Ticker': tickers,
    'Alocação (%)': (pesos_sharpe * 100).round(2)
}))
st.markdown(f"**Retorno Esperado:** {ret_sh:.2f}%  \n"
            f"**Volatilidade Anual:** {vol_sh:.2f}%  \n"
            f"**Sharpe:** {sharpe_sh:.2f}  \n"
            f"**CAGR:** {cagr_sh:.2f}%")

st.subheader("Carteira com Maior Retorno Esperado")
st.dataframe(pd.DataFrame({
    'Ticker': tickers,
    'Alocação (%)': (pesos_retorno * 100).round(2)
}))
st.markdown(f"**Retorno Esperado:** {ret_ret:.2f}%  \n"
            f"**Volatilidade Anual:** {vol_ret:.2f}%  \n"
            f"**Sharpe:** {sharpe_ret:.2f}  \n"
            f"**CAGR:** {cagr_ret:.2f}%")
