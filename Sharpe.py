import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.title("Otimização de Carteira - Teoria Moderna do Portfólio")

# Ações previamente citadas
acoes_dict = {
    "AGRO3.SA": (10, 5, 15),
    "BBAS3.SA": (1.2, 0.5, 5),
    "BBSE3.SA": (6.5, 5, 10),
    "BPAC11.SA": (10.6, 5, 15),
    "EGIE3.SA": (5, 2, 8),
    "ITUB3.SA": (0.5, 0.1, 2),
    "PRIO3.SA": (15, 10, 20),
    "PSSA3.SA": (15, 10, 20),
    "SAPR3.SA": (6.7, 5, 10),
    "SBSP3.SA": (4, 2, 8),
    "VIVT3.SA": (6.4, 5, 10),
    "WEGE3.SA": (15, 10, 20),
    "TOTS3.SA": (1, 0.5, 3),
    "B3SA3.SA": (0.1, 0.05, 2),
    "TAEE3.SA": (3, 2, 6)
}

anos = st.sidebar.slider("Anos de histórico de dados", 1, 10, 7)
simulacoes = 1_000_000

# Preparar listas para inputs personalizados
tickers = list(acoes_dict.keys())
pesos_iniciais = [acoes_dict[t][0] for t in tickers]
minimos = [acoes_dict[t][1]/100 for t in tickers]
maximos = [acoes_dict[t][2]/100 for t in tickers]

# Baixar dados
raw_data = yf.download(tickers, period=f"{anos}y", group_by="ticker", auto_adjust=True, progress=False)

# Verificação e extração dos dados ajustados
precos = pd.DataFrame()
for ticker in tickers:
    try:
        precos[ticker] = raw_data[ticker]["Close"]
    except (KeyError, TypeError):
        st.error(f"Erro ao obter dados para {ticker}. Verifique o ticker ou os dados disponíveis.")

precos.dropna(how="any", inplace=True)
retornos = precos.pct_change().dropna()

# Validação da matriz de covariância
try:
    cov_anual = risk_models.CovarianceShrinkage(retornos).ledoit_wolf()
except ValueError:
    st.error("Erro ao calcular a matriz de covariância. Verifique os dados dos ativos.")
    st.stop()

# Estimativa de retorno esperado
mu = expected_returns.mean_historical_return(precos)

# Otimizador
ef = EfficientFrontier(mu, cov_anual, weight_bounds=(0, 1))
ef.add_objective(lambda w: 0)  # Placeholder para objetivos customizados

# Aplicar restrições e pesos iniciais
pesos_informados = np.array(pesos_iniciais) / 100
ef.set_weights(dict(zip(tickers, pesos_informados)))
ef.efficient_risk(np.std(retornos @ pesos_informados), market_neutral=False)
pesos_informados_ajustados = ef.clean_weights()

# Carteira de maior Sharpe (Monte Carlo)
pesos_random = []
sharpe_scores = []

for _ in range(simulacoes):
    w = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
    if np.all(w >= minimos) and np.all(w <= maximos):
        ret = np.dot(mu, w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_anual, w)))
        sharpe = ret / vol
        pesos_random.append(w)
        sharpe_scores.append(sharpe)

idx_best = np.argmax(sharpe_scores)
pesos_otimizados = pesos_random[idx_best]

# Carteira de maior retorno
retornos_random = [np.dot(mu, w) for w in pesos_random]
idx_max_ret = np.argmax(retornos_random)
pesos_max_ret = pesos_random[idx_max_ret]

# Cálculo dos KPIs
ret_esperado = lambda w: np.dot(mu, w) * 100
vol_esperada = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_anual, w))) * 100
sharpe_ratio = lambda w: ret_esperado(w) / vol_esperada(w)

# CAGR
dias = len(precos)
anos_observados = dias / 252
def calc_cagr(w):
    retorno_total = (1 + (precos.pct_change().dropna() @ w)).prod()
    return (retorno_total ** (1 / anos_observados) - 1) * 100

# Tabelas
resultados = pd.DataFrame({
    "Ticker": tickers,
    "Informada (%)": np.round(pesos_informados * 100, 2),
    "Melhor Sharpe (%)": np.round(pesos_otimizados * 100, 2),
    "Maior Retorno (%)": np.round(pesos_max_ret * 100, 2),
})

st.subheader("Alocações")
st.dataframe(resultados.set_index("Ticker"))

st.subheader("Indicadores")
st.markdown("**Carteira Informada**")
st.write(f"Retorno Esperado: {ret_esperado(pesos_informados):.2f}%")
st.write(f"Volatilidade Anual: {vol_esperada(pesos_informados):.2f}%")
st.write(f"Sharpe: {sharpe_ratio(pesos_informados):.2f}")
st.write(f"CAGR: {calc_cagr(pesos_informados):.2f}%")

st.markdown("**Carteira Otimizada - Melhor Sharpe**")
st.write(f"Retorno Esperado: {ret_esperado(pesos_otimizados):.2f}%")
st.write(f"Volatilidade Anual: {vol_esperada(pesos_otimizados):.2f}%")
st.write(f"Sharpe: {sharpe_ratio(pesos_otimizados):.2f}")
st.write(f"CAGR: {calc_cagr(pesos_otimizados):.2f}%")

st.markdown("**Carteira de Maior Retorno**")
st.write(f"Retorno Esperado: {ret_esperado(pesos_max_ret):.2f}%")
st.write(f"Volatilidade Anual: {vol_esperada(pesos_max_ret):.2f}%")
st.write(f"Sharpe: {sharpe_ratio(pesos_max_ret):.2f}")
st.write(f"CAGR: {calc_cagr(pesos_max_ret):.2f}%")

