import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import DiscreteAllocation
from datetime import datetime

st.title("Otimização de Carteira - Markowitz com Monte Carlo")

# Ações pré-definidas
acoes_default = {
    "AGRO3.SA": {"peso": 10, "min": 5, "max": 15},
    "BBAS3.SA": {"peso": 1.2, "min": 0, "max": 3},
    "BBSE3.SA": {"peso": 6.5, "min": 4, "max": 10},
    "BPAC11.SA": {"peso": 10.6, "min": 5, "max": 15},
    "EGIE3.SA": {"peso": 5, "min": 4, "max": 8},
    "ITUB3.SA": {"peso": 0.5, "min": 0, "max": 2},
    "PRIO3.SA": {"peso": 15, "min": 10, "max": 20},
    "PSSA3.SA": {"peso": 15, "min": 10, "max": 20},
    "SAPR3.SA": {"peso": 6.7, "min": 4, "max": 10},
    "SBSP3.SA": {"peso": 4, "min": 3, "max": 8},
    "VIVT3.SA": {"peso": 6.4, "min": 4, "max": 10},
    "WEGE3.SA": {"peso": 15, "min": 10, "max": 20},
    "TOTS3.SA": {"peso": 1, "min": 0, "max": 3},
    "B3SA3.SA": {"peso": 0.1, "min": 0, "max": 2},
    "TAEE3.SA": {"peso": 3, "min": 2, "max": 5},
}

anos = st.slider("Período de Análise (anos)", 1, 10, 7)

# Interface para edição dos ativos
st.subheader("Parâmetros dos Ativos")
tickers, pesos, bounds = [], [], []
for acao, dados in acoes_default.items():
    tickers.append(acao)
    peso = st.number_input(f"Peso para {acao}", value=dados["peso"], key=f"peso_{acao}")
    min_ = st.number_input(f"Mín. alocação (%) {acao}", value=dados["min"], key=f"min_{acao}")
    max_ = st.number_input(f"Máx. alocação (%) {acao}", value=dados["max"], key=f"max_{acao}")
    pesos.append(peso / 100)
    bounds.append((min_ / 100, max_ / 100))

st.write("---")

try:
    st.write("Baixando dados...")
    dados = yf.download(tickers, period=f"{anos}y", progress=False)["Adj Close"]
    dados = dados.dropna()

    st.write("Calculando retornos esperados e covariância...")
    retornos = mean_historical_return(dados)
    cov_anual = CovarianceShrinkage(dados).ledoit_wolf()

    ef = EfficientFrontier(retornos, cov_anual, weight_bounds=(0, 1))
    ef.add_objective(lambda w: 0)  # placeholder

    # Carteira com melhor índice de Sharpe
    w_sharpe = ef.max_sharpe()
    ret_sharpe, vol_sharpe, sharpe_ratio = ef.portfolio_performance()
    w_sharpe = ef.clean_weights()

    # Carteira com maior retorno
    ef_return = EfficientFrontier(retornos, cov_anual, weight_bounds=(0, 1))
    w_ret = ef_return.max_return()
    ret_ret, vol_ret, sharpe_ret = ef_return.portfolio_performance()
    w_ret = ef_return.clean_weights()

    # Carteira informada
    pesos_informados = np.array(pesos)
    pesos_informados /= pesos_informados.sum()
    ret_inf = np.dot(pesos_informados, retornos)
    vol_inf = np.sqrt(np.dot(pesos_informados.T, np.dot(cov_anual, pesos_informados)))
    sharpe_inf = ret_inf / vol_inf
    cagr_inf = (1 + ret_inf) ** anos - 1
    cagr_sharpe = (1 + ret_sharpe) ** anos - 1

    st.subheader("Resultados da Carteira com Melhor Sharpe")
    st.write("Pesos:")
    st.dataframe(pd.DataFrame.from_dict(w_sharpe, orient='index', columns=['Peso']).sort_values(by="Peso", ascending=False))
    st.write(f"Retorno Esperado: {ret_sharpe*100:.2f}%")
    st.write(f"Volatilidade Anual: {vol_sharpe*100:.2f}%")
    st.write(f"Índice de Sharpe: {sharpe_ratio:.2f}")
    st.write(f"CAGR Estimado: {cagr_sharpe*100:.2f}%")

    st.subheader("Carteira com Maior Retorno Esperado")
    st.write("Pesos:")
    st.dataframe(pd.DataFrame.from_dict(w_ret, orient='index', columns=['Peso']).sort_values(by="Peso", ascending=False))
    st.write(f"Retorno Esperado: {ret_ret*100:.2f}%")
    st.write(f"Volatilidade Anual: {vol_ret*100:.2f}%")
    st.write(f"Índice de Sharpe: {sharpe_ret:.2f}")

    st.subheader("Carteira Informada")
    st.write(f"Retorno Esperado: {ret_inf*100:.2f}%")
    st.write(f"Volatilidade Anual: {vol_inf*100:.2f}%")
    st.write(f"Índice de Sharpe: {sharpe_inf:.2f}")
    st.write(f"CAGR Estimado: {cagr_inf*100:.2f}%")

except Exception as e:
    st.error(f"Erro ao calcular: {e}")
