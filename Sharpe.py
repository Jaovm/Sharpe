import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import expected_returns, risk_models
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Otimização de Carteira com Monte Carlo - Máximo Sharpe")

# Entradas do usuário
st.subheader("Parâmetros da carteira")
num_ativos = st.number_input("Número de ativos", min_value=2, max_value=30, value=5)
anos = st.slider("Anos de histórico", min_value=1, max_value=10, value=7)

tickers = []
pesos_iniciais = []
pesos_min = []
pesos_max = []

st.subheader("Configuração dos ativos")
for i in range(num_ativos):
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col1:
        ticker = st.text_input(f"Ticker {i+1}", key=f"ticker_{i}")
    with col2:
        peso = st.number_input(f"Peso inicial (%)", min_value=0.0, max_value=100.0, value=10.0, key=f"peso_{i}")
    with col3:
        pmin = st.number_input("Alocação mín. (%)", min_value=0.0, max_value=100.0, value=0.0, key=f"min_{i}")
    with col4:
        pmax = st.number_input("Alocação máx. (%)", min_value=0.0, max_value=100.0, value=100.0, key=f"max_{i}")
    
    if ticker:
        tickers.append(ticker.upper())
        pesos_iniciais.append(peso / 100)
        pesos_min.append(pmin / 100)
        pesos_max.append(pmax / 100)

# Botão de cálculo
if st.button("Rodar simulação Monte Carlo"):
    with st.spinner("Baixando dados e simulando..."):

        # Baixar dados de preço ajustado
        raw_data = yf.download(tickers, period=f"{anos}y")["Adj Close"]

        # Preencher faltantes e limpar
        dados = raw_data.dropna()

        # Retornos esperados e matriz de covariância
        mu = expected_returns.mean_historical_return(dados)
        S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

        # Simulação Monte Carlo
        n_simul = 20_000
        results = []
        for _ in range(n_simul):
            pesos = np.random.dirichlet(np.ones(len(tickers)), 1).flatten()

            # Respeitar restrições
            if any(p < minv or p > maxv for p, minv, maxv in zip(pesos, pesos_min, pesos_max)):
                continue

            ret = np.dot(pesos, mu)
            vol = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
            sharpe = ret / vol
            results.append((pesos, ret, vol, sharpe))

        # Ordenar por Sharpe
        melhores = sorted(results, key=lambda x: x[3], reverse=True)
        melhor_pesos, melhor_ret, melhor_vol, melhor_sharpe = melhores[0]

        # Exibir resultados
        st.success("Carteira com melhor Sharpe encontrada!")
        df_result = pd.DataFrame({
            "Ticker": tickers,
            "Peso (%)": (melhor_pesos * 100).round(2)
        })
        st.dataframe(df_result)

        col1, col2, col3 = st.columns(3)
        col1.metric("Retorno esperado anual", f"{melhor_ret:.2%}")
        col2.metric("Volatilidade anual", f"{melhor_vol:.2%}")
        col3.metric("Índice de Sharpe", f"{melhor_sharpe:.2f}")

        # Gráfico de pizza
        fig, ax = plt.subplots()
        ax.pie(melhor_pesos, labels=tickers, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
        
