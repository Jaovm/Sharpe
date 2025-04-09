import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from sklearn.covariance import LedoitWolf

st.set_page_config(layout="wide")
st.title("Otimização de Carteira com Monte Carlo - Melhor Índice de Sharpe")

# Entrada de dados personalizados
tickers_input = st.text_input(
    "Digite os tickers separados por vírgula",
    value="AGRO3.SA, BBAS3.SA, BBSE3.SA, BPAC11.SA, EGIE3.SA, ITUB3.SA, PRIO3.SA, PSSA3.SA, SAPR3.SA, SBSP3.SA, VIVT3.SA, WEGE3.SA, TOTS3.SA, B3SA3.SA, TAEE3.SA"
)
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

st.subheader("Alocações por ativo")
pesos_iniciais = {}
min_aloc = {}
max_aloc = {}

for ticker in tickers:
    col1, col2, col3 = st.columns(3)
    with col1:
        pesos_iniciais[ticker] = st.number_input(f"Peso inicial {ticker} (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    with col2:
        min_aloc[ticker] = st.number_input(f"Mínimo {ticker} (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    with col3:
        max_aloc[ticker] = st.number_input(f"Máximo {ticker} (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)

anos = st.slider("Anos de histórico", 1, 10, 7)
num_simulacoes = st.slider("Número de simulações Monte Carlo", 10_000, 1_000_000, 50_000, step=10_000)

if st.button("Rodar Otimização"):
    raw_data = yf.download(tickers, period=f"{anos}y", progress=False)

    # Tenta obter "Adj Close" ou "Close"
    try:
        dados = raw_data['Adj Close'].dropna()
    except KeyError:
        dados = raw_data['Close'].dropna()

    # Cálculo de retornos e matriz de covariância com Ledoit-Wolf
    retornos = expected_returns.mean_historical_return(dados)
    matriz_cov = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

    # Simulação de Monte Carlo
    num_ativos = len(tickers)
    resultados = []
    melhores_pesos = None
    melhor_sharpe = -np.inf

    for _ in range(num_simulacoes):
        pesos = np.random.dirichlet(np.ones(num_ativos))

        # Respeita as restrições do usuário
        if any(pesos[i]*100 < min_aloc[tickers[i]] or pesos[i]*100 > max_aloc[tickers[i]] for i in range(num_ativos)):
            continue

        retorno_esperado = np.dot(pesos, retornos.values)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos)))
        sharpe = retorno_esperado / volatilidade

        if sharpe > melhor_sharpe:
            melhor_sharpe = sharpe
            melhores_pesos = pesos

    if melhores_pesos is not None:
        st.subheader("Melhor carteira encontrada")
        df_resultado = pd.DataFrame({
            "Ticker": tickers,
            "Alocação (%)": (melhores_pesos * 100).round(2)
        })
        st.dataframe(df_resultado)
        st.metric("Sharpe Ótimo", round(melhor_sharpe, 4))
    else:
        st.error("Nenhuma carteira viável foi encontrada dentro das restrições informadas. Tente ajustá-las.")

