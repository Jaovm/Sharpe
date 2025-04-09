import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import expected_returns, risk_models
import matplotlib.pyplot as plt

# Título
st.title("Otimização de Carteira com Índice de Sharpe - Monte Carlo")

# Entradas do usuário
anos = st.number_input("Período (anos)", min_value=1, max_value=20, value=7)
tabela = st.data_editor(
    pd.DataFrame(
        {
            "Ticker": ["AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA",
                        "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA",
                        "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"],
            "Peso (%)": [10, 1.2, 6.5, 10.6, 5, 0.5, 15, 15, 6.7, 4, 6.4, 15, 1, 0.1, 3],
            "Min (%)": [0] * 15,
            "Max (%)": [100] * 15,
        }
    ),
    num_rows="dynamic",
    use_container_width=True
)

if st.button("Otimizar Carteira"):
    st.subheader("Resultados da Otimização")
    tickers = tabela["Ticker"].tolist()
    pesos_iniciais = np.array(tabela["Peso (%)"]) / 100
    min_weights = np.array(tabela["Min (%)"]) / 100
    max_weights = np.array(tabela["Max (%)"]) / 100

    raw_data = yf.download(tickers, period=f"{anos}y", group_by="ticker", auto_adjust=True, progress=False)

    # Montar DataFrame apenas com preços ajustados de fechamento
    dados = pd.DataFrame({ticker: raw_data[ticker]["Close"] for ticker in tickers if "Close" in raw_data[ticker]})
    dados.dropna(axis=0, how="any", inplace=True)

    # Calcular retornos esperados e matriz de covariância
    mu = expected_returns.mean_historical_return(dados)
    S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

    # Simulação de Monte Carlo
    n_simulacoes = 5000000
    resultados = []
    pesos_simulados = []

    for _ in range(n_simulacoes):
        pesos = np.random.dirichlet(np.ones(len(tickers)))
        if np.all(pesos >= min_weights) and np.all(pesos <= max_weights):
            ret = np.dot(pesos, mu)
            vol = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
            sharpe = ret / vol
            resultados.append((ret, vol, sharpe))
            pesos_simulados.append(pesos)

    resultados = np.array(resultados)
    idx_max_sharpe = np.argmax(resultados[:, 2])
    melhor_pesos = pesos_simulados[idx_max_sharpe]

    # Resultados da carteira otimizada
    retorno_otim = resultados[idx_max_sharpe][0]
    vol_otim = resultados[idx_max_sharpe][1]
    sharpe_otim = resultados[idx_max_sharpe][2]
    cagr_otim = (1 + retorno_otim) ** 1 - 1  # Aproximado para retorno anual já em base anual

    # Resultados da carteira informada
    retorno_inf = np.dot(pesos_iniciais, mu)
    vol_inf = np.sqrt(np.dot(pesos_iniciais.T, np.dot(S, pesos_iniciais)))
    sharpe_inf = retorno_inf / vol_inf
    cagr_inf = (1 + retorno_inf) ** 1 - 1

    # Exibir resultados
    resultados_df = pd.DataFrame({
        "Carteira": ["Informada", "Otimizada"],
        "Retorno Esperado (%)": [retorno_inf * 100, retorno_otim * 100],
        "Volatilidade Anual (%)": [vol_inf * 100, vol_otim * 100],
        "Índice de Sharpe": [sharpe_inf, sharpe_otim],
        "CAGR (%)": [cagr_inf * 100, cagr_otim * 100]
    })
    st.dataframe(resultados_df, use_container_width=True)

    # Pesos otimizados
    st.subheader("Pesos Otimizados")
    pesos_df = pd.DataFrame({
        "Ticker": tickers,
        "Peso Otimizado (%)": (melhor_pesos * 100).round(2)
    })
    st.dataframe(pesos_df, use_container_width=True)

    # Gráfico (opcional)
    fig, ax = plt.subplots()
    ax.scatter(resultados[:, 1], resultados[:, 0], c=resultados[:, 2], cmap="viridis", s=1)
    ax.scatter(vol_otim, retorno_otim, c="red", marker="*", s=200, label="Melhor Sharpe")
    ax.set_xlabel("Volatilidade")
    ax.set_ylabel("Retorno")
    ax.set_title("Simulação Monte Carlo - Fronteira Eficiente")
    ax.legend()
    st.pyplot(fig)

