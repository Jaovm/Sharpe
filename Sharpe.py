import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import expected_returns, risk_models
from sklearn.covariance import LedoitWolf

st.title("Otimização de Carteira - Melhor Sharpe com Monte Carlo")

st.markdown("### Parâmetros da Carteira")

anos = st.slider("Período de análise (anos)", 1, 10, 7)
simulacoes = st.slider("Número de simulações Monte Carlo", 1000, 50000, 10000, step=1000)

st.markdown("### Tabela de Ativos")
df_input = st.data_editor(
    pd.DataFrame({
        "Ticker": ["AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
                    "ITUB3.SA", "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA",
                    "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"],
        "Peso": [10, 1.2, 6.5, 10.6, 5, 0.5, 15, 15, 6.7, 4, 6.4, 15, 1, 0.1, 3],
        "Min": [0]*15,
        "Max": [30]*15
    }),
    num_rows="dynamic",
    use_container_width=True
)

tickers = df_input["Ticker"].tolist()
pesos_informados = np.array(df_input["Peso"].tolist()) / 100
bounds = list(zip(df_input["Min"] / 100, df_input["Max"] / 100))

try:
    raw_data = yf.download(tickers, period=f"{anos}y", auto_adjust=True)["Close"]
    dados = raw_data.dropna()

    mu = expected_returns.mean_historical_return(dados)
    S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

    # Estatísticas da carteira informada
    ret_inf = np.dot(pesos_informados, mu)
    vol_inf = np.sqrt(np.dot(pesos_informados.T, np.dot(S, pesos_informados)))
    sharpe_inf = ret_inf / vol_inf

    # Simulação de Monte Carlo
    resultados = []
    for _ in range(simulacoes):
        pesos = np.random.dirichlet(np.ones(len(tickers)))
        if all(bounds[i][0] <= pesos[i] <= bounds[i][1] for i in range(len(tickers))):
            ret = np.dot(pesos, mu)
            vol = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
            sharpe = ret / vol
            resultados.append((sharpe, ret, vol, pesos))

    if resultados:
        melhor = max(resultados, key=lambda x: x[0])
        melhor_sharpe, melhor_ret, melhor_vol, melhor_pesos = melhor

        st.markdown("### Resultados")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Carteira Otimizada")
            st.write("**Retorno Esperado (%):**", round(melhor_ret * 100, 2))
            st.write("**Volatilidade Anual (%):**", round(melhor_vol * 100, 2))
            st.write("**Índice de Sharpe:**", round(melhor_sharpe, 2))

        with col2:
            st.subheader("Carteira Informada")
            st.write("**Retorno Esperado (%):**", round(ret_inf * 100, 2))
            st.write("**Volatilidade Anual (%):**", round(vol_inf * 100, 2))
            st.write("**Índice de Sharpe:**", round(sharpe_inf, 2))

        st.subheader("Pesos da Carteira Otimizada")
        st.dataframe(pd.DataFrame({"Ticker": tickers, "Peso Otimizado (%)": melhor_pesos * 100}))
    else:
        st.warning("Nenhuma simulação atendeu às restrições de alocação. Tente ajustar os limites mínimos e máximos.")

except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
    
