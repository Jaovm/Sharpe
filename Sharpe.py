import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf

# Título do app
st.title("Otimização de Carteira - Melhor Sharpe com Monte Carlo")

# Entradas do usuário
st.header("Parâmetros da Carteira")
anos = st.slider("Quantos anos de histórico?", min_value=1, max_value=10, value=7)

st.subheader("Informações dos Ativos")
ativos_df = st.data_editor(
    pd.DataFrame({
        "Ticker": ["AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA", "PRIO3.SA", 
                   "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"],
        "Peso Inicial (%)": [10, 1.2, 6.5, 10.6, 5, 0.5, 15, 15, 6.7, 4, 6.4, 15, 1, 0.1, 3],
        "Min (%)": [0]*15,
        "Max (%)": [30]*15
    }),
    num_rows="fixed",
)

# Processar entradas
tickers = ativos_df["Ticker"].tolist()
pesos_iniciais = np.array(ativos_df["Peso Inicial (%)"].tolist()) / 100
pesos_min = np.array(ativos_df["Min (%)"].tolist()) / 100
pesos_max = np.array(ativos_df["Max (%)"].tolist()) / 100

# Baixar dados
st.write("Baixando dados...")
raw_data = yf.download(tickers, period=f"{anos}y", group_by="ticker", auto_adjust=True)

# Corrigir estrutura dos dados
if isinstance(raw_data.columns, pd.MultiIndex):
    dados = pd.DataFrame({ticker: raw_data[ticker]["Close"] for ticker in tickers})
else:
    dados = raw_data[["Close"]]
    dados.columns = [tickers[0]]

dados = dados.dropna()

# Retornos
retornos = dados.pct_change().dropna()

# Matriz de covariância (Ledoit-Wolf)
st.write("Calculando matriz de covariância...")
cov_matrix = LedoitWolf().fit(retornos).covariance_
media_retornos = retornos.mean()

# Simulação de Monte Carlo
st.write("Rodando simulação Monte Carlo...")
num_simulacoes = 50_000
resultados = []
pesos_lista = []

np.random.seed(42)
for _ in range(num_simulacoes):
    pesos = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()

    # Respeitar restrições
    if np.all(pesos >= pesos_min) and np.all(pesos <= pesos_max):
        retorno_esperado = np.dot(pesos, media_retornos) * 252
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * 252, pesos)))
        sharpe = retorno_esperado / volatilidade
        resultados.append([retorno_esperado, volatilidade, sharpe])
        pesos_lista.append(pesos)

# Obter melhor resultado
resultados = np.array(resultados)
melhor_indice = np.argmax(resultados[:, 2])
melhor_pesos = pesos_lista[melhor_indice]
melhor_retorno, melhor_vol, melhor_sharpe = resultados[melhor_indice]

# Resultado final
st.subheader("Resultados")
st.metric("Retorno Esperado (%)", f"{melhor_retorno*100:.2f}")
st.metric("Volatilidade Anual (%)", f"{melhor_vol*100:.2f}")
st.metric("Índice de Sharpe", f"{melhor_sharpe:.2f}")

st.subheader("Alocação Otimizada")
df_resultado = pd.DataFrame({
    "Ticker": tickers,
    "Alocação Ideal (%)": melhor_pesos * 100
})
st.dataframe(df_resultado.style.format({"Alocação Ideal (%)": "{:.2f}"}))
