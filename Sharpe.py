import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import expected_returns, risk_models
import matplotlib.pyplot as plt

st.set_page_config(page_title="Otimização de Carteira", layout="wide")
st.title("Otimização de Carteira - Fronteira Eficiente com Monte Carlo")

anos = st.slider("Anos de histórico", 1, 10, 7)

st.markdown("### Tabela de Ativos")
df_input = st.data_editor(
    pd.DataFrame(columns=["Ticker", "Peso", "Min", "Max"]),
    use_container_width=True,
    num_rows="dynamic"
)

df_input.dropna(inplace=True)
df_input.reset_index(drop=True, inplace=True)

if df_input.empty:
    st.warning("Adicione ao menos um ativo.")
    st.stop()

tickers = df_input["Ticker"].tolist()
pesos_informados = df_input["Peso"].values / 100
limites_min = df_input["Min"].values / 100
limites_max = df_input["Max"].values / 100

# Baixar dados e tratar
raw = yf.download(tickers, period=f"{anos}y", group_by='ticker', auto_adjust=True, progress=False)

dados = pd.DataFrame()
for ticker in tickers:
    try:
        dados[ticker] = raw[ticker]['Close']
    except Exception as e:
        st.error(f"Erro ao processar {ticker}: {e}")
        st.stop()

dados.dropna(inplace=True)

# Retornos esperados e covariância
mu = expected_returns.mean_historical_return(dados)
S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

# Simulações Monte Carlo
num_simulacoes = 1_000_000
resultados = []
pesos_simulados = []

for _ in range(num_simulacoes):
    pesos = np.random.dirichlet(np.ones(len(tickers)))
    if np.all(pesos >= limites_min) and np.all(pesos <= limites_max):
        retorno_esperado = np.dot(pesos, mu)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
        sharpe = retorno_esperado / volatilidade
        resultados.append([pesos, retorno_esperado, volatilidade, sharpe])
        pesos_simulados.append(pesos)

# Ordenar
simulacoes = sorted(resultados, key=lambda x: x[3], reverse=True)
carteira_sharpe = simulacoes[0]
carteira_retorno = max(simulacoes, key=lambda x: x[1])

def desempenho(pesos):
    ret = np.dot(pesos, mu)
    vol = np.sqrt(np.dot(pesos.T, np.dot(S, pesos)))
    sharpe = ret / vol
    cagr = (1 + ret) ** (1 / anos) - 1
    return ret, vol, sharpe, cagr

def mostrar_resultados(nome, pesos):
    ret, vol, sharpe, cagr = desempenho(pesos)
    st.subheader(f"Resultados - {nome}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retorno Esperado (%)", f"{ret*100:.2f}")
    col2.metric("Volatilidade Anual (%)", f"{vol*100:.2f}")
    col3.metric("Índice de Sharpe", f"{sharpe:.2f}")
    col4.metric("CAGR (%)", f"{cagr*100:.2f}")
    st.dataframe(pd.DataFrame({
        "Ticker": tickers,
        "Alocação (%)": np.round(pesos * 100, 2)
    }), use_container_width=True)

# Exibir resultados
mostrar_resultados("Carteira Informada", pesos_informados)
mostrar_resultados("Carteira de Maior Sharpe", carteira_sharpe[0])
mostrar_resultados("Carteira de Maior Retorno", carteira_retorno[0])

# Gráfico da Fronteira Eficiente
sim_retornos = [x[1] * 100 for x in simulacoes]
sim_vols = [x[2] * 100 for x in simulacoes]

ret_inf, vol_inf, _sh, _cg = desempenho(pesos_informados)
ret_sharpe, vol_sharpe, _sh, _cg = desempenho(carteira_sharpe[0])
ret_max, vol_max, _sh, _cg = desempenho(carteira_retorno[0])

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(sim_vols, sim_retornos, c='lightblue', alpha=0.4, label='Carteiras Simuladas')
ax.scatter(vol_inf * 100, ret_inf * 100, color='gray', marker='X', s=100, label='Informada')
ax.scatter(vol_sharpe * 100, ret_sharpe * 100, color='green', marker='*', s=150, label='Maior Sharpe')
ax.scatter(vol_max * 100, ret_max * 100, color='orange', marker='D', s=100, label='Maior Retorno')
ax.set_title("Fronteira Eficiente - Simulação de Monte Carlo")
ax.set_xlabel("Volatilidade Anual (%)")
ax.set_ylabel("Retorno Esperado (%)")
ax.legend()
st.pyplot(fig)
