import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, expected_returns
from pypfopt.risk_models import CovarianceShrinkage

# Evita erro de estilo
plt.style.use("ggplot")

st.set_page_config(page_title="Otimização da Carteira - Sharpe Máximo", layout="wide")
st.title("Otimização da Carteira com Índice de Sharpe")

st.sidebar.header("Configurações da Carteira")

# Ativos e alocações iniciais
ativos = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA", "PRIO3.SA",
    "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]

pesos_atuais = [
    0.10, 0.012, 0.065, 0.106, 0.05, 0.005, 0.15,
    0.15, 0.067, 0.04, 0.064, 0.15, 0.01, 0.001, 0.03
]

carteira_df = pd.DataFrame({"Ativo": ativos, "Peso Atual": pesos_atuais})

# Período de análise
anos = st.sidebar.slider("Anos de histórico", 3, 10, 7)

st.sidebar.markdown("---")
st.sidebar.dataframe(carteira_df.set_index("Ativo"))

# Coleta de dados
st.subheader("1. Coleta de dados históricos")
with st.spinner("Baixando dados..."):
    raw_data = yf.download(ativos, period=f"{anos}y", group_by="ticker")

    try:
        dados = raw_data['Adj Close'].copy()
    except KeyError:
        st.warning("'Adj Close' não disponível, usando 'Close' como fallback.")
        dados = raw_data['Close'].copy()

    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.droplevel(0)

    dados = dados.dropna()
    st.success("Dados carregados com sucesso!")
    st.line_chart(dados)

# Retornos e covariância
st.subheader("2. Cálculo de métricas")
retornos_anuais = expected_returns.mean_historical_return(dados, frequency=252)
matriz_cov = CovarianceShrinkage(dados).ledoit_wolf()

# Verificação de consistência
assert not retornos_anuais.isnull().any(), "Retornos contêm NaNs"
assert not matriz_cov.isnull().any().any(), "Covariância contém NaNs"

# Sharpe da carteira atual
peso_array = np.array(pesos_atuais)
retorno_atual = np.dot(peso_array, retornos_anuais)
vol_atual = np.sqrt(np.dot(peso_array.T, np.dot(matriz_cov, peso_array)))
sharpe_atual = retorno_atual / vol_atual

col1, col2, col3 = st.columns(3)
col1.metric("Retorno Esperado (Atual)", f"{retorno_atual:.2%}")
col2.metric("Volatilidade (Atual)", f"{vol_atual:.2%}")
col3.metric("Sharpe (Atual)", f"{sharpe_atual:.2f}")

# Otimização
st.subheader("3. Otimização da carteira")
ef = EfficientFrontier(retornos_anuais, matriz_cov)
pesos_otimizados = ef.max_sharpe()
pesos_limpos = ef.clean_weights()
retorno_opt, vol_opt, sharpe_opt = ef.portfolio_performance()

col1, col2, col3 = st.columns(3)
col1.metric("Retorno Otimizado", f"{retorno_opt:.2%}")
col2.metric("Volatilidade Otimizada", f"{vol_opt:.2%}")
col3.metric("Sharpe Otimizado", f"{sharpe_opt:.2f}")

# Comparação
st.subheader("4. Comparação gráfica")
fig, ax = plt.subplots(figsize=(10, 5))

indices = np.arange(len(ativos))
largura = 0.35

ax.bar(indices - largura/2, pesos_atuais, largura, label='Atual', color='blue')
ax.bar(indices + largura/2, [pesos_limpos[a] for a in ativos], largura, label='Otimizado', color='green')

ax.set_xticks(indices)
ax.set_xticklabels(ativos, rotation=45)
ax.set_ylabel('Alocação (%)')
ax.set_title('Comparação de Alocação por Ativo')
ax.legend()
st.pyplot(fig)

# Exportar resultados
st.subheader("5. Pesos otimizados")
st.dataframe(pd.DataFrame.from_dict(pesos_limpos, orient='index', columns=['Peso (%)']).applymap(lambda x: f"{x*100:.2f}%"))
