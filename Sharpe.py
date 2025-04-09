import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Otimização de Carteira - Máximo Índice de Sharpe")

# Lista dos ativos e período
ativos = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA",
    "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA",
    "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]
anos = 7

st.subheader("1. Coleta de dados históricos")
with st.spinner("Baixando dados..."):
    raw_data = yf.download(ativos, period=f"{anos}y", group_by="ticker", auto_adjust=True)

    try:
        dados = pd.concat([raw_data[ticker]["Close"] for ticker in ativos], axis=1)
        dados.columns = ativos
    except KeyError:
        st.error("Erro ao acessar os dados de fechamento.")
        st.stop()

    dados = dados.dropna()
    st.success("Dados carregados com sucesso!")
    st.line_chart(dados)

# Retornos esperados e matriz de risco
st.subheader("2. Cálculo dos retornos e risco")
mu = expected_returns.mean_historical_return(dados)
S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()

# Otimização da carteira
st.subheader("3. Otimização da carteira (Sharpe máximo)")
ef = EfficientFrontier(mu, S)
pesos_otimizados = ef.max_sharpe()
limpos = ef.clean_weights()
retorno_esperado, volatilidade, sharpe = ef.portfolio_performance()

st.write("**Pesos otimizados:**")
st.write(pd.DataFrame(limpos.items(), columns=["Ativo", "Peso (%)"]).sort_values("Peso (%)", ascending=False).set_index("Ativo") * 100)

st.markdown(f"""
**Retorno esperado:** {retorno_esperado:.2%}  
**Volatilidade esperada:** {volatilidade:.2%}  
**Índice de Sharpe:** {sharpe:.2f}
""")

# Gráfico de alocação
st.subheader("4. Alocação da carteira")
fig, ax = plt.subplots()
pd.Series(limpos).sort_values().plot(kind='barh', figsize=(10, 6), ax=ax)
plt.xlabel("Peso")
st.pyplot(fig)
