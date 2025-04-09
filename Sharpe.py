import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier
import matplotlib.pyplot as plt

# Configurações da página
st.set_page_config(layout="wide")
st.title("Otimização de Carteira - Máximo Sharpe")

# Ativos e período
ativos = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA",
    "PRIO3.SA", "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA",
    "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]
anos = 7

# Baixar dados
st.subheader("1. Coleta de dados históricos")
with st.spinner("Baixando dados dos ativos..."):
    raw_data = yf.download(ativos, period=f"{anos}y", group_by="ticker", auto_adjust=True)

    try:
        dados = pd.concat([raw_data[ticker]["Close"] for ticker in ativos], axis=1)
        dados.columns = ativos
    except KeyError:
        st.error("Erro ao acessar os dados de fechamento dos ativos.")
        st.stop()

# Pré-processamento robusto
dados = dados.dropna()
dados = dados.astype(float)

if dados.isnull().values.any() or np.isinf(dados.values).any():
    st.error("Dados contêm valores inválidos (NaN ou inf). Verifique os ativos ou ajuste o período.")
    st.stop()

st.success("Dados carregados e validados com sucesso!")
st.line_chart(dados)

# Retornos e risco
st.subheader("2. Cálculo de retornos e risco")
mu = expected_returns.mean_historical_return(dados)
try:
    S = risk_models.CovarianceShrinkage(dados).ledoit_wolf()
except Exception as e:
    st.error(f"Erro ao calcular a matriz de covariância: {e}")
    st.stop()

# Otimização
st.subheader("3. Otimização da carteira (Sharpe Máximo)")
try:
    ef = EfficientFrontier(mu, S)
    pesos_otimizados = ef.max_sharpe()
    limpos = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance()
except Exception as e:
    st.error(f"Erro durante a otimização: {e}")
    st.stop()

# Exibir resultados
st.write("### Pesos Otimizados")
df_pesos = pd.DataFrame(limpos.items(), columns=["Ativo", "Peso (%)"])
df_pesos["Peso (%)"] *= 100
st.dataframe(df_pesos.sort_values("Peso (%)", ascending=False).set_index("Ativo"))

st.markdown(f"""
**Retorno esperado:** {ret:.2%}  
**Volatilidade esperada:** {vol:.2%}  
**Índice de Sharpe:** {sharpe:.2f}
""")

# Gráfico de alocação
st.subheader("4. Gráfico de Alocação")
fig, ax = plt.subplots()
df_pesos.set_index("Ativo").sort_values("Peso (%)").plot(kind='barh', legend=False, ax=ax)
plt.xlabel("Peso (%)")
plt.tight_layout()
st.pyplot(fig)

