import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
import streamlit as st

st.set_page_config(layout="wide")

# Configurações iniciais
start_date = '2017-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

if 'tickers_dict' not in st.session_state:
    st.session_state.tickers_dict = {
        'AGRO3.SA': 0.10, 'BBAS3.SA': 0.012, 'BBSE3.SA': 0.065, 'BPAC11.SA': 0.106,
        'EGIE3.SA': 0.05, 'ITUB3.SA': 0.005, 'PRIO3.SA': 0.15, 'PSSA3.SA': 0.15,
        'SAPR3.SA': 0.067, 'SBSP3.SA': 0.04, 'VIVT3.SA': 0.064, 'WEGE3.SA': 0.15,
        'TOTS3.SA': 0.01, 'B3SA3.SA': 0.001, 'TAEE3.SA': 0.03
    }

st.sidebar.header("Gerenciar Tickers")
novo_ticker = st.sidebar.text_input("Novo ticker (ex: PETR4.SA)")
peso_ticker = st.sidebar.number_input("Peso (%)", min_value=0.0, max_value=1.0, step=0.01)
if st.sidebar.button("Adicionar ticker") and novo_ticker:
    st.session_state.tickers_dict[novo_ticker.upper()] = peso_ticker

remover_ticker = st.sidebar.selectbox("Remover ticker", [""] + list(st.session_state.tickers_dict.keys()))
if st.sidebar.button("Remover") and remover_ticker:
    st.session_state.tickers_dict.pop(remover_ticker, None)

min_aloc = st.sidebar.slider("Alocação mínima por ativo (%)", 0.0, 0.1, 0.0, 0.01)
max_aloc = st.sidebar.slider("Alocação máxima por ativo (%)", 0.1, 1.0, 0.3, 0.01)

@st.cache_data(show_spinner=False)
def baixar_dados(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
        df = df.stack(level=0).rename_axis(index=['Date', 'Ticker']).reset_index()
        df = df.pivot(index='Date', columns='Ticker', values='Close')
        return df.dropna(axis=1, how='any')
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

def calcular_retorno_cov(dados):
    retornos = dados.pct_change().dropna()
    retorno_medio = retornos.mean() * 252
    cov_matrix = LedoitWolf().fit(retornos).covariance_ * 252
    return retorno_medio, cov_matrix

def simular_carteiras(retorno_medio, cov_matrix, num_portfolios=100000, rf=0.0):
    n = len(retorno_medio)
    resultados = []
    pesos_lista = []
    for _ in range(num_portfolios):
        while True:
            pesos = np.random.dirichlet(np.ones(n), size=1)[0]
            if all(min_aloc <= w <= max_aloc for w in pesos):
                break
        retorno = np.dot(pesos, retorno_medio)
        risco = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        sharpe = (retorno - rf) / risco if risco != 0 else 0
        resultados.append([retorno, risco, sharpe])
        pesos_lista.append(pesos)

    resultados = np.array(resultados)
    melhor_idx = np.argmax(resultados[:, 2])
    maior_ret_idx = np.argmax(resultados[:, 0])

    melhor_sharpe = {
        'retorno': resultados[melhor_idx, 0],
        'risco': resultados[melhor_idx, 1],
        'sharpe': resultados[melhor_idx, 2],
        'pesos': pesos_lista[melhor_idx]
    }

    maior_retorno = {
        'retorno': resultados[maior_ret_idx, 0],
        'risco': resultados[maior_ret_idx, 1],
        'sharpe': resultados[maior_ret_idx, 2],
        'pesos': pesos_lista[maior_ret_idx]
    }

    return resultados, pesos_lista, melhor_sharpe, maior_retorno

def plotar_grafico(resultados):
    plt.figure(figsize=(12, 6))
    plt.scatter(resultados[:, 1], resultados[:, 0], c=resultados[:, 2], cmap='viridis', s=3)
    plt.xlabel('Risco (Volatilidade)')
    plt.ylabel('Retorno Esperado')
    plt.title('Fronteira Eficiente - Simulação de Monte Carlo')
    st.pyplot(plt.gcf())

def sugerir_ativos_por_cenario():
    st.sidebar.subheader("Cenário Macroeconômico")
    cenarios = {
        "Alta de Juros": ["Bancos", "Seguradoras", "Tesouro Direto"],
        "Inflação em Alta": ["Setor de energia", "Commodities"],
        "PIB em Crescimento": ["Varejo", "Construção Civil", "Tecnologia"],
        "Recessão": ["Utilities", "Alimentos", "Saúde"],
        "Dólar em Alta": ["Exportadoras", "Mineração", "Petróleo"]
    }
    cenario = st.sidebar.selectbox("Selecione um cenário macroeconômico", [""] + list(cenarios.keys()))
    if cenario:
        st.sidebar.info(f"Setores/ativos que tendem a se beneficiar: {', '.join(cenarios[cenario])}")

def exibir_resultados(dados, pesos_informados):
    retorno_medio, cov_matrix = calcular_retorno_cov(dados)
    ativos_validos = dados.columns.intersection(pesos_informados.keys())
    if len(ativos_validos) == 0:
        st.error("Nenhum ativo com dados válidos para análise.")
        return

    retorno_medio = retorno_medio[ativos_validos]
    cov_matrix_df = pd.DataFrame(cov_matrix, index=dados.columns, columns=dados.columns)
    cov_matrix = cov_matrix_df.loc[ativos_validos, ativos_validos].values
    pesos_informados_arr = np.array([pesos_informados[tic] for tic in ativos_validos])
    pesos_informados_arr /= pesos_informados_arr.sum()

    ret_informado = np.dot(pesos_informados_arr, retorno_medio)
    risco_informado = np.sqrt(np.dot(pesos_informados_arr.T, np.dot(cov_matrix, pesos_informados_arr)))

    st.subheader("Carteira Informada")
    st.write(f"Retorno esperado anualizado: {ret_informado:.2%}")
    st.write(f"Volatilidade anualizada: {risco_informado:.2%}")

    resultados, pesos, melhor_sharpe, maior_retorno = simular_carteiras(retorno_medio, cov_matrix)

    st.subheader("Carteira com Melhor Índice de Sharpe")
    st.write(f"Retorno: {melhor_sharpe['retorno']:.2%}")
    st.write(f"Risco: {melhor_sharpe['risco']:.2%}")
    st.write(f"Sharpe: {melhor_sharpe['sharpe']:.2f}")
    st.dataframe(pd.DataFrame({'Ticker': ativos_validos, 'Peso': melhor_sharpe['pesos']}))

    st.subheader("Carteira com Maior Retorno Esperado")
    st.write(f"Retorno: {maior_retorno['retorno']:.2%}")
    st.write(f"Risco: {maior_retorno['risco']:.2%}")
    st.write(f"Sharpe: {maior_retorno['sharpe']:.2f}")
    st.dataframe(pd.DataFrame({'Ticker': ativos_validos, 'Peso': maior_retorno['pesos']}))

    plotar_grafico(resultados)

def rodar_analise(tickers_dict, start, end):
    dados = baixar_dados(list(tickers_dict.keys()), start, end)
    if not dados.empty:
        exibir_resultados(dados, tickers_dict)
    else:
        st.error("Erro ao carregar os dados. Verifique os tickers ou a conexão.")

sugerir_ativos_por_cenario()
rodar_analise(st.session_state.tickers_dict, start_date, end_date)
