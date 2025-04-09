import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Ativos e alocações fornecidas
tickers = [
    "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA", "PRIO3.SA",
    "PSSA3.SA", "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"
]

pesos_atuais = np.array([
    0.10, 0.012, 0.065, 0.106, 0.05, 0.005, 0.15,
    0.15, 0.067, 0.04, 0.064, 0.15, 0.01, 0.001, 0.03
])

# Baixando dados dos últimos 7 anos
dados = yf.download(tickers, start="2018-04-01", end="2025-04-01")["Adj Close"]
dados = dados.dropna()

# Retornos diários e anuais esperados
retornos_diarios = dados.pct_change().dropna()
retornos_anuais = expected_returns.mean_historical_return(dados, frequency=252)

# Matriz de covariância
matriz_cov = risk_models.sample_cov(dados, frequency=252)

# Índice de Sharpe da carteira atual
retorno_port_atual = np.dot(pesos_atuais, retornos_anuais)
vol_port_atual = np.sqrt(np.dot(pesos_atuais.T, np.dot(matriz_cov, pesos_atuais)))
sharpe_atual = (retorno_port_atual - 0.0) / vol_port_atual  # Rf = 0

print(f"Retorno esperado (atual): {retorno_port_atual:.2%}")
print(f"Volatilidade (atual): {vol_port_atual:.2%}")
print(f"Índice de Sharpe (atual): {sharpe_atual:.2f}")

# Otimização com PyPortfolioOpt
ef = EfficientFrontier(retornos_anuais, matriz_cov)
pesos_otimizados = ef.max_sharpe()
limpos = ef.clean_weights()

print("\nPesos otimizados para máximo Sharpe:")
for ativo, peso in limpos.items():
    print(f"{ativo}: {peso:.2%}")

# Desempenho da carteira otimizada
ret, vol, sharpe = ef.portfolio_performance(verbose=True)

# Comparando graficamente
plt.figure(figsize=(10, 6))
sns.barplot(x=tickers, y=pesos_atuais, color='blue', label='Atual')
sns.barplot(x=list(limpos.keys()), y=list(limpos.values()), color='green', alpha=0.6, label='Otimizado')
plt.xticks(rotation=45)
plt.ylabel("Alocação")
plt.title("Comparação entre Carteira Atual e Otimizada")
plt.legend()
plt.tight_layout()
plt.show()
