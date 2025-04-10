import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from sklearn.covariance import LedoitWolf
import warnings

warnings.filterwarnings("ignore")

# Ações já informadas
ativos_tabela = pd.DataFrame({
    "Ticker": ["AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA", "ITUB3.SA", "PRIO3.SA", "PSSA3.SA",
               "SAPR3.SA", "SBSP3.SA", "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA"],
    "Peso": [10, 1.2, 6.5, 10.6, 5, 0.5, 15, 15, 6.7, 4, 6.4, 15, 1, 0.1, 3],
    "Min": [0]*15,
    "Max": [1]*15
})

ativos_tabela["Peso"] = ativos_tabela["Peso"] / 100

# Permite adicionar mais ativos
adicionar_mais = input("Deseja adicionar mais ativos? (s/n): ").lower()
if adicionar_mais == 's':
    while True:
        ticker = input("Ticker (ex: PETR4.SA): ").upper()
        peso = float(input("Peso desejado (%): ")) / 100
        min_peso = float(input("Alocação mínima (%): ")) / 100
        max_peso = float(input("Alocação máxima (%): ")) / 100
        ativos_tabela = pd.concat([ativos_tabela, pd.DataFrame([{
            "Ticker": ticker,
            "Peso": peso,
            "Min": min_peso,
            "Max": max_peso
        }])], ignore_index=True)

        if input("Deseja adicionar outro? (s/n): ").lower() != 's':
            break

# Parâmetros
tickers = ativos_tabela["Ticker"].tolist()
anos = 7
simulacoes = 1_000_000
rf = 0.0  # taxa livre de risco

# Coleta de dados com fallback robusto
print("Baixando dados...")
raw_data = yf.download(tickers, period=f"{anos}y", group_by='ticker', auto_adjust=False)

# Tentativa de obter "Adj Close", fallback para "Close" se necessário
try:
    dados = raw_data["Adj Close"].dropna(how="all")
except KeyError:
    print("Coluna 'Adj Close' não encontrada. Usando 'Close'.")
    if isinstance(raw_data.columns, pd.MultiIndex):
        dados = pd.DataFrame({t: raw_data[t]["Close"] for t in tickers if "Close" in raw_data[t]}).dropna(how="all")
    else:
        dados = raw_data["Close"].dropna(how="all")

# Verificação final
if dados.empty or dados.isnull().all().all():
    raise ValueError("Erro ao calcular a matriz de covariância. Verifique os dados dos ativos.")

# Cálculos financeiros
retornos = dados.pct_change().dropna()
media_retornos = expected_returns.mean_historical_return(dados, frequency=252)
cov_anual = risk_models.CovarianceShrinkage(retornos).ledoit_wolf()

# Monte Carlo
np.random.seed(42)
resultados = []
pesos_array = []

for _ in range(simulacoes):
    pesos = np.random.dirichlet(np.ones(len(tickers)))
    if not all((pesos >= ativos_tabela["Min"].values) & (pesos <= ativos_tabela["Max"].values)):
        continue
    retorno_esp = np.dot(pesos, media_retornos)
    volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov_anual, pesos)))
    sharpe = (retorno_esp - rf) / volatilidade
    resultados.append([retorno_esp, volatilidade, sharpe])
    pesos_array.append(pesos)

resultados = np.array(resultados)
pesos_array = np.array(pesos_array)

# Melhor Sharpe
idx_max_sharpe = resultados[:, 2].argmax()
melhor_pesos = pesos_array[idx_max_sharpe]
melhor_retorno, melhor_vol, melhor_sharpe = resultados[idx_max_sharpe]
melhor_cagr = (1 + melhor_retorno) ** anos - 1

# Maior retorno
idx_max_retorno = resultados[:, 0].argmax()
retorno_max, vol_max, sharpe_max = resultados[idx_max_retorno]
cagr_max = (1 + retorno_max) ** anos - 1

# Carteira informada
pesos_inf = ativos_tabela["Peso"].values
retorno_inf = np.dot(pesos_inf, media_retornos)
vol_inf = np.sqrt(np.dot(pesos_inf.T, np.dot(cov_anual, pesos_inf)))
sharpe_inf = (retorno_inf - rf) / vol_inf
cagr_inf = (1 + retorno_inf) ** anos - 1

# Exibir resultados
print("\n--- Resultados ---")
print(f"Carteira Informada:\n  Retorno Esperado: {retorno_inf*100:.2f}%\n  Volatilidade: {vol_inf*100:.2f}%\n  Sharpe: {sharpe_inf:.2f}\n  CAGR: {cagr_inf*100:.2f}%")

print(f"\nMelhor Sharpe:\n  Retorno Esperado: {melhor_retorno*100:.2f}%\n  Volatilidade: {melhor_vol*100:.2f}%\n  Sharpe: {melhor_sharpe:.2f}\n  CAGR: {melhor_cagr*100:.2f}%")
print(pd.DataFrame({"Ticker": tickers, "Peso (%)": (melhor_pesos * 100).round(2)}))

print(f"\nMaior Retorno:\n  Retorno Esperado: {retorno_max*100:.2f}%\n  Volatilidade: {vol_max*100:.2f}%\n  Sharpe: {sharpe_max:.2f}\n  CAGR: {cagr_max*100:.2f}%")
print(pd.DataFrame({"Ticker": tickers, "Peso (%)": (pesos_array[idx_max_retorno] * 100).round(2)}))

# Plot opcional
plt.figure(figsize=(10, 6))
plt.scatter(resultados[:, 1], resultados[:, 0], c=resultados[:, 2], cmap="viridis", alpha=0.3)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatilidade Anual")
plt.ylabel("Retorno Esperado")
plt.title("Fronteira Eficiente - Monte Carlo")
plt.scatter([melhor_vol], [melhor_retorno], c="red", label="Melhor Sharpe", marker="*")
plt.scatter([vol_inf], [retorno_inf], c="blue", label="Carteira Informada", marker="X")
plt.scatter([vol_max], [retorno_max], c="green", label="Maior Retorno", marker="D")
plt.legend()
plt.tight_layout()
plt.show()
