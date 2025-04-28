

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulando dados (em um cenário real, aqui entraríamos com scraping/API)
dates = pd.date_range(start="2022-01-01", periods=500, freq="D")
data = pd.DataFrame(index=dates)
data["Sensex"] = np.cumsum(np.random.normal(0.1, 1.5, size=len(dates))) + 60000
data["USD/INR"] = 74 + np.cumsum(np.random.normal(0.01, 0.2, size=len(dates)))
data["VIX"] = 20 + np.random.normal(0, 3, size=len(dates))
data["CDS_India"] = 90 + np.cumsum(np.random.normal(0, 0.3, size=len(dates)))
data["Gold"] = 1800 + np.cumsum(np.random.normal(0.1, 2, size=len(dates)))

# Título\st.title("Painel de Risco Índia x Paquistão - Visão para Portfólios")

# Seletor de ativos\st.sidebar.header("Selecione os ativos para visualização")
ativos = st.sidebar.multiselect(
    "Ativos:",
    options=data.columns.tolist(),
    default=["Sensex", "USD/INR", "VIX"]
)

# Plot das séries selecionadas
st.subheader("Evolução dos Ativos Selecionados")
fig, ax = plt.subplots(figsize=(10, 5))
data[ativos].plot(ax=ax)
plt.grid(True)
st.pyplot(fig)

# Correlações
st.subheader("Correlação com VIX")
correlacoes = data.pct_change().corr()["VIX"].sort_values(ascending=False)
st.dataframe(correlacoes)

# Sensibilidade (Beta em relação ao VIX)
st.subheader("Sensibilidade dos Ativos ao VIX (Beta)")
betas = {}
returns = data.pct_change().dropna()
for ativo in data.columns:
    if ativo != "VIX":
        cov = np.cov(returns[ativo], returns["VIX"])[0][1]
        var = np.var(returns["VIX"])
        betas[ativo] = cov / var

betas_df = pd.DataFrame.from_dict(betas, orient='index', columns=["Beta_VIX"])
fig_beta, ax_beta = plt.subplots()
betas_df.sort_values("Beta_VIX").plot(kind='barh', ax=ax_beta, legend=False)
ax_beta.axvline(0, color='black', linestyle='--')
ax_beta.set_title("Beta dos Ativos em Relação ao VIX")
ax_beta.grid(True)
st.pyplot(fig_beta)

# Interpretação\st.markdown("""
**Interpretação:**
- Betas positivos indicam ativos que sobem com o aumento da volatilidade (como o ouro e o dólar).
- Betas negativos indicam ativos que caem quando há stress (ações indianas, por exemplo).
- O monitoramento contínuo desses indicadores ajuda a antecipar movimentos de portfólio frente a escaladas geopolíticas.
""")
