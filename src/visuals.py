import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração estética dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# --- GRÁFICO 1: TOP 10 FAVORITOS À COPA 2026 ---
df_ranking = pd.read_csv('outputs/ranking_probabilidades.csv')
top_10 = df_ranking.head(10)

plt.figure()
ax = sns.barplot(
    x='probabilidade_campeao_pct', 
    y='team_name', 
    data=top_10, 
    palette='viridis'
)
plt.title('Top 10 Seleções com Maior Probabilidade - Copa 2026', fontsize=14, pad=15)
plt.xlabel('Probabilidade de Vitória (%)', fontsize=12)
plt.ylabel('Seleção', fontsize=12)

# Adicionar os valores percentuais nas barras
for i, v in enumerate(top_10['probabilidade_campeao_pct']):
    ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=11, fontweight='bold')

plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('outputs/grafico_top_10_favoritos.png', dpi=300)
print("Gráfico do Top 10 gerado com sucesso!")

# --- GRÁFICO 2 ---
df_importance = pd.read_csv('outputs/feature_importance.csv')
top_features = df_importance.head(8) # Pegar as 8 que mais pesaram

plt.figure()
sns.barplot(
    x='importancia', 
    y='variavel', 
    data=top_features, 
    palette='mako'
)
plt.title('Drivers do Modelo: Quais atributos pesaram mais?', fontsize=14, pad=15)
plt.xlabel('Grau de Importância (Gini Importance)', fontsize=12)
plt.ylabel('Variáveis do Dataset', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/grafico_feature_importance.png', dpi=300)
print("Gráfico de Importância das Variáveis gerado com sucesso!")
