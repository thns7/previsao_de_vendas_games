"""
Treina o modelo final (regressão múltipla) usando exatamente o mesmo pré-processamento
do notebook.ipynb, e salva tudo o que o app.py precisa em modelo/modelo.pkl.

Rodar com: python train_model.py
(precisa ser executado a partir da raiz do repositório, com dados/vgsales_bruto.csv presente)
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preparacao import limpar_base, base_de_modelagem, COLUNAS_MODELO, COLUNA_RESPOSTA

# ---------------------------------------------------------------
# 1. Carregamento e limpeza (seção 3.3 do notebook, via preparacao.py)
# ---------------------------------------------------------------
df_bruto = pd.read_csv('dados/vgsales_bruto.csv')
df = limpar_base(df_bruto)

# ---------------------------------------------------------------
# 2. Seleção de variáveis e split treino/teste (seção 3.5 do notebook)
# ---------------------------------------------------------------
df_modelagem = base_de_modelagem(df)

X = df_modelagem[COLUNAS_MODELO]
y = df_modelagem[COLUNA_RESPOSTA]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# ---------------------------------------------------------------
# 3. Pipeline de pré-processamento + modelo final (regressão múltipla, seção 3.6.3)
# ---------------------------------------------------------------
colunas_numericas = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
colunas_categoricas = ['Genre']

preprocessador = ColumnTransformer(transformers=[
    ('num', 'passthrough', colunas_numericas),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), colunas_categoricas),
])

modelo_multiplo = Pipeline(steps=[
    ('preprocessador', preprocessador),
    ('regressor', LinearRegression()),
])
modelo_multiplo.fit(X_train, y_train)

# ---------------------------------------------------------------
# 4. Métricas e dados para os gráficos do app (mesmo conjunto de teste do notebook)
# ---------------------------------------------------------------
y_pred_teste = modelo_multiplo.predict(X_test)

metricas = {
    'MAE': float(mean_absolute_error(y_test, y_pred_teste)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_teste))),
    'R2': float(r2_score(y_test, y_pred_teste)),
}

# Intervalos observados no treino, para o aviso de extrapolação no app
intervalos = {
    col: (float(X_train[col].min()), float(X_train[col].max()))
    for col in colunas_numericas
}

artefato = {
    'modelo': modelo_multiplo,
    'metricas': metricas,
    'y_test': y_test.to_numpy(),
    'y_pred_teste': y_pred_teste,
    'intervalos': intervalos,
    'generos': sorted(X_train['Genre'].unique().tolist()),
    'n_treino': int(X_train.shape[0]),
    'n_teste': int(X_test.shape[0]),
}

with open('modelo/modelo.pkl', 'wb') as f:
    pickle.dump(artefato, f)

print('Modelo treinado e salvo em modelo/modelo.pkl')
print(f"MAE={metricas['MAE']:.4f}  RMSE={metricas['RMSE']:.4f}  R2={metricas['R2']:.4f}")
print(f"Linhas de treino: {X_train.shape[0]}  |  Linhas de teste: {X_test.shape[0]}")
