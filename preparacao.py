"""
Limpeza da base, em um único lugar.

Esta função reproduz exatamente as decisões documentadas na seção 3.3 do `notebook.ipynb`
(duplicados, 'tbd', anos inválidos, categorias de Rating, linhas sem Name/Genre).

Ela é importada tanto por `train_model.py` (que treina o modelo) quanto por `app.py`
(que exibe a base na aplicação), de modo que notebook, treino e aplicação usem
obrigatoriamente a mesma base tratada — sem regras diferentes em cada lugar.
"""
import numpy as np
import pandas as pd

# Variáveis do modelo final (seção 3.5.2 do notebook)
COLUNAS_MODELO = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count', 'Genre']
COLUNA_RESPOSTA = 'Global_Sales'
CHAVE_DUPLICADOS = ['Name', 'Platform', 'Year_of_Release']


def limpar_base(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Aplica os tratamentos da seção 3.3 do notebook e devolve a base tratada."""
    df = df_bruto.copy()

    # 3.3.1 duplicados pela chave do jogo: mantemos o registro de maior Global_Sales
    df = (df.sort_values(COLUNA_RESPOSTA, ascending=False)
            .drop_duplicates(subset=CHAVE_DUPLICADOS, keep='first'))

    # 3.3.2 'tbd' em User_Score é ausência de nota, não um número
    df['User_Score'] = df['User_Score'].replace('tbd', np.nan).astype(float)

    # 3.3.3 anos posteriores a 2016 são impossíveis (base compilada em 12/2016)
    df.loc[df['Year_of_Release'] > 2016, 'Year_of_Release'] = np.nan

    # 3.3.4 'K-A' é o nome antigo de 'E'; 'RP' é ausência de classificação
    df['Rating'] = df['Rating'].replace({'K-A': 'E', 'RP': np.nan})

    # 3.3.5 única remoção de linha: registro órfão sem Name/Genre
    df = df.dropna(subset=['Name', 'Genre'])

    return df


def base_de_modelagem(df_tratado: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto usado no modelo: só linhas completas nas variáveis escolhidas (seção 3.5.2)."""
    return df_tratado[COLUNAS_MODELO + [COLUNA_RESPOSTA]].dropna().copy()
