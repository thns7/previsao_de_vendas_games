import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preparacao import limpar_base, base_de_modelagem

st.set_page_config(
    page_title="Fusca Games | Previsão de vendas de video games",
    layout="wide",
)

# ---------------------------------------------------------------
# Carregamento (cacheado) dos dados e do modelo treinado
# ---------------------------------------------------------------
@st.cache_data
def carregar_dados():
    """Carrega a base bruta e aplica exatamente a mesma limpeza do notebook (seção 3.3)."""
    df_bruto = pd.read_csv('dados/vgsales_bruto.csv')
    df_tratado = limpar_base(df_bruto)
    df_modelagem = base_de_modelagem(df_tratado)
    return df_tratado, df_modelagem


@st.cache_resource
def carregar_modelo():
    with open('modelo/modelo.pkl', 'rb') as f:
        artefato = pickle.load(f)
    return artefato


df, df_modelagem = carregar_dados()
artefato = carregar_modelo()
modelo = artefato['modelo']
metricas = artefato['metricas']
intervalos = artefato['intervalos']
generos = artefato['generos']

# ---------------------------------------------------------------
# Cabeçalho e descrição do projeto
# ---------------------------------------------------------------
st.title("Previsão de vendas globais de video games")
st.write(
    "Aplicação interativa do projeto de regressão linear. O modelo estima as **vendas globais "
    "de um video game** (em milhões de unidades) a partir da nota da crítica, da nota dos "
    "usuários, da quantidade de avaliações e do gênero do jogo."
)
st.caption(
    "Fonte dos dados: *Video Game Sales with Ratings* (Kaggle, usuário Rush Kirubi) — "
    "https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings · "
    "Modelo final: regressão linear múltipla (seção 3.6.3 do notebook)."
)

st.subheader("1. Variáveis do modelo")
col_y, col_x = st.columns(2)
with col_y:
    st.markdown("**Variável resposta**")
    st.markdown("- `Global_Sales` — vendas globais (milhões de unidades)")
with col_x:
    st.markdown("**Variáveis explicativas**")
    st.markdown(
        "- `Critic_Score` (nota da crítica, 0-100)\n"
        "- `User_Score` (nota dos usuários, 0-10)\n"
        "- `Critic_Count` (nº de críticos)\n"
        "- `User_Count` (nº de usuários)\n"
        "- `Genre` (gênero do jogo)"
    )

# ---------------------------------------------------------------
# Amostra da base e estatísticas descritivas
# ---------------------------------------------------------------
st.subheader("2. Amostra da base e estatísticas descritivas")
st.caption(
    f"Base tratada: {len(df):,} jogos. Destes, {len(df_modelagem):,} têm todas as variáveis do "
    f"modelo preenchidas e formam o conjunto usado no treino e no teste "
    f"({artefato['n_treino']:,} + {artefato['n_teste']:,} jogos)."
    .replace(',', '.')
)

with st.expander("Ver amostra da base tratada"):
    st.dataframe(df.sample(10, random_state=1), width='stretch')

with st.expander("Ver estatísticas descritivas (variáveis do modelo)"):
    st.dataframe(
        df_modelagem[['Global_Sales', 'Critic_Score', 'User_Score',
                      'Critic_Count', 'User_Count']].describe().round(2),
        width='stretch',
    )

# ---------------------------------------------------------------
# Gráficos exploratórios (pelo menos 2, conforme exigido)
# ---------------------------------------------------------------
st.subheader("3. Gráficos exploratórios")

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.hist(df['Global_Sales'], bins=60, range=(0, 10), color='#4C72B0', edgecolor='white')
    ax1.set_title('Distribuição das vendas globais')
    ax1.set_xlabel('Vendas globais (milhões de unidades)')
    ax1.set_ylabel('Número de jogos')
    st.pyplot(fig1)
    plt.close(fig1)
    st.caption(
        "A maioria dos jogos vende menos de 1 milhão de unidades; poucos títulos vendem muito "
        "(eixo recortado em 10 milhões para leitura). É a assimetria discutida na seção 3.4.2 "
        "do notebook."
    )

with col_g2:
    vendas_por_genero = df.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(vendas_por_genero.index, vendas_por_genero.values, color='#8172B2')
    ax2.set_title('Vendas médias por gênero')
    ax2.set_xlabel('Gênero')
    ax2.set_ylabel('Vendas globais médias (milhões)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption(
        "Médias simples, sem controlar nota nem volume de avaliações — no modelo múltiplo, com "
        "essas variáveis controladas, o ranking muda (seção 3.6.3 do notebook)."
    )

# ---------------------------------------------------------------
# Métricas do modelo final
# ---------------------------------------------------------------
st.subheader("4. Desempenho do modelo final (regressão múltipla)")
st.caption("Todas as métricas abaixo são calculadas no **conjunto de teste**, não visto no treino.")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("MAE", f"{metricas['MAE']:.3f} milhões")
col_m2.metric("RMSE", f"{metricas['RMSE']:.3f} milhões")
col_m3.metric("R²", f"{metricas['R2']:.3f}")

col_d1, col_d2 = st.columns(2)
y_test = artefato['y_test']
y_pred_teste = artefato['y_pred_teste']

with col_d1:
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(y_test, y_pred_teste, alpha=0.3, s=12, color='#4C72B0')
    lim = max(y_test.max(), y_pred_teste.max())
    ax3.plot([0, lim], [0, lim], color='black', linestyle='--', label='Previsão perfeita')
    ax3.set_title('Valores reais versus previstos (teste)')
    ax3.set_xlabel('Vendas reais (milhões)')
    ax3.set_ylabel('Vendas previstas (milhões)')
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

with col_d2:
    residuos_teste = y_test - y_pred_teste
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    ax4.scatter(y_pred_teste, residuos_teste, alpha=0.3, s=12, color='#C44E52')
    ax4.axhline(0, color='black', linestyle='--')
    ax4.set_title('Resíduos versus valores ajustados (teste)')
    ax4.set_xlabel('Valores ajustados (milhões)')
    ax4.set_ylabel('Resíduo (real - previsto)')
    st.pyplot(fig4)
    plt.close(fig4)

st.caption(
    "Em média, a previsão do modelo erra em torno do valor do MAE acima, para mais ou para "
    "menos. O modelo tende a **subestimar** jogos com vendas muito altas (grandes sucessos) — "
    "veja a discussão completa no notebook, seção 3.8."
)

# ---------------------------------------------------------------
# Formulário de previsão
# ---------------------------------------------------------------
st.subheader("5. Faça uma previsão")

with st.form("formulario_previsao"):
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        critic_score = st.slider(
            "Nota da crítica (Critic Score, 0-100)",
            min_value=0, max_value=100, value=75,
        )
        critic_count = st.number_input(
            "Número de críticos que avaliaram",
            min_value=0, value=30, step=1,
        )
        genero = st.selectbox("Gênero do jogo", options=generos)

    with col_f2:
        user_score = st.slider(
            "Nota dos usuários (User Score, 0-10)",
            min_value=0.0, max_value=10.0, value=7.5, step=0.1,
        )
        user_count = st.number_input(
            "Número de usuários que avaliaram",
            min_value=0, value=200, step=1,
        )

    enviado = st.form_submit_button("Prever vendas globais")

if enviado:
    entrada = pd.DataFrame([{
        'Critic_Score': critic_score,
        'User_Score': user_score,
        'Critic_Count': critic_count,
        'User_Count': user_count,
        'Genre': genero,
    }])

    previsao = float(modelo.predict(entrada)[0])
    previsao_exibida = max(previsao, 0.0)

    st.success(f"**Previsão de vendas globais: {previsao_exibida:.2f} milhões de unidades**")

    # A regressão linear não tem restrição de sinal e pode prever valores negativos
    # (~10% do conjunto de teste — seção 3.8.7 do notebook). Truncamos em zero e avisamos,
    # em vez de esconder o que aconteceu.
    if previsao < 0:
        st.info(
            f"ℹ️ O modelo calculou **{previsao:.2f}** milhão de unidades. Como vendas negativas "
            "não existem, o valor foi truncado em zero — leia o resultado como *vendas próximas "
            "de zero*. Isso acontece porque a regressão linear não tem restrição de sinal, e é "
            "mais frequente em jogos com nota baixa e poucas avaliações (seção 3.8.7 do notebook)."
        )

    # Aviso de extrapolação: entrada fora do intervalo observado no treino
    fora_do_intervalo = []
    valores_entrada = {
        'Critic_Score': critic_score,
        'User_Score': user_score,
        'Critic_Count': critic_count,
        'User_Count': user_count,
    }
    for coluna, valor in valores_entrada.items():
        minimo, maximo = intervalos[coluna]
        if valor < minimo or valor > maximo:
            fora_do_intervalo.append(
                f"`{coluna}` = {valor} (intervalo observado: {minimo:.1f} a {maximo:.1f})"
            )

    if fora_do_intervalo:
        st.warning(
            "⚠️ Uma ou mais entradas estão **fora do intervalo observado** nos dados de treino "
            "do modelo. A previsão é uma extrapolação e não tem garantia estatística:\n\n"
            + "\n".join(f"- {msg}" for msg in fora_do_intervalo)
        )
