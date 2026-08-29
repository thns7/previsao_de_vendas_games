# Previsão de vendas globais de video games — Projeto de Regressão Linear

## Objetivo do projeto

Desenvolver um projeto de regressão linear com uma base pública e real, documentando as decisões
de preparação dos dados, modelagem, avaliação e comunicação dos resultados, com aplicação
interativa em Streamlit.

**Pergunta de pesquisa:** em que medida a nota da crítica especializada, a nota dos usuários, o
volume de avaliações recebidas e o gênero ajudam a explicar as vendas globais de um video game?

- **Repositório:** <PREENCHER: https://github.com/usuario/repositorio>
- **Aplicação publicada:** <PREENCHER: https://....streamlit.app>
- **Modelo final:** regressão linear múltipla — MAE 0,709 · RMSE 1,327 · R² 0,220 (conjunto de teste)

## Origem dos dados

- **Nome da base:** Video Game Sales with Ratings
- **Fonte:** Kaggle (usuário Rush Kirubi) — <https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings>
- **Conteúdo:** 16.719 jogos lançados entre 1980 e 2016, com vendas por região, nota de crítica e
  de usuários (Metacritic), gênero, plataforma, publisher/developer e classificação etária.
- O arquivo bruto (sem nenhum tratamento) está em `dados/vgsales_bruto.csv`. Toda a limpeza é
  feita de forma reproduzível dentro do `notebook.ipynb` (seção 3.3) e replicada no
  `train_model.py`.

## Estrutura do repositório

```
projeto/
|-- app.py               # Aplicação Streamlit
|-- preparacao.py        # Limpeza da base (seção 3.3), compartilhada por train_model.py e app.py
|-- train_model.py       # Script que treina e salva o modelo usado pelo app.py
|-- notebook.ipynb       # Análise completa: proposição do problema, wrangling, EDA,
|                         # modelagem, avaliação, diagnóstico e conclusão
|-- requirements.txt     # Dependências do projeto (notebook + app)
|-- README.md            # Este arquivo
|-- dados/
|   `-- vgsales_bruto.csv    # Base bruta, sem tratamento
`-- modelo/
    `-- modelo.pkl            # Modelo final (regressão múltipla) já treinado
```

> **Importante:** os arquivos `dados/vgsales_bruto.csv` e `modelo/modelo.pkl` precisam estar
> nessas pastas. O notebook e o `app.py` usam esses caminhos relativos e só funcionam se
> executados a partir da raiz do repositório.

## Instalação das dependências

```bash
pip install -r requirements.txt
```

## Como executar o notebook

Abra `notebook.ipynb` no Jupyter, VS Code (com a extensão Jupyter) ou Google Colab, e execute as
células em ordem, a partir da raiz do repositório (o notebook espera encontrar
`dados/vgsales_bruto.csv` no caminho relativo `dados/`).

```bash
jupyter notebook notebook.ipynb
```

## Como (re)treinar o modelo

O modelo usado pelo `app.py` já está salvo em `modelo/modelo.pkl`. Caso precise treiná-lo
novamente (por exemplo, depois de alguma alteração na limpeza dos dados), rode, a partir da raiz
do repositório:

```bash
python train_model.py
```

Isso recria `modelo/modelo.pkl` com o pré-processamento e o modelo final (regressão múltipla)
usados também no notebook. A limpeza da base vive em um único arquivo (`preparacao.py`),
importado tanto pelo `train_model.py` quanto pelo `app.py`: notebook, treino e aplicação usam
obrigatoriamente a mesma base tratada e o mesmo pré-processamento, sem regra duplicada.

Reproduzindo o treino, os números devem bater exatamente com os do notebook:
`MAE=0.7089  RMSE=1.3271  R2=0.2198`, com 4.911 linhas de treino e 2.105 de teste.

## Como executar a aplicação Streamlit

```bash
streamlit run app.py
```

A aplicação permite explorar a base, ver o desempenho do modelo final (MAE, RMSE, R²) e gerar
previsões de vendas globais a partir de valores informados pelo usuário para nota da crítica,
nota dos usuários, número de avaliações e gênero.

## Publicação

A aplicação está publicada no Streamlit Community Cloud:
**<PREENCHER: https://....streamlit.app>**

Para publicar (ou republicar): entre em <https://share.streamlit.io>, conecte a conta do GitHub,
selecione este repositório, a branch e o arquivo `app.py`. As dependências são instaladas
automaticamente a partir do `requirements.txt`.

## Principais limitações conhecidas

- Apenas ~42% da base tinha nota de crítica e de usuários preenchidas simultaneamente; jogos sem
  essas notas (geralmente mais antigos ou de nicho) não entraram no treino/teste do modelo.
- A base vai até 2016 e não reflete o mercado atual (jogos free-to-play, DLCs, assinaturas etc.).
- O modelo final explica cerca de 22% da variância das vendas (R² ≈ 0,22) — a maior parte da
  variação depende de fatores não presentes na base, como orçamento de marketing e força de
  franquia.
- O modelo tende a **subestimar sistematicamente** os grandes sucessos de vendas ("hits"), dado o
  comportamento fortemente assimétrico da variável resposta (ver seção 3.8 do notebook).
- O modelo **não deve ser usado para decisões de pré-lançamento**, já que depende de notas de
  crítica/usuários que só existem depois que o jogo é lançado e avaliado.
- Cerca de **10% das previsões do conjunto de teste são negativas** (vendas negativas não
  existem): a regressão linear não tem restrição de sinal. Na aplicação, esses valores são
  truncados em zero e o usuário é avisado; no notebook, o fenômeno é medido e discutido na
  seção 3.8.7.

Discussão completa de todas as decisões, hipóteses e limitações está no `notebook.ipynb`.
