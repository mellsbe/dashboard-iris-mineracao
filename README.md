# Dashboard Iris — Trabalho 7 de Mineração de Dados

Dashboard interativo em Streamlit apresentando análise exploratória, classificação
e clusterização do dataset Iris.

## Antes de rodar: coloque o arquivo de dados na pasta

O app lê o arquivo **"iris dataset.xls"** de dentro da própria pasta do projeto
(caminho relativo, não o caminho do seu OneDrive). Copie o arquivo:

```
C:\Users\beatr\OneDrive\Mineração de Dados\projeto7\iris dataset.xls
```

para dentro desta pasta (`iris-dashboard`), mantendo o nome exatamente como
`iris dataset.xls`. A estrutura final deve ficar assim:

```
iris-dashboard/
├── app.py
├── requirements.txt
├── README.md
└── iris dataset.xls   <- copie o arquivo para cá
```

Se o app não encontrar o arquivo, ele avisa na tela e usa automaticamente o
dataset Iris embutido do scikit-learn como alternativa (para o dashboard não
quebrar durante testes/apresentação).

O código reconhece automaticamente as colunas mais comuns do dataset Iris
(`sepal length`, `Sepal.Length`, `SepalLengthCm`, `class`, `species`,
`variety`, valores como `Iris-setosa`, etc.). Se o seu arquivo tiver nomes de
coluna muito diferentes, me envie os nomes das colunas (ou as 2 primeiras
linhas) que eu ajusto o mapeamento.

## Como rodar localmente (VSCode)

1. Abra a pasta no VSCode.
2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Rode o app:
   ```bash
   streamlit run app.py
   ```
5. O navegador vai abrir automaticamente em `http://localhost:8501`.

## Estrutura

```
iris-dashboard/
├── app.py              # aplicação Streamlit (todas as páginas)
├── requirements.txt    # dependências Python
└── README.md
```

## Páginas do dashboard

- **Visão Geral** — resumo do dataset, estatísticas descritivas, distribuição das classes.
- **Análise Exploratória** — histogramas, dispersão, matriz de correlação, boxplots.
- **Classificação** — KNN / Árvore de Decisão / Random Forest, matriz de confusão,
  importância de atributos e simulador de previsão para uma nova flor.
- **Clusterização** — K-Means, PCA 2D, método do cotovelo, silhouette score e
  comparação com as espécies reais.

## Como colocar no ar (hospedagem gratuita)

1. Crie um repositório no GitHub e suba estes arquivos: `app.py`,
   `requirements.txt`, `README.md` **e também `iris dataset.xls`** (o app
   precisa do arquivo de dados junto dele para funcionar depois de hospedado —
   sem ele, cai automaticamente no dataset do scikit-learn).
2. Acesse **https://share.streamlit.io** (Streamlit Community Cloud) e faça login
   com sua conta do GitHub.
3. Clique em **"New app"**, selecione o repositório, o branch e o arquivo
   principal (`app.py`).
4. Clique em **Deploy**. Em 1–2 minutos você terá uma URL pública tipo
   `https://seu-app.streamlit.app` para colocar no relatório/apresentação do
   trabalho.

## Observações para o relatório do trabalho

- O dataset Iris não tem valores ausentes nem inconsistências, então a limpeza
  não é necessária — o foco (Opção B) é a qualidade da apresentação.
- Print de cada página + a URL pública funcionam bem no relatório escrito.
- Vale citar no relatório: acurácia do melhor modelo, silhouette score do
  K-Means e o Adjusted Rand Index (mostra o quão bem o cluster "acertou" as
  espécies reais sem ver os rótulos).
