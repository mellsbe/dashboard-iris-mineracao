# Dashboard Iris — Mineração de Dados

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dashboard-iris-mineracao-76awtccz8htcftxuzkmb3z.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Aplicação web interativa desenvolvida como parte da avaliação prática da disciplina de Mineração de Dados, focada na exploração visual, classificação supervisionada e clusterização não supervisionada do clássico *Dataset Iris*.

---

## Equipe e Contexto Acadêmico

* **Instituição:** Universidade Federal do Pará (UFPA)
* **Campus:** Campus Universitário de Tucuruí
* **Faculdade:** Faculdade de Engenharia de Computação
* **Disciplina:** Mineração de Dados
* **Docente:** Prof. Dr. Iago Medeiros
* **Discentes (Autores):**
  * Beatriz Silva de Melo
  * Izadora Cunha
  * Rafael Vicente
  * Thiago Teixeira

---

## Objetivo do Projeto

O objetivo principal deste dashboard é demonstrar de forma visual e interativa o pipeline completo de análise de dados (EDA, Classificação e Agrupamento). Como o conjunto de dados Iris é amplamente conhecido e livre de ruídos ou valores ausentes, o foco do trabalho (**Opção B**) concentra-se na excelência da **apresentação e comunicação de resultados**, permitindo que o usuário manipule hiperparâmetros de modelos de *Machine Learning* e veja as respostas em tempo real.

## Funcionalidades e Estrutura do Dashboard

A aplicação está organizada de forma modular através de uma barra lateral de navegação:

1. **Visão Geral:** Apresentação resumida do dataset, principais cartões de métricas estatísticas (amostras, atributos, classes, dados ausentes), amostragem das primeiras linhas e a distribuição volumétrica das espécies.
2. **Análise Exploratória de Dados (EDA):** Gráficos dinâmicos alimentados pela biblioteca Plotly:
   * Histogramas de atributos com distribuições marginais (*boxplots* integrados).
   * Gráficos de dispersão customizáveis bidimensionais (Eixo X vs. Eixo Y).
   * Matriz de dispersão (*Scatter Matrix*) englobando todas as combinações de atributos.
   * Mapa de calor de correlação linear (Pearson).
   * *Boxplots* das características segregados por espécie.
3. **Classificação Supervisionada:** Módulo de treinamento dinâmico com divisão configurável da proporção da base de teste via *slider*:
   * **Modelos Disponíveis:** K-Nearest Neighbors (KNN), Árvore de Decisão e Random Forest.
   * **Ajuste de Hiperparâmetros ao Vivo:** Quantidade de vizinhos (k), profundidade máxima da árvore e número de estimadores.
   * **Métricas de Avaliação:** Exibição da acurácia global, Matriz de Confusão dinâmica e tabela com o relatório completo de classificação (*Precision*, *Recall*, *F1-Score*).
   * **Importância de Atributos:** Gráfico horizontal exibindo quais variáveis exerceram maior peso na tomada de decisão (para modelos baseados em árvores).
   * **Simulador Prático:** Painel interativo para inserção de medidas customizadas (via sliders) que retorna imediatamente a predição da espécie correspondente com base no modelo ajustado.
4. **Clusterização Não Supervisionada:** Agrupamento "cego" utilizando o algoritmo **K-Means** (com escolha dinâmica de k entre 2 e 6):
   * **Visualização Espacial via PCA:** Redução de dimensionalidade linear (Análise de Componentes Principais) de 4 para 2 dimensões para plotagem lado a lado dos clusters encontrados vs. as espécies reais.
   * **Métricas de Validação de Cluster:** Cálculo em tempo real do *Silhouette Score* e do *Adjusted Rand Index* (ARI).
   * **Método do Cotovelo (Elbow Method):** Gráfico de linha demonstrando a evolução da inércia para guiar a escolha ideal do número de partições.
   * **Tabela Cruzada (*Crosstab*):** Matriz que confronta matematicamente o agrupamento predito pelo cluster com a etiqueta biológica real da flor.

---

## Como Executar o Projeto Localmente

### 1. Clonar o Repositório
```bash
git clone [https://github.com/mellsbe/dashboard-iris-mineracao.git](https://github.com/mellsbe/dashboard-iris-mineracao.git)
cd dashboard-iris-mineracao
```

### 2. Base de Dados Local (Antes de rodar)
O app lê o arquivo **"iris dataset.xls"** de dentro da própria pasta do projeto. Certifique-se de copiar o arquivo de dados para a raiz do diretório, mantendo a seguinte estrutura de arquivos:

```text
dashboard-iris-mineracao/
├── app.py
├── requirements.txt
├── README.md
└── iris dataset.xls   <- copie o arquivo para cá
```

*(Nota: Caso o arquivo de dados local não esteja presente na pasta, o script possui uma rotina de fallback automático que carrega o dataset embutido da biblioteca scikit-learn para garantir o funcionamento básico da aplicação).*

### 3. Configurar Ambiente Virtual e Dependências
Recomenda-se isolar o ambiente de execução local para evitar conflitos de versões:
```bash
# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
venv\Scripts\activate

# Ativar o ambiente (Linux/Mac)
source venv/bin/activate

# Instalar as bibliotecas requeridas
pip install -r requirements.txt
```

### 4. Inicializar o Servidor Streamlit
```bash
streamlit run app.py
```
A plataforma iniciará o servidor local e abrirá automaticamente uma nova aba no seu navegador web padrão apontando para `http://localhost:8501`.

---

## Stack Tecnológica

* **Linguagem Principal:** Python 3.9+
* **Framework Web / UI:** Streamlit (v1.30+)
* **Estruturação e Manipulação:** Pandas & NumPy
* **Core de Inteligência Artificial:** Scikit-Learn
* **Engenharia de Visualização:** Plotly Open Source (Gráficos interativos inter-relacionados)
