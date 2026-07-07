"""
Dashboard Iris — Trabalho 7 (Mineração de Dados)
Opção B: apresentação de resultados (exploração, classificação e clusterização)

Como rodar:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

st.set_page_config(
    page_title="Dashboard Iris — Mineração de Dados",
    layout="wide",
)

# Configuração do arquivo de dados local
NOME_ARQUIVO_LOCAL = "iris dataset.xls"
CAMINHO_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOME_ARQUIVO_LOCAL)


def normalizar_colunas(df_bruto):
    """Detecta e renomeia colunas para o padrão usado no dashboard,
    aceitando variações comuns de nomenclatura do dataset Iris."""
    df_bruto = df_bruto.copy()

    for col_id in ["Id", "id", "ID", "Unnamed: 0"]:
        if col_id in df_bruto.columns:
            df_bruto = df_bruto.drop(columns=[col_id])

    mapa_colunas = {}
    for col in df_bruto.columns:
        chave = str(col).strip().lower().replace(" ", "").replace("_", "").replace(".", "").replace("(cm)", "").replace("cm", "")
        if "sepal" in chave and "length" in chave:
            mapa_colunas[col] = "comprimento_sepala"
        elif "sepal" in chave and "width" in chave:
            mapa_colunas[col] = "largura_sepala"
        elif "petal" in chave and "length" in chave:
            mapa_colunas[col] = "comprimento_petala"
        elif "petal" in chave and "width" in chave:
            mapa_colunas[col] = "largura_petala"
        elif chave in ["species", "class", "variety", "target", "name"]:
            mapa_colunas[col] = "species"

    df_bruto = df_bruto.rename(columns=mapa_colunas)

    colunas_esperadas = ["comprimento_sepala", "largura_sepala", "comprimento_petala", "largura_petala", "species"]
    faltando = [c for c in colunas_esperadas if c not in df_bruto.columns]
    if faltando:
        st.error(
            f"Não foi possível reconhecer automaticamente as colunas {faltando} no arquivo. "
            f"Colunas encontradas: {list(df_bruto.columns)}. "
            "Avise para eu ajustar o mapeamento de colunas."
        )
        st.stop()

    df_bruto = df_bruto[colunas_esperadas]

    if pd.api.types.is_numeric_dtype(df_bruto["species"]):
        mapa_numerico = dict(enumerate(load_iris().target_names))
        df_bruto["species"] = df_bruto["species"].map(mapa_numerico)
    else:
        df_bruto["species"] = (
            df_bruto["species"].astype(str).str.strip().str.lower().str.replace("iris-", "", regex=False)
        )

    return df_bruto


def tentar_ler_arquivo(caminho):
    """Tenta ler o arquivo com vários métodos, já que a extensão .xls nem
    sempre corresponde ao formato real do conteúdo. Retorna (df, None) em
    caso de sucesso ou (None, mensagem_de_erro) se todos falharem."""
    erros = []

    # 1) Excel binário antigo (.xls real)
    try:
        return pd.read_excel(caminho, engine="xlrd"), None
    except Exception as e:
        erros.append(f"xlrd: {e}")

    # 2) Excel moderno (.xlsx renomeado como .xls)
    try:
        return pd.read_excel(caminho, engine="openpyxl"), None
    except Exception as e:
        erros.append(f"openpyxl: {e}")

    # 3) CSV com separador vírgula
    for encoding in ["utf-8", "latin1"]:
        try:
            return pd.read_csv(caminho, encoding=encoding), None
        except Exception as e:
            erros.append(f"csv(sep=',', {encoding}): {e}")

    # 4) CSV com separador ponto e vírgula (comum em arquivos exportados no Brasil)
    for encoding in ["utf-8", "latin1"]:
        try:
            return pd.read_csv(caminho, sep=";", encoding=encoding), None
        except Exception as e:
            erros.append(f"csv(sep=';', {encoding}): {e}")

    # 5) Tab-separated
    try:
        return pd.read_csv(caminho, sep="\t"), None
    except Exception as e:
        erros.append(f"tsv: {e}")

    return None, " | ".join(erros)


def mostrar_tabela_estilizada(df_tabela):
    """Mostra a tabela com gradiente de cor; se o matplotlib não estiver
    instalado, mostra a tabela normal em vez de quebrar o app."""
    try:
        st.dataframe(df_tabela.style.background_gradient(cmap="Purples"), width='stretch')
    except ImportError:
        st.dataframe(df_tabela, width='stretch')


@st.cache_data
def carregar_dados():
    if os.path.exists(CAMINHO_LOCAL):
        df_bruto, erro_leitura = tentar_ler_arquivo(CAMINHO_LOCAL)
        if df_bruto is None:
            st.error(
                f"Não consegui ler '{NOME_ARQUIVO_LOCAL}' com nenhum método testado.\n\n"
                f"Detalhe técnico: {erro_leitura}\n\n"
                "O arquivo pode estar corrompido, protegido por senha, ou não ser realmente "
                "um Excel/CSV válido."
            )
            st.stop()
        fonte = f"Arquivo local: {NOME_ARQUIVO_LOCAL}"
    else:
        st.warning(
            f"Arquivo '{NOME_ARQUIVO_LOCAL}' não encontrado na pasta do projeto. "
            "Usando o dataset Iris embutido do scikit-learn como alternativa."
        )
        iris = load_iris(as_frame=True)
        df_bruto = iris.frame.copy()
        df_bruto["species"] = df_bruto["target"].map(dict(enumerate(iris.target_names)))
        df_bruto = df_bruto.drop(columns=["target"])
        fonte = "Dataset embutido do scikit-learn (fallback)"

    df = normalizar_colunas(df_bruto)
    return df, fonte


df, fonte_dados = carregar_dados()
features = ["comprimento_sepala", "largura_sepala", "comprimento_petala", "largura_petala"]
nomes_bonitos = {
    "comprimento_sepala": "Comprimento da Sépala (cm)",
    "largura_sepala": "Largura da Sépala (cm)",
    "comprimento_petala": "Comprimento da Pétala (cm)",
    "largura_petala": "Largura da Pétala (cm)",
}
cores_especies = {"setosa": "#6C5CE7", "versicolor": "#00B894", "virginica": "#E17055"}

# Sidebar
st.sidebar.title("Dashboard Iris")
st.sidebar.markdown("**Trabalho 7 — Mineração de Dados**")
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Análise Exploratória", "Classificação", "Clusterização"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: Iris — {len(df)} amostras, {df['species'].nunique()} espécies, 4 atributos.")
st.sidebar.caption(f"Fonte: {fonte_dados}")

# PÁGINA 1 — Visão Geral
if pagina == "Visão Geral":
    st.title("Dashboard de Análise — Dataset Iris")
    st.markdown(
        "Este dashboard apresenta uma análise completa do clássico **dataset Iris**, "
        "cobrindo exploração dos dados, classificação supervisionada e clusterização não supervisionada."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Amostras", len(df))
    col2.metric("Atributos", len(features))
    col3.metric("Classes", df["species"].nunique())
    col4.metric("Valores ausentes", int(df[features].isna().sum().sum()))

    st.markdown("### Amostra dos dados")
    st.dataframe(df.head(10), width='stretch')

    st.markdown("### Estatísticas descritivas")
    mostrar_tabela_estilizada(df[features].describe().T)

    st.markdown("### Distribuição das espécies")
    contagem = df["species"].value_counts().reset_index()
    contagem.columns = ["species", "count"]
    fig = px.bar(
        contagem, x="species", y="count", color="species",
        color_discrete_map=cores_especies, text="count",
        labels={"species": "Espécie", "count": "Quantidade"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.info(
        "O dataset Iris não possui valores ausentes nem inconsistências — "
        "por isso este trabalho foca em **apresentação e visualização de resultados**, "
        "conforme a Opção B da proposta."
    )

# PÁGINA 2 — Análise Exploratória
elif pagina == "Análise Exploratória":
    st.title("Análise Exploratória de Dados (EDA)")

    st.markdown("### Distribuição de cada atributo")
    atributo = st.selectbox("Escolha um atributo", features, format_func=lambda x: nomes_bonitos[x])
    fig_hist = px.histogram(
        df, x=atributo, color="species", marginal="box", nbins=25,
        color_discrete_map=cores_especies,
        labels={atributo: nomes_bonitos[atributo], "species": "Espécie"},
    )
    st.plotly_chart(fig_hist, width='stretch')

    st.markdown("### Dispersão entre dois atributos")
    c1, c2 = st.columns(2)
    with c1:
        eixo_x = st.selectbox("Eixo X", features, index=2, format_func=lambda x: nomes_bonitos[x])
    with c2:
        eixo_y = st.selectbox("Eixo Y", features, index=3, format_func=lambda x: nomes_bonitos[x])

    fig_scatter = px.scatter(
        df, x=eixo_x, y=eixo_y, color="species", symbol="species",
        color_discrete_map=cores_especies, size_max=10,
        labels={eixo_x: nomes_bonitos[eixo_x], eixo_y: nomes_bonitos[eixo_y], "species": "Espécie"},
    )
    st.plotly_chart(fig_scatter, width='stretch')

    st.markdown("### Matriz de dispersão (todos os atributos)")
    fig_matrix = px.scatter_matrix(
        df, dimensions=features, color="species",
        color_discrete_map=cores_especies,
        labels=nomes_bonitos,
    )
    fig_matrix.update_layout(height=700)
    st.plotly_chart(fig_matrix, width='stretch')

    st.markdown("### Mapa de correlação")
    corr = df[features].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="Purples",
        x=[nomes_bonitos[f] for f in features], y=[nomes_bonitos[f] for f in features],
    )
    st.plotly_chart(fig_corr, width='stretch')

    st.markdown("### Boxplots por espécie")
    atributo_box = st.selectbox("Atributo para o boxplot", features, key="box", format_func=lambda x: nomes_bonitos[x])
    fig_box = px.box(
        df, x="species", y=atributo_box, color="species",
        color_discrete_map=cores_especies,
        labels={"species": "Espécie", atributo_box: nomes_bonitos[atributo_box]},
    )
    st.plotly_chart(fig_box, width='stretch')

# PÁGINA 3 — Classificação
elif pagina == "Classificação":
    st.title("Classificação Supervisionada")
    st.markdown(
        "Treinamos modelos para prever a **espécie** de uma flor com base nas quatro medidas. "
        "Escolha o modelo e a proporção de teste."
    )

    col1, col2 = st.columns(2)
    with col1:
        modelo_nome = st.selectbox("Modelo", ["KNN", "Árvore de Decisão", "Random Forest"])
    with col2:
        test_size = st.slider("Proporção de teste", 0.1, 0.5, 0.3, 0.05)

    X = df[features]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if modelo_nome == "KNN":
        k = st.slider("Número de vizinhos (k)", 1, 15, 5)
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train_s, y_train)
        y_pred = modelo.predict(X_test_s)
    elif modelo_nome == "Árvore de Decisão":
        profundidade = st.slider("Profundidade máxima", 1, 10, 3)
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    else:
        n_arvores = st.slider("Número de árvores", 10, 200, 100, 10)
        modelo = RandomForestClassifier(n_estimators=n_arvores, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.markdown("### Resultado")
    c1, c2 = st.columns(2)
    c1.metric("Acurácia no teste", f"{acc:.1%}")
    c2.metric("Amostras de teste", len(y_test))

    labels_ordenados = sorted(df["species"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_ordenados)
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        x=labels_ordenados, y=labels_ordenados,
        labels=dict(x="Previsto", y="Real", color="Contagem"),
    )
    st.markdown("### Matriz de confusão")
    st.plotly_chart(fig_cm, width='stretch')

    st.markdown("### Relatório de classificação")
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    mostrar_tabela_estilizada(pd.DataFrame(relatorio).T)

    if modelo_nome in ["Árvore de Decisão", "Random Forest"]:
        st.markdown("### Importância dos atributos")
        importancias = pd.DataFrame({
            "Atributo": [nomes_bonitos[f] for f in features],
            "Importância": modelo.feature_importances_,
        }).sort_values("Importância", ascending=True)
        fig_imp = px.bar(importancias, x="Importância", y="Atributo", orientation="h", color="Importância", color_continuous_scale="Purples")
        st.plotly_chart(fig_imp, width='stretch')

    st.markdown("---")
    st.markdown("### Testar uma nova flor")
    st.caption("Ajuste os valores e veja a previsão do modelo treinado acima.")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        v1 = st.slider(nomes_bonitos["comprimento_sepala"], float(df.comprimento_sepala.min()), float(df.comprimento_sepala.max()), float(df.comprimento_sepala.mean()))
    with cc2:
        v2 = st.slider(nomes_bonitos["largura_sepala"], float(df.largura_sepala.min()), float(df.largura_sepala.max()), float(df.largura_sepala.mean()))
    with cc3:
        v3 = st.slider(nomes_bonitos["comprimento_petala"], float(df.comprimento_petala.min()), float(df.comprimento_petala.max()), float(df.comprimento_petala.mean()))
    with cc4:
        v4 = st.slider(nomes_bonitos["largura_petala"], float(df.largura_petala.min()), float(df.largura_petala.max()), float(df.largura_petala.mean()))

    nova_amostra = pd.DataFrame([[v1, v2, v3, v4]], columns=features)
    if modelo_nome == "KNN":
        previsao = modelo.predict(scaler.transform(nova_amostra))[0]
    else:
        previsao = modelo.predict(nova_amostra)[0]
    st.success(f"Espécie prevista: **{previsao}**")

# PÁGINA 4 — Clusterização
elif pagina == "Clusterização":
    st.title("Clusterização (K-Means)")
    st.markdown(
        "Aqui aplicamos **K-Means** ignorando os rótulos reais das espécies, "
        "para ver se o algoritmo consegue redescobrir os grupos naturais dos dados."
    )

    k = st.slider("Número de clusters (k)", 2, 6, 3)

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_cluster = df.copy()
    df_cluster["cluster"] = clusters.astype(str)

    sil = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(df["species"], clusters)

    c1, c2 = st.columns(2)
    c1.metric("Silhouette Score", f"{sil:.3f}")
    c2.metric("Adjusted Rand Index (vs. espécie real)", f"{ari:.3f}")

    st.markdown("### Visualização via PCA (2 componentes)")
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled)
    df_cluster["pca1"] = componentes[:, 0]
    df_cluster["pca2"] = componentes[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        fig_cluster = px.scatter(
            df_cluster, x="pca1", y="pca2", color="cluster",
            title="Clusters encontrados pelo K-Means",
            labels={"pca1": "Componente 1", "pca2": "Componente 2"},
        )
        st.plotly_chart(fig_cluster, width='stretch')
    with col2:
        fig_real = px.scatter(
            df_cluster, x="pca1", y="pca2", color="species",
            color_discrete_map=cores_especies,
            title="Espécies reais",
            labels={"pca1": "Componente 1", "pca2": "Componente 2"},
        )
        st.plotly_chart(fig_real, width='stretch')

    st.markdown("### Método do cotovelo (Elbow Method)")
    inercias = []
    ks = range(1, 10)
    for i in ks:
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)
    fig_elbow = px.line(x=list(ks), y=inercias, markers=True, labels={"x": "Número de clusters (k)", "y": "Inércia"})
    st.plotly_chart(fig_elbow, width='stretch')

    st.markdown("### Tabela cruzada: cluster vs. espécie real")
    tabela_cruzada = pd.crosstab(df_cluster["cluster"], df_cluster["species"])
    mostrar_tabela_estilizada(tabela_cruzada)

    st.info(
        "Compare os dois gráficos de dispersão: quanto mais parecidos, "
        "melhor o K-Means conseguiu redescobrir os grupos reais sem usar os rótulos."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para o Trabalho 7 de Mineração de Dados.")
