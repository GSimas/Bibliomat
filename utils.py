import pandas as pd
import rispy
import io
import re
import networkx as nx
import numpy as np
import scipy.stats as stats
from collections import Counter
from itertools import combinations
from streamlit_agraph import Node, Edge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyecharts import options as opts
from pyecharts.charts import WordCloud as PyechartsWordCloud
import json
from pyecharts.commons.utils import JsCode
import random
import streamlit as st

# --- FUNÇÕES AUXILIARES DE AGREGAÇÃO PARA TABELAS ---

# --- FUNÇÕES AUXILIARES DE AGREGAÇÃO CORRIGIDAS ---

def _format_timeline(group):
    """Gera a string agrupada por ano garantindo unicidade de documentos no grupo."""
    import pandas as pd
    anos = {}
    # Removemos duplicatas do próprio grupo (caso o mesmo doc apareça 2x por erro de índice)
    group_clean = group.drop_duplicates(subset=['TITLE', 'YEAR CLEAN'])
    
    for _, row in group_clean.iterrows():
        ano = str(int(row['YEAR CLEAN'])) if pd.notna(row.get('YEAR CLEAN')) else "S/D"
        tit = str(row.get('TITLE', 'Sem título')).strip()
        cit = int(row.get('TOTAL CITATIONS', 0)) if pd.notna(row.get('TOTAL CITATIONS')) else 0
        if ano not in anos: anos[ano] = []
        anos[ano].append(f"{tit} ({cit} citações)")
    
    out = []
    for a in sorted(anos.keys(), reverse=True):
        out.append(f"{a}: {'; '.join(anos[a])}")
    return " | ".join(out)

def _get_top_doc(group):
    """Retorna o documento mais citado usando posição (iloc) para evitar erro de índice ambíguo."""
    import pandas as pd
    if group.empty: return ""
    
    # Em vez de idxmax (que retorna o nome do índice), usamos argmax (que retorna a posição)
    posicao_max = group['TOTAL CITATIONS'].fillna(0).values.argmax()
    # iloc[posicao] sempre retorna uma única Series, mesmo com índices duplicados
    row = group.iloc[posicao_max]
    
    tit = str(row.get('TITLE', 'Sem Título')).strip()
    val_cit = row.get('TOTAL CITATIONS', 0)
    cit = int(val_cit) if pd.notna(val_cit) else 0
    
    return f"{tit} ({cit} citações)"

# --- MOTORES DE GERAÇÃO DE TABELAS ---

def gerar_tabela_autores(df):
    import pandas as pd
    df_exp = df.copy()
    df_exp['AUTHOR'] = df_exp['AUTHORS'].astype(str).str.split(';')
    df_exp = df_exp.explode('AUTHOR')
    df_exp['AUTHOR'] = df_exp['AUTHOR'].str.strip().str.title()
    df_exp = df_exp[df_exp['AUTHOR'] != '']

    res = []
    for autor, group in df_exp.groupby('AUTHOR'):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        # Coautores
        coautores = set()
        for auth_list in group['AUTHORS'].dropna():
            coautores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip().title() != autor])
            
        paises = set()
        if 'COUNTRY' in group.columns:
            for c_list in group['COUNTRY'].dropna():
                paises.update([c.strip().title() for c in str(c_list).split(';') if c.strip()])

        res.append({
            'Autor': autor,
            'País do Autor': ", ".join(paises),
            'Documentos': " | ".join(group['TITLE'].dropna().astype(str)),
            'Qtd. de Documentos': len(group),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Coautores': ", ".join(coautores)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False)

def gerar_tabela_paises(df):
    import pandas as pd
    if 'COUNTRY' not in df.columns: return pd.DataFrame()
    
    df_exp = df.copy()
    df_exp['PAIS'] = df_exp['COUNTRY'].astype(str).str.split(';')
    df_exp = df_exp.explode('PAIS')
    df_exp['PAIS'] = df_exp['PAIS'].str.strip().str.title()
    df_exp = df_exp[(df_exp['PAIS'] != '') & (df_exp['PAIS'] != 'Nan')]

    res = []
    for pais, group in df_exp.groupby('PAIS'):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        autores = set()
        for auth_list in group['AUTHORS'].dropna():
            autores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip()])

        res.append({
            'País': pais,
            'Autores': ", ".join(autores),
            'Qtd. de Autores': len(autores),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Documento com Mais Citações': _get_top_doc(group)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False)

def gerar_tabela_venues(df):
    import pandas as pd
    col_venue = 'SECONDARY TITLE'
    if col_venue not in df.columns: return pd.DataFrame()

    df_ven = df.dropna(subset=[col_venue]).copy()
    
    res = []
    for venue, group in df_ven.groupby(col_venue):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        autores = set()
        for auth_list in group['AUTHORS'].dropna():
            autores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip()])

        res.append({
            'Local de Publicação (Venue)': str(venue).upper(),
            'Autores': ", ".join(autores),
            'Qtd. de Autores': len(autores),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Documento com Mais Citações': _get_top_doc(group)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False)

def gerar_tabela_keywords(df):
    import pandas as pd
    if 'KEYWORDS' not in df.columns: return pd.DataFrame()
    
    df_exp = df.copy()
    df_exp['KW'] = df_exp['KEYWORDS'].astype(str).str.split(';')
    df_exp = df_exp.explode('KW')
    df_exp['KW'] = df_exp['KW'].str.strip().str.title()
    df_exp = df_exp[(df_exp['KW'] != '') & (df_exp['KW'] != 'Nan')]

    res = []
    for kw, group in df_exp.groupby('KW'):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        autores = set()
        for auth_list in group['AUTHORS'].dropna():
            autores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip()])

        res.append({
            'Palavra-chave': kw,
            'Autores que usaram': ", ".join(autores),
            'Qtd. de Autores': len(autores),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Documento com Mais Citações': _get_top_doc(group)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False)

def categorizar_temas_por_cluster(df, api_key, max_clusters=10):
    """Clusteriza documentos com TF-IDF + K-Means (otimizado por Silhouette) e nomeia via Gemini."""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import google.generativeai as genai
    import streamlit as st
    import time

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # 1. Preparação: Consolida o texto rico de cada documento
    df_text = df.copy()
    df_text['TEXTO_COMBINADO'] = df_text['TITLE'].fillna('') + " " + \
                                 df_text['KEYWORDS'].fillna('') + " " + \
                                 df_text['ABSTRACT'].fillna('')
    
    textos = df_text['TEXTO_COMBINADO'].tolist()

    # Trava de segurança para bases muito pequenas
    if len(textos) < 5:
        df['TEMA_GEMINI'] = "Amostra Insuficiente"
        return df

    with st.spinner("Vetorizando textos (TF-IDF)..."):
        # 2. Vetorização: Transforma o texto em matriz matemática ignorando palavras comuns (stop words)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
        X = vectorizer.fit_transform(textos)

    with st.spinner("Calculando o K ideal via Silhouette Score..."):
        # 3. Otimização do K: Testa de 2 até max_clusters para achar a melhor divisão natural
        best_k = 2
        best_score = -1
        limite_k = min(max_clusters, len(textos) - 1)
        
        if limite_k >= 3:
            for k in range(2, limite_k + 1):
                kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans_test.fit_predict(X)
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

    # 4. Aplica o agrupamento definitivo
    st.info(f"📊 Algoritmo Silhouette identificou **{best_k} agrupamentos ótimos** (Score: {best_score:.2f}). Nomeando clusters via IA...")
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_text['CLUSTER_ID'] = kmeans_final.fit_predict(X)

    # 5. LLM Naming: Pega amostras de cada cluster para o Gemini rotular
    cluster_names = {}
    progress_bar = st.progress(0)
    
    for cluster_id in range(best_k):
        # Seleciona os 5 documentos mais relevantes do cluster (baseado no tamanho do texto)
        amostra = df_text[df_text['CLUSTER_ID'] == cluster_id].copy()
        amostra['TAM_TEXTO'] = amostra['TEXTO_COMBINADO'].str.len()
        amostra_top = amostra.sort_values(by='TAM_TEXTO', ascending=False).head(5)
        
        textos_amostra = ""
        for _, row in amostra_top.iterrows():
            # Pegamos os primeiros 600 caracteres do abstract para economizar tokens
            resumo = str(row['ABSTRACT'])[:600] if pd.notna(row['ABSTRACT']) else "Sem resumo"
            textos_amostra += f"- Título: {row['TITLE']}\n  Resumo: {resumo}...\n\n"
            
        prompt = f"""
        Você é um cientista de dados especialista em revisão de literatura.
        Abaixo estão amostras representativas de artigos científicos que um algoritmo agrupou no mesmo cluster temático:
        
        {textos_amostra}
        
        Sua tarefa: Sintetize o tema central unificador desta escola de pesquisa.
        Responda APENAS com o nome do tema em Português, de forma acadêmica e concisa (máximo de 2 a 4 palavras). 
        Nenhuma pontuação final, aspas ou texto adicional.
        """
        
        try:
            # Pausa reduzida, pois agora faremos pouquíssimas requisições (ex: 4 ou 5)
            time.sleep(2) 
            response = model.generate_content(prompt)
            tema_nome = response.text.strip().replace('\n', '').replace('"', '').replace('*', '').title()
            cluster_names[cluster_id] = tema_nome
        except Exception as e:
            cluster_names[cluster_id] = f"Tema {cluster_id + 1} (Processamento Falhou)"
            
        progress_bar.progress((cluster_id + 1) / best_k)

    progress_bar.empty()

    # 6. Mapeamento Final: Atribui o nome gerado pela IA aos índices originais
    # Criamos uma Series para garantir que o Pandas alinhe os dados pelo INDEX
    series_temas = df_text['CLUSTER_ID'].map(cluster_names)
    
    # Gravamos na coluna final usando o índice original do DataFrame passado
    df['TEMA_GEMINI'] = series_temas
    
    # Garantimos que não existam valores nulos residuais em strings
    df['TEMA_GEMINI'] = df['TEMA_GEMINI'].fillna("Outros/Não Categorizado")
    
    return df

def gerar_mapas_conceituais(df, top_n_words=50, n_clusters=4):
    """Gera Mapas Conceituais 2D e 3D usando PCA e K-Means Clustering."""
    import pandas as pd
    import plotly.express as px
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    if 'KEYWORDS' not in df.columns:
        return None, None

    # 1. Tokenizador Customizado (Evita quebrar palavras compostas como 'Knowledge Management')
    def custom_tokenizer(text):
        return [w.strip().title() for w in str(text).split(';') if w.strip()]

    textos_originais = df['KEYWORDS'].dropna().astype(str).tolist()
    if not textos_originais:
        return None, None

    # 2. Matriz de Coocorrência (Term-Document Matrix)
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=top_n_words)
    try:
        X = vectorizer.fit_transform(textos_originais)
    except:
        return None, None
        
    termos = vectorizer.get_feature_names_out()
    X_T = X.T.toarray() # Transpõe para focar nos termos em vez dos documentos

    if len(termos) < n_clusters:
        return None, None

    # 3. K-Means Clustering (Identificação de Escolas de Pensamento)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_T)
    clusters_str = [f"Cluster {c+1}" for c in clusters]

    # 4. Redução de Dimensionalidade (PCA - Escalonamento Multidimensional)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_T)

    df_mapa = pd.DataFrame({
        'Termo': termos,
        'Dim1': X_pca[:, 0],
        'Dim2': X_pca[:, 1],
        'Dim3': X_pca[:, 2],
        'Cluster': clusters_str,
        'Frequência': X_T.sum(axis=1) # Tamanho da bolha
    })

    # --- ESTÉTICA TRANSPARENTE E SIMETRICS ---
    layout_bg = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    # 5. Renderização do Gráfico 2D
    fig_2d = px.scatter(
        df_mapa, x='Dim1', y='Dim2', text='Termo', color='Cluster', size='Frequência',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_2d.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
    fig_2d.update_layout(**layout_bg)
    fig_2d.update_xaxes(title="Dimensão 1", showgrid=True, gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)')
    fig_2d.update_yaxes(title="Dimensão 2", showgrid=True, gridcolor='rgba(128,128,128,0.2)', zerolinecolor='rgba(128,128,128,0.5)')

    # 6. Renderização do Gráfico 3D (Ajuste de visibilidade de grade)
    fig_3d = px.scatter_3d(
        df_mapa, x='Dim1', y='Dim2', z='Dim3', text='Termo', color='Cluster', size='Frequência',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_3d.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='white')))
    fig_3d.update_layout(**layout_bg)
    
    # --- ATUALIZAÇÃO PARA EIXOS E GRADE VISÍVEIS ---
    fig_3d.update_layout(scene=dict(
        xaxis=dict(
            title="Dim 1", 
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.5)', # Aumentada a opacidade para 0.5
            showline=True, 
            linecolor='rgba(128,128,128,0.8)', # Linha sólida do eixo
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.2)', # Marca o centro do mapa
            showbackground=False
        ),
        yaxis=dict(
            title="Dim 2", 
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.5)',
            showline=True,
            linecolor='rgba(128,128,128,0.8)',
            showbackground=False
        ),
        zaxis=dict(
            title="Dim 3", 
            showgrid=True, 
            gridcolor='rgba(128,128,128,0.5)',
            showline=True,
            linecolor='rgba(128,128,128,0.8)',
            showbackground=False
        ),
        # Ajusta a câmera para uma perspectiva que favoreça a visão da grade
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)) 
    ))

    return fig_2d, fig_3d

def get_country_collaboration_network(df, top_n=30):
    """Gera a rede matemática de colaboração entre países."""
    import networkx as nx
    from itertools import combinations
    from collections import Counter

    if 'COUNTRY' not in df.columns: return None

    edges_list = []
    node_counts = Counter()

    for row in df['COUNTRY'].dropna():
        # Limpa, padroniza e remove duplicatas dentro do mesmo artigo
        paises = sorted(list(set([c.strip().title() for c in str(row).split(';') if c.strip()])))
        
        if len(paises) > 0:
            for c in paises: node_counts[c] += 1
        
        if len(paises) > 1:
            edges_list.extend(list(combinations(paises, 2)))

    if not edges_list: return None

    edge_counts = Counter(edges_list)
    top_countries = [c for c, _ in node_counts.most_common(top_n)]

    G = nx.Graph()
    for (u, v), w in edge_counts.items():
        if u in top_countries and v in top_countries:
            G.add_edge(u, v, weight=w)
            
    for c in top_countries:
        if c not in G: G.add_node(c)
        
    nx.set_node_attributes(G, {n: node_counts[n] for n in G.nodes()}, 'count')
    return G

def plot_circular_collaboration(df, top_n=30):
    """Gera o grafo de rede circular com hover interativo detalhando parcerias."""
    import plotly.graph_objects as go
    import networkx as nx

    G = get_country_collaboration_network(df, top_n)
    if not G or len(G.nodes) == 0: return None

    # O layout circular posiciona os nós em anel
    pos = nx.circular_layout(G)

    edge_x, edge_y = [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color='rgba(220, 53, 69, 0.25)'),
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, node_text, node_size, node_hover_texts = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        count = G.nodes[node]['count']
        node_size.append(count)
        
        # --- LÓGICA DE HOVER: Identifica parceiros e pesos no anel ---
        neighbors = G[node]
        if len(neighbors) > 0:
            # Ordena os parceiros pela força da colaboração
            collabs = sorted([(n, d['weight']) for n, d in neighbors.items()], key=lambda x: x[1], reverse=True)
            collab_list = [f"{p}: {w} docs" for p, w in collabs]
            collab_str = "<br>  - " + "<br>  - ".join(collab_list)
        else:
            collab_str = "<br>  (Sem parcerias diretas)"
            
        hover_text = f"<b>{node}</b><br>Total de Documentos: {count}<br><b>Principais Parcerias:</b>{collab_str}"
        node_hover_texts.append(hover_text)

    # Escala de tamanho das bolhas
    max_s, min_s = (max(node_size), min(node_size)) if node_size else (1, 0)
    scaled_sizes = [15 + ((s - min_s) / (max_s - min_s + 1)) * 40 for s in node_size]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text', 
        text=node_text,
        hovertext=node_hover_texts, # Injeta as informações de hover calculadas
        hoverinfo='text',           # Garante que apenas o nosso texto personalizado apareça
        textposition="top center",
        marker=dict(
            color='#e76f51', 
            size=scaled_sizes, 
            line=dict(width=1, color='white'), 
            opacity=0.9
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Country Collaboration (Circular Network)", font=dict(size=18), x=0.5),
            showlegend=False, 
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=550,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=60, b=20)
        )
    )
    return fig

def plot_map_collaboration(df, top_n=30):
    """Gera o mapa-múndi de colaboração científica com preenchimento (Choropleth) e arestas dinâmicas."""
    import plotly.graph_objects as go
    import networkx as nx

    G = get_country_collaboration_network(df, top_n)
    if not G or len(G.nodes) == 0: return None

    # Dicionário de coordenadas de alta performance para o traçado das LINHAS
    lat_lon = {
        'Usa': (37.09, -95.71), 'United States': (37.09, -95.71), 'Brazil': (-14.23, -51.92),
        'China': (35.86, 104.19), 'Spain': (40.46, -3.74), 'United Kingdom': (55.37, -3.43),
        'England': (55.37, -3.43), 'Italy': (41.87, 12.56), 'Australia': (-25.27, 133.77),
        'Germany': (51.16, 10.45), 'France': (46.22, 2.21), 'Canada': (56.13, -106.34),
        'India': (20.59, 78.96), 'Japan': (36.20, 138.25), 'South Africa': (-30.55, 22.93),
        'Mexico': (23.63, -102.55), 'Portugal': (39.39, -8.22), 'Netherlands': (52.13, 5.29),
        'Sweden': (60.12, 18.64), 'Switzerland': (46.81, 8.22), 'Colombia': (4.57, -74.29),
        'Argentina': (-38.41, -63.61), 'Chile': (-35.67, -71.54), 'Russia': (61.52, 105.31),
        'South Korea': (35.90, 127.76), 'Denmark': (56.26, 9.50), 'Norway': (60.47, 8.46),
        'Finland': (61.92, 25.74), 'Belgium': (50.50, 4.46), 'Austria': (47.51, 14.55),
        'New Zealand': (-40.90, 174.88), 'Turkey': (38.96, 35.24), 'Iran': (32.42, 53.68),
        'Israel': (31.04, 34.85), 'Poland': (51.91, 19.14), 'Saudi Arabia': (23.88, 45.07),
        'Taiwan': (23.69, 120.96), 'Singapore': (1.35, 103.81), 'Malaysia': (4.21, 101.97),
        'Greece': (39.07, 21.82), 'Ireland': (53.14, -7.69), 'Nigeria': (9.08, 8.67),
        'Romania': (45.94, 24.96), 'Czech Republic': (49.81, 15.47), 'Hungary': (47.16, 19.50)
    }

    fig = go.Figure()

    # 1. PREPARAÇÃO DOS DADOS PARA O CHOROPLETH (Preenchimento) e HOVER
    countries = list(G.nodes())
    plot_countries = [] # Para o Plotly reconhecer corretamente "Usa"
    z_values = []
    hover_texts = []

    for node in countries:
        # Padroniza nomes para o Plotly
        if node.lower() == 'usa': plot_countries.append('United States')
        elif node.lower() == 'uk': plot_countries.append('United Kingdom')
        else: plot_countries.append(node)
        
        # Valores de densidade (Qtd de documentos)
        z_values.append(G.nodes[node]['count'])
        
        # Constrói o texto do Tooltip iterando sobre as arestas do nó
        neighbors = G[node]
        if len(neighbors) > 0:
            # Ordena os parceiros por quantidade de colaboração (decrescente)
            collabs = sorted([(n, d['weight']) for n, d in neighbors.items()], key=lambda x: x[1], reverse=True)
            collab_str_list = [f"{parceiro}: {peso} docs" for parceiro, peso in collabs]
            collab_str = "<br>  - " + "<br>  - ".join(collab_str_list)
        else:
            collab_str = "<br>  (Sem colaborações diretas)"
            
        hover_text = f"<b>{node}</b><br>Documentos Totais: {G.nodes[node]['count']}<br><b>Principais Parceiros:</b>{collab_str}"
        hover_texts.append(hover_text)

    # 2. DESENHO DOS PAÍSES (Choropleth)
    fig.add_trace(go.Choropleth(
        locations=plot_countries,
        locationmode='country names',
        z=z_values,
        text=hover_texts,
        hoverinfo='text',
        colorscale='Teal', # Fica em sintonia com a cor das bolhas antigas
        showscale=False,   # Oculta a barra lateral para um visual mais limpo
        marker_line_color='rgba(255, 255, 255, 0.5)',
        marker_line_width=0.5
    ))

    # 3. DESENHO DAS ARESTAS PROPORCIONAIS
    edges = list(G.edges(data=True))
    if edges:
        max_weight = max([d['weight'] for u, v, d in edges])
        min_weight = min([d['weight'] for u, v, d in edges])
        
        for u, v, d in edges:
            if u in lat_lon and v in lat_lon:
                weight = d['weight']
                # Escalonamento da grossura da linha (de 1 a 6 pixels)
                if max_weight == min_weight:
                    line_width = 1.5
                else:
                    line_width = 1 + ((weight - min_weight) / (max_weight - min_weight)) * 5
                
                fig.add_trace(go.Scattergeo(
                    lat=[lat_lon[u][0], lat_lon[v][0]],
                    lon=[lat_lon[u][1], lat_lon[v][1]],
                    mode='lines',
                    line=dict(width=line_width, color='rgba(220, 53, 69, 0.55)'), # Vermelho transparente
                    hoverinfo='none', 
                    showlegend=False
                ))

    # 4. ESTÉTICA GERAL ALINHADA AO STREAMLIT
    fig.update_layout(
        title=dict(text="Mapa de Colaboração", font=dict(size=18)),
        geo=dict(
            showframe=False,             # Remove a borda quadrada do mapa
            showcoastlines=False,        # Remove a linha preta dura da costa
            projection_type="equirectangular",
            showland=True, 
            landcolor="rgba(128, 128, 128, 0.15)", # Países sem dados ficam com um cinza bem sutil
            showocean=True, 
            oceancolor="rgba(0,0,0,0)",  # Oceano fica 100% transparente para herdar fundo do Streamlit
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=550,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def plot_top_keywords_metric(df, metric_name, top_n=20):
    """Gera um gráfico de barras para as Top Keywords com legenda gradiente."""
    import pandas as pd
    import plotly.express as px

    col_kw = 'KEYWORDS'
    if col_kw not in df.columns or df[col_kw].dropna().empty:
        return None

    # Preparação e explosão dos dados
    df_kw = df[[col_kw, 'TOTAL CITATIONS']].dropna(subset=[col_kw]).copy()
    df_kw[col_kw] = df_kw[col_kw].str.split(';')
    df_kw = df_kw.explode(col_kw)
    df_kw[col_kw] = df_kw[col_kw].str.strip().str.title()
    df_kw = df_kw[df_kw[col_kw] != '']

    # Agrupamento
    res = df_kw.groupby(col_kw)['TOTAL CITATIONS'].agg(['count', 'sum', 'mean']).reset_index()
    res.columns = [col_kw, 'Qtd. de Documentos', 'Total de Citações', 'Média de Citações']
    res = res.sort_values(by=metric_name, ascending=False).head(top_n)

    # Construção do Gráfico
    fig = px.bar(
        res, 
        x=metric_name, 
        y=col_kw, 
        orientation='h',
        labels={col_kw: 'Palavra-chave', metric_name: metric_name},
        color=metric_name,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title=dict(text=metric_name, font=dict(size=12)),
            thicknessmode="pixels", thickness=15,
            lenmode="fraction", len=0.8,
            yanchor="middle", y=0.5
        )
    )
    
    return fig

def plot_author_production_over_time(df, top_n=10):
    """Gera o gráfico Top-Authors' Production over Time"""
    import pandas as pd
    import plotly.graph_objects as go

    if 'AUTHORS' not in df.columns or 'YEAR CLEAN' not in df.columns: return None

    # 1. Desagrega os autores em linhas individuais preservando Ano e Citações
    df_auth = df.dropna(subset=['AUTHORS', 'YEAR CLEAN']).copy()
    df_auth['YEAR CLEAN'] = pd.to_numeric(df_auth['YEAR CLEAN'], errors='coerce')
    df_auth = df_auth.dropna(subset=['YEAR CLEAN'])

    author_data = []
    for _, row in df_auth.iterrows():
        auth_list = [a.strip() for a in str(row['AUTHORS']).split(';') if a.strip()]
        cit = row.get('TOTAL CITATIONS', 0)
        if pd.isna(cit): cit = 0
        for a in auth_list:
            author_data.append({'Author': a, 'Year': int(row['YEAR CLEAN']), 'Citations': float(cit), 'Docs': 1})

    df_expanded = pd.DataFrame(author_data)
    if df_expanded.empty: return None

    # 2. Identifica os Top N Autores pelo total de documentos
    top_authors = df_expanded.groupby('Author')['Docs'].sum().nlargest(top_n).index.tolist()
    df_top = df_expanded[df_expanded['Author'].isin(top_authors)]

    # 3. Agrupa por Autor e Ano
    df_grouped = df_top.groupby(['Author', 'Year']).agg({'Docs': 'sum', 'Citations': 'sum'}).reset_index()

    # 4. Construção do Plotly Chart
    fig = go.Figure()

    for author in top_authors:
        df_a = df_grouped[df_grouped['Author'] == author]
        if df_a.empty: continue
        
        # Linha fina conectando o primeiro ao último ano de publicação do autor
        min_year, max_year = df_a['Year'].min(), df_a['Year'].max()
        fig.add_trace(go.Scatter(
            x=[min_year, max_year], y=[author, author],
            mode='lines', line=dict(color='rgba(200, 100, 100, 0.4)', width=2),
            showlegend=False, hoverinfo='none'
        ))

    # Adição das "Bolhas" (Scatter)
    fig.add_trace(go.Scatter(
        x=df_grouped['Year'],
        y=df_grouped['Author'],
        mode='markers',
        marker=dict(
            size=df_grouped['Docs'],
            sizemode='area',
            sizeref=2.*max(df_grouped['Docs'])/(25.**2), # Escala bibliometrix
            sizemin=8,
            color=df_grouped['Citations'],
            colorscale='Teal', # Tons de azul/verde acadêmicos
            showscale=True,
            colorbar=dict(title='Total de Citações<br>por Ano', thickness=15),
            line=dict(color='gray', width=1),
            opacity=0.8
        ),
        text=df_grouped.apply(lambda r: f"Autor: {r['Author']}<br>Ano: {r['Year']}<br>Artigos: {r['Docs']}<br>Citações Acumuladas: {r['Citations']}", axis=1),
        hoverinfo='text',
        name='Produção'
    ))

    fig.update_layout(
        title=dict(text="Produção dos principais autores ao longo do tempo", font=dict(size=20)), # Removida cor estática
        xaxis=dict(title='Ano', tickformat='d', showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(title='', autorange="reversed", showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
        paper_bgcolor='rgba(0,0,0,0)', # Fundo do papel transparente
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
        height=550,
        margin=dict(l=150, r=20, t=60, b=40)
    )

    return fig

def plot_lotkas_law(df):
    """Gera a distribuição de Lotka (Frequência de Produtividade Científica)."""
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    if 'AUTHORS' not in df.columns: return None

    # 1. Contagem de documentos por autor
    flat_auths = [a.strip() for sublist in df['AUTHORS'].dropna().astype(str).str.split(';') for a in sublist if a.strip()]
    if not flat_auths: return None
    
    author_counts = pd.Series(flat_auths).value_counts()
    
    # 2. Distribuição Observada (Quantos autores escreveram X artigos?)
    freq_dist = author_counts.value_counts().sort_index()
    x_obs = freq_dist.index.values
    y_obs = (freq_dist.values / freq_dist.values.sum())
    
    # 3. Distribuição Teórica de Lotka (y = c / x^n)
    # A Lei de Lotka estima que 'n' é aproximadamente 2.
    # 'c' é a constante para garantir que a soma das probabilidades tenda a 1.
    max_x = max(x_obs)
    x_theo = np.arange(1, max_x + 1)
    c = 1.0 / np.sum(1.0 / (x_theo**2))
    y_theo = c / (x_theo**2)

    # 4. Construção do Gráfico Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_obs, y=y_obs, mode='lines', name='Observado', 
        line=dict(color='blue', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_theo, y=y_theo, mode='lines', name='Teórico (Lotka)', 
        line=dict(color='red', width=1.5, dash='dash')
    ))

    fig.update_layout(
        title=dict(text="Produtividade científica (Lei de Lotka)", font=dict(size=20)), # Removida cor estática
        xaxis=dict(title="Artigos", showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.3)', mirror=True),
        yaxis=dict(title="Frequência de Autores", showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.3)', mirror=True),
        paper_bgcolor='rgba(0,0,0,0)', # Fundo do papel transparente
        plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
        height=550,
        legend=dict(x=0.65, y=0.9, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(128, 128, 128, 0.3)', borderwidth=1), # Legenda transparente
        margin=dict(l=60, r=20, t=60, b=40)
    )

    return fig

def gerar_historiograph(df, top_n=30):
    """Gera um gráfico histórico de citações diretas com busca flexível de padrões."""
    import networkx as nx
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import re

    if 'YEAR CLEAN' not in df.columns or 'TOTAL CITATIONS' not in df.columns or 'AUTHORS' not in df.columns:
        return None

    col_ref = next((c for c in ['REFERENCES_UNIFIED', 'CITED REFERENCES', 'REFERENCES', 'CR'] if c in df.columns), None)
    if not col_ref:
        return None

    # Top N mais citados da base para manter o gráfico legível
    df_top = df.dropna(subset=['YEAR CLEAN', 'TOTAL CITATIONS', col_ref, 'AUTHORS']).copy()
    if df_top.empty: return None
        
    df_top = df_top.sort_values(by='TOTAL CITATIONS', ascending=False).head(top_n)

    def get_short_name(row):
        first_author = str(row['AUTHORS']).split(';')[0].split(',')[0].strip().title()
        return f"{first_author}, {int(row['YEAR CLEAN'])}"

    df_top['HistName'] = df_top.apply(get_short_name, axis=1)

    edges = []
    # Otimização: Criamos uma lista de dicionários para evitar o overhead do iterrows no loop duplo
    records = df_top.to_dict('records')

    for i, row_A in enumerate(records):
        refs_A = str(row_A[col_ref]).lower()
        year_A = row_A['YEAR CLEAN']
        name_A = row_A['HistName']

        for j, row_B in enumerate(records):
            if i == j: continue
            
            year_B = row_B['YEAR CLEAN']
            name_B = row_B['HistName']
            
            # Extraímos apenas o sobrenome do primeiro autor para a busca
            first_author_B = str(row_B['AUTHORS']).split(';')[0].split(',')[0].strip().lower()
            ano_B_str = str(int(year_B))

            # --- CORE DA CORREÇÃO: Busca Flexível ---
            # Verificamos se o Sobrenome + Ano estão presentes na mesma string de referência.
            # Isso captura formatos como "SILVA, 2020", "SILVA_A, 2020" ou "SILVA A, 2020".
            if year_A > year_B: # Um documento só pode citar algo que veio antes dele
                if first_author_B in refs_A and ano_B_str in refs_A:
                    edges.append((name_A, name_B))

    G = nx.DiGraph()
    for row in records:
        G.add_node(row['HistName'], year=int(row['YEAR CLEAN']), citations=row['TOTAL CITATIONS'])
    
    G.add_edges_from(edges)

    if len(G.nodes) == 0: return None

    # Posicionamento
    pos = {}
    years = nx.get_node_attributes(G, 'year')
    year_groups = {}
    for node, yr in years.items():
        year_groups.setdefault(yr, []).append(node)

    for yr, nodes in year_groups.items():
        y_vals = np.linspace(0.1, 0.9, len(nodes) + 2)[1:-1]
        for idx, node in enumerate(nodes):
            pos[node] = (yr, y_vals[idx])

    # --- CORREÇÃO VISUAL: Linhas mais escuras e visíveis ---
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.2, color='#888888'), # Cor cinza médio para contraste no branco
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]['citations'])

    # Escalonamento de tamanho das bolhas
    max_s, min_s = (max(node_size), min(node_size)) if node_size else (1, 0)
    scaled_sizes = [20 + ((s - min_s) / (max_s - min_s + 1)) * 40 for s in node_size]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color='#82c2c2', 
            size=scaled_sizes, 
            line=dict(width=1, color='white'),
            opacity=0.85
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            # --- ATUALIZAÇÕES ESTÉTICAS ---
            paper_bgcolor='rgba(0,0,0,0)', # Fundo do papel transparente
            plot_bgcolor='rgba(0,0,0,0)',  # Fundo do gráfico transparente
            height=600,
            xaxis=dict(
                title="Linha do Tempo", 
                showgrid=True, 
                gridcolor='rgba(128, 128, 128, 0.2)', # Grid suave e adaptável
                tickmode='linear', 
                dtick=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            margin=dict(l=40, r=40, t=20, b=40)
            # ------------------------------
        )
    )
    return fig

def plot_grafo_estatico(G, titulo="Rede Estática"):
    """Gera um gráfico estático usando Matplotlib e NetworkX estilo VOSviewer."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    if len(G.nodes) == 0: 
        return None
    
    # Criamos a figura em alta resolução
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    
    # Remove nós isolados para limpar a visualização
    G_clean = G.copy()
    G_clean.remove_nodes_from(list(nx.isolates(G_clean)))
    if len(G_clean.nodes) == 0: 
        return None

    # Layout de forças (Kamada-Kawai ou Spring)
    pos = nx.spring_layout(G_clean, k=0.8, iterations=100, seed=42)
    
    # Detecção de Comunidades para colorir clusters
    try:
        communities = list(greedy_modularity_communities(G_clean, weight='weight'))
    except:
        communities = [list(G_clean.nodes())]
        
    cores_cluster = ['#e15759', '#4e79a7', '#59a14f', '#f28e2c', '#af7aa1', '#edc949', '#76b7b2', '#ff9da7', '#9c755f', '#bab0ab']
    color_map = {}
    for i, comm in enumerate(communities):
        c = cores_cluster[i % len(cores_cluster)]
        for node in comm:
            color_map[node] = c
            
    node_colors = [color_map.get(n, '#cccccc') for n in G_clean.nodes()]
    
    # Tamanho dinâmico baseado no grau (Centralidade)
    degrees = dict(G_clean.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [(degrees[n] / max_deg) * 3500 + 150 for n in G_clean.nodes()]
    
    # Espessura das arestas baseada no peso
    edges = G_clean.edges(data=True)
    edge_weights = [d.get('weight', 1) * 0.4 for u, v, d in edges]
    
    # Desenho das arestas
    nx.draw_networkx_edges(G_clean, pos, ax=ax, width=edge_weights, alpha=0.25, edge_color='#999999')
    
    # Desenho dos nós (bolhas coloridas)
    nx.draw_networkx_nodes(G_clean, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.8, edgecolors='white', linewidths=1.5)
    
    # Desenho dos rótulos (texto proporcional ao tamanho do nó)
    for node, (x, y) in pos.items():
        size = (degrees[node] / max_deg) * 18 + 7 # fonte mínima 7, máxima 25
        ax.text(x, y, str(node), fontsize=size, ha='center', va='center', fontweight='bold', color='black', alpha=0.85, 
                bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
        
    ax.set_title(titulo, fontsize=22, fontweight='bold', pad=20, color='#333333')
    ax.axis('off')
    
    return fig

@st.cache_data
def _engine_calculo_sna(nodes_list, edges_list, node_types):
    """Engine interna para processar NetworkX. Retorna dados formatados e dicionários brutos."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)
    
    # Cálculos Brutos
    degree_abs = dict(G.degree())
    degree_cent = nx.degree_centrality(G)
    
    # OTIMIZAÇÃO: O parâmetro 'k' usa uma amostra matemática. 
    # Em redes maiores que 100 nós, ele estima o resultado com 99% de precisão,
    # mas roda em uma fração minúscula de segundo.
    amostra = min(100, len(G))
    bet_cent = nx.betweenness_centrality(G, k=amostra)
    
    clos_cent = nx.closeness_centrality(G)
    
    # Eigenvector é sensível e pode falhar em redes desconexas
    try:
        eigen_cent = nx.eigenvector_centrality_numpy(G)
    except:
        eigen_cent = {n: 0 for n in G.nodes()}

    data_list = []
    for node in G.nodes():
        data_list.append({
            "Item": node,
            "Tipo": node_types.get(node, "Outro"),
            "Grau Absoluto": degree_abs[node],
            "Grau Centralidade": round(degree_cent[node], 4),
            "Centralidade (Eigen)": round(eigen_cent.get(node, 0), 4),
            "Betweenness": round(bet_cent[node], 4),
            "Closeness": round(clos_cent[node], 4)
        })
    
    # Retornamos a lista para a tabela E os dicionários para o grafo
    return data_list, degree_abs, eigen_cent, bet_cent, clos_cent

def gerar_tabela_metricas_completas(df, _pbar=None):
    """Interface que gerencia a barra de progresso e chama a engine para a tabela SNA."""
    total = len(df)
    col_titulos = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
    col_autores = next((c for c in ['AUTHORS', 'AU'] if c in df.columns), None)
    col_paises = next((c for c in ['COUNTRY'] if c in df.columns), None)
    col_venue = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
    
    nodes, edges, node_types = [], [], {}

    for i, (_, row) in enumerate(df.iterrows()):
        if _pbar: _pbar.progress((i + 1) / total, text=f"Mapeando topologia: {i+1}/{total}")
        doc = str(row[col_titulos]) if col_titulos and pd.notna(row[col_titulos]) else None
        if not doc: continue
        nodes.append(doc); node_types[doc] = "Documento"
        
        if col_autores and pd.notna(row[col_autores]):
            for a in [x.strip() for x in str(row[col_autores]).split(';') if x.strip()]:
                nodes.append(a); node_types[a] = "Autor"; edges.append((doc, a))
        if col_paises and pd.notna(row[col_paises]):
            for p in [x.strip() for x in str(row[col_paises]).split(';') if x.strip()]:
                nodes.append(p); node_types[p] = "País"; edges.append((doc, p))
        if col_venue and pd.notna(row[col_venue]):
            v = str(row[col_venue]).strip(); nodes.append(v); node_types[v] = "Local de Publicação (Venue)"; edges.append((doc, v))

    if _pbar: _pbar.progress(1.0, text="Executando algoritmos de centralidade...")
    
    # Unpack apenas do primeiro item (a lista de dados)
    res_data, _, _, _, _ = _engine_calculo_sna(list(set(nodes)), list(set(edges)), node_types)
    return pd.DataFrame(res_data).sort_values(by="Grau Absoluto", ascending=False)

def criar_grafo_e_metricas(df, coluna, top_n, metric_for_size="Tamanho Fixo"):
    from networkx.algorithms.community import greedy_modularity_communities
    
    docs_items = []
    for row in df[coluna].dropna():
        items = [x.strip() for x in str(row).split(';') if x.strip()]
        if items: docs_items.append(items)

    all_items = [item for sublist in docs_items for item in sublist]
    item_counts = Counter(all_items)
    top_items = set([x[0] for x in item_counts.most_common(top_n)])

    G = nx.Graph()
    for item in top_items: G.add_node(item, count=item_counts[item])
    for items in docs_items:
        filtered_items = [x for x in items if x in top_items]
        if len(filtered_items) > 1:
            for u, v in combinations(sorted(filtered_items), 2):
                if G.has_edge(u, v): G[u][v]['weight'] += 1
                else: G.add_edge(u, v, weight=1)

    degree = dict(G.degree())
    try: betweenness = nx.betweenness_centrality(G, weight='weight')
    except: betweenness = {n: 0 for n in G.nodes()}
    try: closeness = nx.closeness_centrality(G)
    except: closeness = {n: 0 for n in G.nodes()}
    try: eigenvector = nx.eigenvector_centrality_numpy(G)
    except: eigenvector = {n: 0 for n in G.nodes()}

    node_data = []
    for node in G.nodes():
        node_data.append({
            "Nó": node,
            "Grau Absoluto": degree.get(node, 0),
            "Centralidade (Eigenvector)": round(eigenvector.get(node, 0), 4),
            "Betweenness": round(betweenness.get(node, 0), 4),
            "Closeness": round(closeness.get(node, 0), 4)
        })
    df_nodes = pd.DataFrame(node_data).sort_values("Grau Absoluto", ascending=False)

    def get_scaled_size(val, min_val, max_val, min_size=15, max_size=55):
        if max_val == min_val: return min_size
        return min_size + (val - min_val) * (max_size - min_size) / (max_val - min_val)

    metric_dict = {"Grau Absoluto": degree, "Centralidade (Eigen)": eigenvector, "Betweenness": betweenness, "Closeness": closeness}
    nodes_agraph = []
    font_config = {"color": "black", "strokeWidth": 3, "strokeColor": "white"}
    
    # ATUALIZAÇÃO: Construção dos Nós com Tooltip (hover)
    for node in G.nodes():
        # Monta o texto do hover
        hover_text = (
            f"Nó: {node}\n"
            f"Grau Absoluto: {degree.get(node, 0)}\n"
            f"Eigen: {round(eigenvector.get(node, 0), 4)}\n"
            f"Betweenness: {round(betweenness.get(node, 0), 4)}\n"
            f"Closeness: {round(closeness.get(node, 0), 4)}"
        )
        
        # Define o tamanho
        if metric_for_size != "Tamanho Fixo" and metric_for_size in metric_dict:
            m_dict = metric_dict[metric_for_size]
            min_m, max_m = (min(m_dict.values()), max(m_dict.values())) if m_dict.values() else (0, 1)
            n_size = get_scaled_size(m_dict.get(node, 0), min_m, max_m)
        else:
            n_size = 25
            
        nodes_agraph.append(Node(id=node, label=node, size=n_size, color="#1273B9", font=font_config, title=hover_text))

    edges_agraph = [Edge(source=u, target=v, value=d['weight'], color="#E0E0E0") for u, v, d in G.edges(data=True)]
    
    # Cálculos de métricas gerais da rede (mantidos iguais)
    net_metrics = {}
    if len(G) > 0:
        net_metrics['densidade'] = nx.density(G)
        # ... (seu código de métricas continua o mesmo aqui, o importante é o final)
    
    # Retornamos o objeto G para usar no gráfico estático
    return nodes_agraph, edges_agraph, df_nodes, net_metrics, G

def processar_excel_wos(file):
    
    # 1. Identifica a extensão para escolher o motor correto
    engine = 'openpyxl' if file.name.endswith('.xlsx') else 'xlrd'
    
    # 2. Carrega o arquivo com o motor específico
    df = pd.read_excel(file, engine=engine)
    
    # 1. Mapeamento de Colunas WoS -> Padrão Bibliomat
    mapa_colunas = {
        'Article Title': 'TITLE',
        'Publication Year': 'YEAR',
        'Source Title': 'SECONDARY TITLE',
        'Abstract': 'ABSTRACT',
        'Document Type': 'DOCUMENT TYPE',
        'DOI': 'DOI',
        'Authors': 'AUTHORS',
        'Cited References': 'CITED REFERENCES',
        'Cited References': 'REFERENCES_UNIFIED'
    }
    df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})

    # 2. Tratamento de Citações (WoS Core é o padrão de impacto)
    col_cit = next((c for c in ['Times Cited, WoS Core', 'Times Cited, All Databases'] if c in df.columns), None)
    if col_cit:
        df['TOTAL CITATIONS'] = pd.to_numeric(df[col_cit], errors='coerce').fillna(0)

    # 3. Tratamento de Palavras-Chave (Unindo Author Keywords e Keywords Plus)
    col_de = 'Author Keywords'
    col_id = 'Keywords Plus'
    df['KEYWORDS'] = df[[c for c in [col_de, col_id] if c in df.columns]].fillna('').astype(str).apply(
        lambda x: '; '.join([k for k in x if k.strip() != '']), axis=1
    )

    # 4. Extração de Países (A partir da coluna 'Addresses')
    if 'Addresses' in df.columns:
        def extrair_paises_wos(addr_str):
            if pd.isna(addr_str): return ""
            
            enderecos = str(addr_str).split(';')
            paises_encontrados = [] # Usando nome claro para a lista
            
            for addr in enderecos:
                partes = addr.split(',')
                if len(partes) > 0:
                    # 'pais_texto' é uma string
                    pais_texto = partes[-1].replace('.', '').strip()
                    # Limpa números e CEPs
                    pais_limpo = re.sub(r'\d+', '', pais_texto).strip()
                    
                    # CORREÇÃO: Adicionamos à lista (plural), não à string (singular)
                    if pais_limpo: 
                        paises_encontrados.append(pais_limpo)
            
            # Remove duplicatas e junta com ponto-e-vírgula
            return "; ".join(list(set(paises_encontrados)))
        
        df['COUNTRY'] = df['Addresses'].apply(extrair_paises_wos)

    # 5. Ano Limpo
    if 'YEAR' in df.columns:
        df['YEAR CLEAN'] = pd.to_numeric(df['YEAR'], errors='coerce')

    return df


def processar_csv_scopus(file):
    """Lê um CSV (Scopus) e padroniza as colunas para o ecossistema Bibliomat."""
    
    # Tenta ler o CSV. O Scopus pode usar vírgula e aspas específicas.
    df = pd.read_csv(file, sep=',', encoding='utf-8')
    
    # 1. Mapeamento Direto de Colunas
    mapa_colunas = {
        'Title': 'TITLE',
        'Year': 'YEAR',
        'Source title': 'SECONDARY TITLE',
        'Abstract': 'ABSTRACT',
        'Document Type': 'DOCUMENT TYPE',
        'DOI': 'DOI',
        'References': 'REFERENCES',
        'References': 'REFERENCES_UNIFIED'
    }
    # Renomeia as colunas que existem no CSV
    df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})

    # 2. Tratamento de Citações
    if 'Cited by' in df.columns:
        df['TOTAL CITATIONS'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0)

    # 3. Tratamento de Autores (Convertendo vírgulas para ponto-e-vírgula se necessário)
    if 'Authors' in df.columns:
        # Scopus às vezes traz "Silva A., Santos B." - vamos garantir que a separação por ';' exista
        df['AUTHORS'] = df['Authors'].apply(
            lambda x: str(x).replace('.,', '.;') if pd.notna(x) else ""
        )

    # 4. Tratamento de Palavras-Chave (Juntando as do autor e as indexadas)
    col_kw_existentes = [c for c in ['Author Keywords', 'Index Keywords'] if c in df.columns]
    
    if col_kw_existentes:
        # Garantimos que todos os dados sejam strings e substituímos nulos por texto vazio
        # Usamos uma função lambda para filtrar apenas o que não for vazio antes de juntar
        df['KEYWORDS'] = df[col_kw_existentes].fillna('').astype(str).apply(
            lambda x: '; '.join([termo for termo in x if termo.strip() != '']), 
            axis=1
        )
    else:
        df['KEYWORDS'] = ""

    # 5. Tratamento de País (Extraindo da coluna de afiliações)
    if 'Affiliations' in df.columns:
        def extrair_paises(affil_str):
            if pd.isna(affil_str) or str(affil_str).strip() == '':
                return ""
            
            paises = []
            # Scopus separa múltiplas afiliações por ';'
            lista_affils = str(affil_str).split(';')
            for affil in lista_affils:
                # O país geralmente é a última palavra após a última vírgula
                partes = affil.split(',')
                if partes:
                    pais = partes[-1].strip()
                    # Removemos números ou CEPs que às vezes vêm grudados no nome do país
                    pais_limpo = ''.join([i for i in pais if not i.isdigit()]).strip()
                    paises.append(pais_limpo)
            
            # Remove duplicatas e retorna separado por ponto-e-vírgula
            return "; ".join(list(set(paises)))

        df['COUNTRY'] = df['Affiliations'].apply(extrair_paises)

    # 6. Ano Limpo (Para gráficos temporais)
    if 'YEAR' in df.columns:
        df['YEAR CLEAN'] = pd.to_numeric(df['YEAR'], errors='coerce')

    return df


@st.cache_data
def calcular_metricas_bibliometrix(df):

    """Calcula métricas avançadas baseadas no relatório Main Information do Bibliometrix."""
    import pandas as pd
    import numpy as np

    # 1. Taxa de Crescimento Anual (%)
    anos = df['YEAR CLEAN'].dropna().unique()
    if len(anos) > 1:
        n_anos = anos.max() - anos.min()
        doc_inicio = len(df[df['YEAR CLEAN'] == anos.min()])
        doc_fim = len(df[df['YEAR CLEAN'] == anos.max()])
        # Fórmula CAGR: [(Vfinal/Vinicial)^(1/t) - 1] * 100
        growth_rate = ((doc_fim / doc_inicio)**(1/n_anos) - 1) * 100 if doc_inicio > 0 else 0
    else:
        growth_rate = 0

    # 2. Citações Médias por Ano por Doc
    # NTC: Normalized Total Citations (TC / Média de TC do ano)
    if 'TOTAL CITATIONS' in df.columns and 'YEAR CLEAN' in df.columns:
        df['TCperYear'] = df['TOTAL CITATIONS'] / (2026 - df['YEAR CLEAN'] + 1)
        media_por_ano = df.groupby('YEAR CLEAN')['TOTAL CITATIONS'].transform('mean')
        df['NTC'] = df['TOTAL CITATIONS'] / media_por_ano
    
    # 3. Colaboração (SCP vs MCP)
    # SCP: Single Country Pubs | MCP: Multiple Country Pubs
    mcp_count = 0
    if 'COUNTRY' in df.columns:
        mcp_count = df['COUNTRY'].dropna().apply(lambda x: len(set(str(x).split(';'))) > 1).sum()
    
    # 4. Índice de Coautoria
    autores_por_doc = 0
    if 'AUTHORS' in df.columns:
        counts = df['AUTHORS'].dropna().apply(lambda x: len(str(x).split(';')))
        autores_por_doc = counts.mean()
        docs_unico_autor = (counts == 1).sum()
    else:
        docs_unico_autor = 0

    return {
        "growth_rate": round(growth_rate, 2),
        "mcp": mcp_count,
        "scp": len(df) - mcp_count,
        "coauth_index": round(autores_por_doc, 2),
        "single_author_docs": docs_unico_autor,
        "avg_cit_year": round(df['TCperYear'].mean(), 2) if 'TCperYear' in df.columns else 0
    }

@st.cache_data
def gerar_mapa_tematico(df, coluna_texto, n_palavras=150):
    """Gera um Mapa Temático inspirado no Bibliometrix (Centralidade vs Densidade)."""
    import networkx as nx
    import pandas as pd
    import plotly.express as px
    from collections import Counter
    from networkx.algorithms.community import greedy_modularity_communities
    import re
    from wordcloud import STOPWORDS

    # 1. Limpeza e Extração do Corpus
    textos = df[coluna_texto].dropna().astype(str).tolist()
    stopwords = set(STOPWORDS)
    stopwords.update(["research", "study", "analysis", "results", "using", "paper", "article", "author", "may", "can", "will"])

    docs_words = []
    for text in textos:
        words = re.findall(r'\b\w{3,}\b', text.lower())
        words = [w for w in words if w not in stopwords]
        docs_words.append(words)

    todas_palavras = [w for doc in docs_words for w in doc]
    top_words = [w for w, c in Counter(todas_palavras).most_common(n_palavras)]
    top_words_set = set(top_words)

    if not top_words_set: 
        return None

    # 2. Construção da Rede de Co-ocorrência
    G = nx.Graph()
    for doc in docs_words:
        valid_words = [w for w in doc if w in top_words_set]
        for i in range(len(valid_words)):
            for j in range(i+1, len(valid_words)):
                w1, w2 = valid_words[i], valid_words[j]
                if G.has_edge(w1, w2):
                    G[w1][w2]['weight'] += 1
                else:
                    G.add_edge(w1, w2, weight=1)

    if len(G.nodes) == 0: 
        return None

    # 3. Detecção de Comunidades (Temas) e Cálculo de Métricas
    # greedy_modularity aproxima o algoritmo de Louvain nativamente no networkx
    comunidades = list(greedy_modularity_communities(G, weight='weight'))

    dados_clusters = []
    for idx, com in enumerate(comunidades):
        com = list(com)
        if len(com) < 2: continue

        # Frequência total do cluster (tamanho da bolha)
        freq = sum([Counter(todas_palavras)[w] for w in com])

        # Força Interna (Densidade) e Externa (Centralidade)
        internal_weight = 0
        external_weight = 0

        for node in com:
            for vizinho, dict_arestas in G[node].items():
                if vizinho in com:
                    internal_weight += dict_arestas['weight']
                else:
                    external_weight += dict_arestas['weight']

        internal_weight /= 2 # Divide por 2 pois arestas internas foram contadas duas vezes

        # Seleciona as 3 palavras mais proeminentes para nomear o cluster
        palavras_ordenadas = sorted(com, key=lambda w: Counter(todas_palavras)[w], reverse=True)
        nome_tema = "<br>".join(palavras_ordenadas[:3])
        tooltip_words = ", ".join(palavras_ordenadas[:6])

        dados_clusters.append({
            'Cluster': f"Tema {idx+1}",
            'Palavras': tooltip_words,
            'Label': nome_tema,
            'Grau de Desenvolvimento (Densidade)': internal_weight,
            'Grau de Relevância (Centralidade)': external_weight,
            'Frequência': freq
        })

    df_clusters = pd.DataFrame(dados_clusters)
    if df_clusters.empty: 
        return None

    # 4. Construção do Gráfico Plotly
    mean_cent = df_clusters['Grau de Relevância (Centralidade)'].mean()
    mean_dens = df_clusters['Grau de Desenvolvimento (Densidade)'].mean()

    fig = px.scatter(
        df_clusters, 
        x='Grau de Relevância (Centralidade)', 
        y='Grau de Desenvolvimento (Densidade)', 
        size='Frequência',
        color='Cluster', 
        text='Label', 
        hover_data=['Palavras'],
        size_max=50, 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(
        textposition='middle center', 
        textfont_size=11, 
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )

    # Linhas divisórias dos quadrantes
    fig.add_hline(y=mean_dens, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=mean_cent, line_dash="dash", line_color="gray", opacity=0.5)

    # Anotações dos 4 Quadrantes (usando referências fixas da tela xref/yref)
    quadrantes = [
        dict(x=0.99, y=0.99, text="<b>Temas Motores</b><br>(Alta Centralidade/Alta Densidade)", xanchor="right", yanchor="top"),
        dict(x=0.01, y=0.99, text="<b>Temas de Nicho</b><br>(Baixa Centralidade/Alta Densidade)", xanchor="left", yanchor="top"),
        dict(x=0.99, y=0.01, text="<b>Temas Básicos/Transversais</b><br>(Alta Centralidade/Baixa Densidade)", xanchor="right", yanchor="bottom"),
        dict(x=0.01, y=0.01, text="<b>Temas Emergentes/Declínio</b><br>(Baixa Centralidade/Baixa Densidade)", xanchor="left", yanchor="bottom")
    ]
    
    for q in quadrantes:
        fig.add_annotation(
            x=q['x'], y=q['y'], xref="paper", yref="paper", 
            text=q['text'], showarrow=False, 
            font=dict(color="gray", size=11), align=q['xanchor']
        )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=650,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


@st.cache_data
def calcular_similares_biblio(termo_ativo, tipo_busca, df):
    """Calcula a similaridade (Jaccard) do 'DNA acadêmico' entre entidades."""
    if not termo_ativo:
        return {}
    
    # Identifica colunas-chave do dataset
    col_titulos = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
    col_autores = next((c for c in ['AUTHORS', 'AU'] if c in df.columns), None)
    col_kw = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None)
    col_venue = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
    col_paises = next((c for c in ['COUNTRY'] if c in df.columns), None)

    # Dentro da função calcular_similares_biblio, substitua a função extrair_features por:
    
    def extrair_features(df_subset):
        """Extrai as impressões digitais de forma otimizada e sem iterrows."""
        kws, aus, venues = set(), set(), set()
        
        if col_kw:
            # Junta todos os textos, separa por ; e limpa em uma linha
            kws.update([k.strip().lower() for k in ';'.join(df_subset[col_kw].dropna().astype(str)).split(';') if k.strip()])
        if col_autores:
            aus.update([a.strip() for a in ';'.join(df_subset[col_autores].dropna().astype(str)).split(';') if a.strip()])
        if col_venue:
            venues.update(df_subset[col_venue].dropna().astype(str).str.strip().unique())
            
        return kws, aus, venues
        
    # Isola a entidade buscada e captura seu "DNA"
    if tipo_busca == "Documento": subset_alvo = df[df[col_titulos] == termo_ativo]
    elif tipo_busca == "Autor": subset_alvo = df[df[col_autores].fillna('').str.contains(termo_ativo, regex=False)]
    elif tipo_busca == "País": subset_alvo = df[df[col_paises].fillna('').str.contains(termo_ativo, regex=False)] if col_paises else pd.DataFrame()
    elif tipo_busca == "Local de Publicação (Venue)": subset_alvo = df[df[col_venue] == termo_ativo]
    else: return {}

    kw_alvo, au_alvo, venue_alvo = extrair_features(subset_alvo)
    
    # Previne que a entidade combine consigo mesma no escore
    if tipo_busca in ["Documento", "Autor"]: au_alvo.discard(termo_ativo)
    
    dna_alvo = kw_alvo.union(au_alvo).union(venue_alvo)
    if not dna_alvo: return {}

    resultados = []
    
    # Compara o Alvo com todos os Candidatos
    if tipo_busca == "Documento":
        for _, r in df[df[col_titulos] != termo_ativo].iterrows():
            cand_nome = r[col_titulos]
            k, a, v = extrair_features(pd.DataFrame([r]))
            dna_cand = k.union(a).union(v)
            inter = dna_alvo.intersection(dna_cand)
            if inter:
                jaccard = len(inter) / len(dna_alvo.union(dna_cand))
                resultados.append({'Item': cand_nome, 'Similaridade (%)': round(jaccard * 100, 1), 'Traços em Comum': " | ".join(list(inter)[:4])})
                
    elif tipo_busca == "Autor":
        todos_autores = set()
        for au_str in df[col_autores].dropna(): todos_autores.update([a.strip() for a in str(au_str).split(';') if a.strip()])
        todos_autores.discard(termo_ativo)
        
        for cand in todos_autores:
            sub_cand = df[df[col_autores].fillna('').str.contains(cand, regex=False)]
            k, a, v = extrair_features(sub_cand)
            a.discard(cand)
            dna_cand = k.union(a).union(v)
            inter = dna_alvo.intersection(dna_cand)
            if inter:
                jaccard = len(inter) / len(dna_alvo.union(dna_cand))
                resultados.append({'Item': cand, 'Similaridade (%)': round(jaccard * 100, 1), 'Traços em Comum': " | ".join(list(inter)[:4])})
                
    elif tipo_busca in ["País", "Local de Publicação (Venue)"]:
         col_busca = col_paises if tipo_busca == 'País' else col_venue
         if col_busca:
             todos_itens = set([x.strip() for s in df[col_busca].dropna() for x in str(s).split(';') if x.strip()])
             todos_itens.discard(termo_ativo)
             for cand in todos_itens:
                 sub_cand = df[df[col_busca].fillna('').str.contains(cand, regex=False)]
                 k, a, v = extrair_features(sub_cand)
                 dna_cand = k.union(a).union(v)
                 inter = dna_alvo.intersection(dna_cand)
                 if inter:
                     jaccard = len(inter) / len(dna_alvo.union(dna_cand))
                     resultados.append({'Item': cand, 'Similaridade (%)': round(jaccard * 100, 1), 'Traços em Comum': " | ".join(list(inter)[:4])})

    # Ordena, remove quem não tem nada a ver e corta nos top 15
    resultados = sorted([r for r in resultados if r['Similaridade (%)'] > 0], key=lambda x: x['Similaridade (%)'], reverse=True)[:15]
    
    if tipo_busca == "Documento": return {'Documentos': resultados}
    elif tipo_busca == "Autor": return {'Autores': resultados}
    else: return {'Itens': resultados}

def limpar_termo_busca():
    """Limpa o termo de busca quando o usuário clica manualmente no botão de rádio."""
    st.session_state['busca_termo_biblio'] = None

def navegar_busca(novo_tipo, novo_termo):
    """Atualiza o estado global para mudar o perfil exibido no motor de busca."""
    st.session_state['busca_tipo_biblio'] = novo_tipo
    st.session_state['busca_termo_biblio'] = novo_termo

def gerar_nuvem_echarts(df, coluna, fonte="Arial", paleta=None):
    """Gera o dicionário nativo da nuvem de palavras, livre de erros de conversão JS."""
    texto = " ".join(df[coluna].dropna().astype(str)).lower()
    if not texto.strip():
        return None

    stopwords = set(STOPWORDS)
    stopwords.update(["research", "study", "analysis", "results", "using", "paper", "article", "author", "will", "may", "can"])

    palavras_limpas = re.findall(r'\b\w{3,}\b', texto)
    palavras_filtradas = [w for w in palavras_limpas if w not in stopwords]
    contagem = Counter(palavras_filtradas).most_common(150)

    if not paleta:
        paleta = ["#0077b6", "#00b4d8", "#90e0ef", "#03045e", "#023e8a"]

    # Atribuímos a cor individualmente para cada palavra aqui mesmo no Python
    dados_palavras = []
    for palavra, freq in contagem:
        dados_palavras.append({
            "name": palavra,
            "value": freq,
            "textStyle": {
                "color": random.choice(paleta)
            }
        })

    # Dicionário puro e perfeito que o ECharts entende instantaneamente
    opcoes_echarts = {
        "tooltip": {"show": True},
        "toolbox": {
            "feature": {
                "saveAsImage": {"show": True, "title": "Baixar Nuvem", "type": "png"}
            }
        },
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "sizeRange": [15, 80],
            "rotationRange": [-45, 90],
            "rotationStep": 45,
            "gridSize": 8,
            "textStyle": {
                "fontFamily": fonte,
                "fontWeight": "bold"
            },
            "data": dados_palavras
        }]
    }

    return opcoes_echarts

def process_multiple_ris(uploaded_files, db_mapping):
    """Lê múltiplos arquivos RIS e retorna um DataFrame padronizado."""
    all_entries = []
    
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            entries = rispy.load(stringio)
            base_origem = db_mapping.get(uploaded_file.name, "Outra")
            
            for entry in entries:
                entry['Base_de_Dados'] = base_origem
                if 'unknown_tag' in entry and isinstance(entry['unknown_tag'], dict):
                    unknown_dict = entry.pop('unknown_tag')
                    for key, value in unknown_dict.items():
                        if isinstance(value, list):
                            entry[key] = "; ".join(value)
                        else:
                            entry[key] = value
                elif 'unknown_tag' in entry:
                    entry.pop('unknown_tag')
                all_entries.append(entry)
        except Exception as e:
            continue 
            
    if all_entries:
        df = pd.DataFrame(all_entries)
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: "; ".join([str(i) for i in x]) if isinstance(x, list) else x)
        df.columns = [str(col).upper().replace('_', ' ') for col in df.columns]
        
        if 'YEAR' in df.columns:
            df['YEAR CLEAN'] = pd.to_numeric(df['YEAR'], errors='coerce')
            
        # --- LÓGICA DE EXTRAÇÃO DE CITAÇÕES ---
        def extract_citations(row):
            for col in ['TC', 'Z9', 'TIMES CITED', 'CITED BY']:
                if col in df.columns and pd.notna(row[col]):
                    try: return float(row[col])
                    except: pass
            
            if 'NOTES' in df.columns and pd.notna(row['NOTES']):
                notes_str = str(row['NOTES'])
                match_scopus = re.search(r'Cited\s+By:\s*(\d+)', notes_str, re.IGNORECASE)
                if match_scopus: return float(match_scopus.group(1))
                match_wos = re.search(r'Times\s+Cited(?:.*?):\s*(\d+)', notes_str, re.IGNORECASE)
                if match_wos: return float(match_wos.group(1))
            
            return None 
            
        # 1º PASSO: Cria a coluna usando a regra acima
        df['TOTAL CITATIONS'] = df.apply(extract_citations, axis=1)
        
        # 2º PASSO: Converte a coluna recém-criada para numérico (evita o erro do nlargest)
        df['TOTAL CITATIONS'] = pd.to_numeric(df['TOTAL CITATIONS'], errors='coerce')
        
        # --- LIMPEZA DO TIPO DE REFERÊNCIA ---
        if 'TYPE OF REFERENCE' in df.columns:
            df['TYPE OF REFERENCE'] = (
                df['TYPE OF REFERENCE']
                .astype(str)
                .str.replace('label.ris.referenceType.', '', regex=False)
                .str.replace('_', ' ')
                .str.title()
            )

        addr_col = next((c for c in ['AUTHOR ADDRESS', 'AD', 'C1', 'AFFILIATIONS'] if c in df.columns), None)
        
        if addr_col:
            # Dicionário geográfico abrangente
            PAISES = [
                "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
                "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
                "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Peoples R China", "Taiwan", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Czechoslovakia",
                "Denmark", "Djibouti", "Dominica", "Dominican Republic",
                "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
                "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
                "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
                "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "North Korea", "South Korea", "Kuwait", "Kyrgyzstan",
                "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
                "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
                "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway",
                "Oman", "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
                "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
                "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
                "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "UK", "England", "Scotland", "Wales", "North Ireland", "USA", "United States", "U S A", "U.S.A.", "Uruguay", "Uzbekistan",
                "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
            ]
            
            # Mapa para traduzir variações para o nome padrão oficial
            MAPA_PAISES = {
                "peoples r china": "China",
                "taiwan": "Taiwan",
                "usa": "USA",
                "u s a": "USA",
                "u.s.a.": "USA",
                "united states": "USA",
                "uk": "United Kingdom",
                "england": "United Kingdom",
                "scotland": "United Kingdom",
                "wales": "United Kingdom",
                "north ireland": "United Kingdom"
            }

            # Compila o regex com limites de palavras (\b) para buscar o país exato e ignorar o resto
            paises_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, PAISES)) + r')\b', re.IGNORECASE)

            def extract_countries_robust(text):
                if pd.isna(text) or not str(text).strip():
                    return None
                
                # Ignora propositalmente as linhas de sujeira clássica do RIS
                text_clean = re.sub(r'\b(PU|C3|C1|AD|FU)\s+-\s+', ' ', str(text))
                
                found = set()
                # O Regex procura apenas os nomes geográficos na string bagunçada
                for match in paises_pattern.finditer(text_clean):
                    pais_encontrado = match.group(1).lower()
                    # Passa pelo mapa corretor (se for peoples r china, vira China)
                    pais_padrao = MAPA_PAISES.get(pais_encontrado, pais_encontrado.title())
                    found.add(pais_padrao)
                    
                return "; ".join(sorted(list(found))) if found else None

            df['COUNTRY'] = df[addr_col].apply(extract_countries_robust)
        else:
            df['COUNTRY'] = None

        return df
    return None

def deduplicar_por_doi(df):
    # Criamos uma cópia e ordenamos: mais citações primeiro, NaNs por último
    df_clean = df.sort_values(by='TOTAL CITATIONS', ascending=False, na_position='last').copy()
    
    doi_col = next((c for c in ['DOI', 'DO'] if c in df_clean.columns), None)
    title_col = next((c for c in ['TITLE', 'TI'] if c in df_clean.columns), None)
    
    if not doi_col: 
        return df_clean, pd.DataFrame()

    valid_doi = df_clean[df_clean[doi_col].notna() & (df_clean[doi_col] != '')]
    
    # Ao usar keep='first' em um DF ordenado, ele mantém o registro com mais citações
    first_occurrence = valid_doi.groupby(doi_col).apply(lambda x: x.index[0]).to_dict()
    dupe_mask = valid_doi.duplicated(subset=[doi_col], keep='first')
    dupes_indices = valid_doi[dupe_mask].index
    
    df_dupes = df_clean.loc[dupes_indices].copy()
    
    # Preenche a nova coluna com o título do documento mantido
    if not df_dupes.empty and title_col:
        ref_titles = [df_clean.loc[first_occurrence[doi], title_col] for doi in df_dupes[doi_col]]
        df_dupes['DOCUMENTO DE REFERÊNCIA (MANTIDO)'] = ref_titles
        
    df_unified = df_clean.drop(index=dupes_indices).copy()
    return df_unified, df_dupes

def deduplicar_por_similaridade(df, threshold=0.90):
    
    # Ordenação crucial: coloca os mais citados no topo da lista de comparação
    df_clean = df.sort_values(by='TOTAL CITATIONS', ascending=False, na_position='last').copy()
    
    title_col = next((c for c in ['TITLE', 'TI'] if c in df_clean.columns), None)
    
    if not title_col or len(df_clean) < 2: 
        return df_clean, pd.DataFrame()

    # Como o DF está ordenado, indices_reais[0] terá mais citações que indices_reais[10]
    indices_reais = df_clean.index.tolist()
    
    temp_titles = df_clean[title_col].astype(str).str.lower().str.strip()
    vectorizer = TfidfVectorizer(stop_words='english')
    
    indices_para_excluir = set()
    ref_mapping = {}

    try:
        tfidf_matrix = vectorizer.fit_transform(temp_titles)
        cosine_sim = cosine_similarity(tfidf_matrix)
        upper_tri = np.triu(cosine_sim, k=1)
        
        # Encontra as posições onde a similaridade é alta
        rows_pos, cols_pos = np.where(upper_tri >= threshold)
        
        for r_p, c_p in zip(rows_pos, cols_pos):
            # Mapeia a posição (0, 1, 2...) de volta para o Index real (ex: 520, 800...)
            idx_r = indices_reais[r_p] # Documento que será mantido
            idx_c = indices_reais[c_p] # Documento identificado como duplicado
            
            if idx_c not in indices_para_excluir and idx_r not in indices_para_excluir:
                indices_para_excluir.add(idx_c)
                ref_mapping[idx_c] = df_clean.loc[idx_r, title_col]
                
    except Exception as e:
        print(f"Erro na similaridade: {e}")

    df_dupes = df_clean.loc[list(indices_para_excluir)].copy()
    
    if not df_dupes.empty:
        df_dupes['DOCUMENTO DE REFERÊNCIA (MANTIDO)'] = [ref_mapping[idx] for idx in list(indices_para_excluir)]
        
    df_unified = df_clean.drop(index=list(indices_para_excluir)).copy()
    return df_unified, df_dupes    
