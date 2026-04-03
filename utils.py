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
    bet_cent = nx.betweenness_centrality(G)
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

# --- CAMADA DE INTERFACE (Barra de Progresso - SEM CACHE DIRETO) ---

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

def criar_grafo_e_metricas(df, coluna, top_n, metric_for_size="Tamanho Fixo", _pbar=None):
    """Interface para o grafo agraph. Agora com métricas recuperadas da engine."""
    docs_items = []
    for row in df[coluna].dropna():
        items = [x.strip() for x in str(row).split(';') if x.strip()]
        if items: docs_items.append(items)

    all_items = [item for sublist in docs_items for item in sublist]
    item_counts = Counter(all_items)
    top_items = set([x[0] for x in item_counts.most_common(top_n)])
    
    edges_list = []
    total_docs = len(docs_items)
    for i, items in enumerate(docs_items):
        if _pbar: _pbar.progress((i + 1) / total_docs, text=f"Tecendo redes: {i+1}/{total_docs}")
        filtered = [x for x in items if x in top_items]
        if len(filtered) > 1:
            edges_list.extend(list(combinations(sorted(filtered), 2)))

    # Chamada da Engine com Unpack completo das métricas
    node_types_gen = {node: "Entidade" for node in top_items}
    if _pbar: _pbar.progress(1.0, text="Calculando topologia SNA...")
    
    res_list, deg, eigen, betw, clos = _engine_calculo_sna(list(top_items), list(set(edges_list)), node_types_gen)
    
    # Criamos o DataFrame para a interface
    df_nodes = pd.DataFrame(res_list).rename(columns={"Item": "Nó", "Centralidade (Eigen)": "Centralidade (Eigenvector)"})
    
    # Mapeamento para o redimensionamento dos nós
    metric_dict = {
        "Grau Absoluto": deg, 
        "Centralidade (Eigen)": eigen, 
        "Betweenness": betw, 
        "Closeness": clos
    }
    
    def get_scaled_size(val, min_val, max_val, min_size=15, max_size=55):
        if max_val == min_val: return min_size
        return min_size + (val - min_val) * (max_size - min_size) / (max_val - min_val)

    nodes_agraph = []
    font_config = {"color": "black", "strokeWidth": 3, "strokeColor": "white"}
    
    if metric_for_size != "Tamanho Fixo" and metric_for_size in metric_dict:
        m_dict = metric_dict[metric_for_size]
        values = list(m_dict.values())
        min_m, max_m = (min(values), max(values)) if values else (0, 1)
        for node in top_items:
            nodes_agraph.append(Node(id=node, label=node, size=get_scaled_size(m_dict.get(node, 0), min_m, max_m), color="#1273B9", font=font_config))
    else:
        for node in top_items:
            nodes_agraph.append(Node(id=node, label=node, size=25, color="#1273B9", font=font_config))

    edges_agraph = [Edge(source=u, target=v, color="#E0E0E0") for u, v in list(set(edges_list))]

    return nodes_agraph, edges_agraph, df_nodes, {}

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
        'Authors': 'AUTHORS'
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
        'DOI': 'DOI'
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

    def extrair_features(df_subset):
        """Extrai as impressões digitais da entidade (Assuntos, Parceiros, Locais)."""
        kws, aus, venues = set(), set(), set()
        for _, r in df_subset.iterrows():
            if col_kw and pd.notna(r[col_kw]): 
                kws.update([k.strip().lower() for k in str(r[col_kw]).split(';') if k.strip()])
            if col_autores and pd.notna(r[col_autores]): 
                aus.update([a.strip() for a in str(r[col_autores]).split(';') if a.strip()])
            if col_venue and pd.notna(r[col_venue]): 
                venues.add(str(r[col_venue]).strip())
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
