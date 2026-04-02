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
# (As funções criar_grafo_e_metricas permanecem inalteradas. Mantenha-as aqui como na resposta anterior)
def criar_grafo_e_metricas(df, coluna, top_n, metric_for_size="Tamanho Fixo"):
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
    
    if metric_for_size != "Tamanho Fixo" and metric_for_size in metric_dict:
        m_dict = metric_dict[metric_for_size]
        min_m, max_m = (min(m_dict.values()), max(m_dict.values())) if m_dict.values() else (0, 1)
        for node in G.nodes():
            nodes_agraph.append(Node(id=node, label=node, size=get_scaled_size(m_dict.get(node, 0), min_m, max_m), color="#1273B9", font=font_config))
    else:
        for node in G.nodes():
            nodes_agraph.append(Node(id=node, label=node, size=25, color="#1273B9", font=font_config))

    edges_agraph = [Edge(source=u, target=v, value=d['weight'], color="#E0E0E0") for u, v, d in G.edges(data=True)]

    net_metrics = {}
    if len(G) > 0:
        net_metrics['densidade'] = nx.density(G)
        try: net_metrics['eficiencia'] = nx.global_efficiency(G)
        except: net_metrics['eficiencia'] = 0
        try: net_metrics['clustering'] = nx.average_clustering(G)
        except: net_metrics['clustering'] = 0
        
        deg_vals = list(degree.values())
        if deg_vals:
            net_metrics['mean_links'] = np.mean(deg_vals)
            net_metrics['std_links'] = np.std(deg_vals)
            net_metrics['min_links'] = np.min(deg_vals)
            net_metrics['max_links'] = np.max(deg_vals)
            deg_counts = Counter(deg_vals)
            probs = [c / len(deg_vals) for c in deg_counts.values()]
            net_metrics['entropia'] = -np.sum([p * np.log2(p) for p in probs if p > 0])
            k, pk = np.array(list(deg_counts.keys())), np.array(list(deg_counts.values())) / len(deg_vals)
            valid = (k > 0) & (pk > 0)
            if np.sum(valid) > 1:
                slope, _ = np.polyfit(np.log10(k[valid]), np.log10(pk[valid]), 1)
                net_metrics['powerlaw'] = abs(slope)
            else: net_metrics['powerlaw'] = 0
        
        net_metrics['pagerank'] = np.mean(list(nx.pagerank(G).values())) if len(G)>0 else 0
        net_metrics['eigen_mean'] = np.mean(list(eigenvector.values())) if eigenvector else 0
        try: net_metrics['constraint'] = np.mean(list(nx.constraint(G).values()))
        except: net_metrics['constraint'] = 0
        try:
            eff_size = nx.effective_size(G)
            net_metrics['redundancia'] = np.mean([degree[n] - eff_size[n] for n in G.nodes() if degree[n] > 0])
        except: net_metrics['redundancia'] = 0
        try: net_metrics['assortatividade'] = nx.degree_assortativity_coefficient(G)
        except: net_metrics['assortatividade'] = 0
        try:
            rho, _ = stats.spearmanr(list(degree.values()), list(betweenness.values()))
            net_metrics['spearman'] = rho if not np.isnan(rho) else 0
        except: net_metrics['spearman'] = 0
        try:
            if max(deg_vals) > 1:
                rc = nx.rich_club_coefficient(G, normalized=False)
                t_k = int(max(deg_vals) * 0.8)
                t_k = t_k if t_k in rc else max(rc.keys())
                net_metrics['rich_club'] = rc[t_k]
            else: net_metrics['rich_club'] = 0
        except: net_metrics['rich_club'] = 0

    return nodes_agraph, edges_agraph, df_nodes, net_metrics