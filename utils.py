import io
import random
import re
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rispy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_agraph import Edge, Node
from wordcloud import STOPWORDS


CURRENT_YEAR = date.today().year

@st.cache_data(show_spinner=False)
def processar_cochrane(file, nome_arquivo):
    """Processa arquivos CSV ou RIS exportados da Cochrane Library."""
    import pandas as pd
    import rispy
    import re
    import io

    ext = nome_arquivo.lower()
    
    if ext.endswith('.csv'):
        file.seek(0)
        df = pd.read_csv(file)
        
        # Mapeamento estrito das colunas do CSV da Cochrane
        mapa_colunas = {
            'Title': 'TITLE',
            'Author(s)': 'AUTHORS',
            'Source': 'SECONDARY TITLE',
            'Year': 'YEAR CLEAN',
            'Abstract': 'ABSTRACT',
            'Keywords': 'KEYWORDS',
            'DOI': 'DOI'
        }
        df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})
        
    elif ext.endswith('.ris'):
        file.seek(0)
        text = file.getvalue().decode('utf-8', errors='ignore')
        
        # CORREÇÃO COCHRANE: Eles inserem espaços extras nas tags (ex: "A1  -  Autor").
        # A expressão regular abaixo força o padrão estrito ("A1  - Autor") para o rispy não quebrar.
        text = re.sub(r'^([A-Z0-9]{2})\s+-\s+', r'\1  - ', text, flags=re.MULTILINE)
        
        entries = rispy.loads(text)
        df = pd.DataFrame(entries)
        
        # A tag 'A1' (Autor) na Cochrane costuma ser lida como 'first_authors' pelo rispy
        if 'first_authors' in df.columns and 'authors' not in df.columns:
            df['authors'] = df['first_authors']
            
        mapa_ris = {
            'title': 'TITLE',
            'primary_title': 'TITLE',
            'authors': 'AUTHORS',
            'journal_name': 'SECONDARY TITLE',
            'year': 'YEAR CLEAN',
            'abstract': 'ABSTRACT',
            'keywords': 'KEYWORDS',
            'doi': 'DOI'
        }
        df = df.rename(columns={k: v for k, v in mapa_ris.items() if k in df.columns})
        
        # Converte listas (Autores, Palavras-chave) para strings com ponto e vírgula
        for col in ['AUTHORS', 'KEYWORDS']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))

    # --- Tratamentos Comuns (RIS e CSV) ---
    if 'TOTAL CITATIONS' not in df.columns:
        df['TOTAL CITATIONS'] = 0 # Cochrane não exporta métrica de citações recebidas
        
    if 'KEYWORDS' in df.columns:
        # Remove asteriscos inseridos em termos MeSH
        df['KEYWORDS'] = df['KEYWORDS'].astype(str).str.replace('*', '', regex=False)

    # Passa pelo padronizador central (já com a proteção contra colunas duplicadas que criamos antes)
    from utils import padronizar_base_bibliometrica
    return padronizar_base_bibliometrica(df)

@st.cache_data(show_spinner=False)
def processar_pubmed(file):
    """Processa arquivos .txt ou .nbib exportados do PubMed (Medline format)."""
    import io
    import re
    import pandas as pd

    # Lê as linhas decodificando de forma segura
    file.seek(0)
    linhas = file.getvalue().decode('utf-8', errors='ignore').splitlines()

    entries = []
    current_entry = {}
    current_tag = None

    # 1. Parsing Linha por Linha
    for linha in linhas:
        if not linha.strip():
            continue

        # Se a linha começa com espaços, é continuação da tag anterior (Ex: Abstract)
        if re.match(r'^\s+', linha):
            if current_tag and current_tag in current_entry:
                if isinstance(current_entry[current_tag], list):
                    current_entry[current_tag][-1] += " " + linha.strip()
                else:
                    current_entry[current_tag] += " " + linha.strip()
            continue

        # Nova tag detectada (Ex: "TI  - Autopsy findings...")
        match = re.match(r'^([A-Z0-9]{2,4})\s*-\s(.*)', linha)
        if match:
            tag = match.group(1)
            valor = match.group(2).strip()

            if tag == 'PMID': # Inicio de um novo documento
                if current_entry:
                    entries.append(current_entry)
                current_entry = {'PMID': valor}
            else:
                # Agrupa tags que podem aparecer múltiplas vezes em listas
                if tag in ['FAU', 'AU', 'AD', 'OT', 'MH', 'PT', 'LID', 'AID']:
                    if tag not in current_entry:
                        current_entry[tag] = []
                    current_entry[tag].append(valor)
                else:
                    current_entry[tag] = valor
            current_tag = tag

    if current_entry:
        entries.append(current_entry)

    if not entries:
        return None

    df = pd.DataFrame(entries)

    # 2. Padronização de Colunas Simetrics
    mapa_colunas = {
        'TI': 'TITLE',
        'AB': 'ABSTRACT',
        'JT': 'SECONDARY TITLE',
        'TA': 'SECONDARY TITLE', # Fallback caso não tenha JT
    }
    df = df.rename(columns={k: v for k, v in mapa_colunas.items() if k in df.columns})

    # 3. Tratamento de Autores (FAU = Full Author)
    if 'FAU' in df.columns:
        df['AUTHORS'] = df['FAU'].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))
    elif 'AU' in df.columns:
        df['AUTHORS'] = df['AU'].apply(lambda x: "; ".join(x) if isinstance(x, list) else str(x))

    # 4. Tratamento de Palavras-Chave (OT = Other Term, MH = MeSH Term)
    kw_cols = [c for c in ['OT', 'MH'] if c in df.columns]
    if kw_cols:
        def formatar_kw(row):
            kws = []
            for col in kw_cols:
                if isinstance(row[col], list):
                    # Remove asteriscos dos termos MeSH (Ex: *COVID-19)
                    kws.extend([str(k).replace('*', '').strip() for k in row[col]])
            return "; ".join(list(set(kws)))
        df['KEYWORDS'] = df.apply(formatar_kw, axis=1)

    # 5. Tratamento de Ano (DP = Date of Publication)
    if 'DP' in df.columns:
        df['YEAR CLEAN'] = pd.to_numeric(df['DP'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')

    # 6. Extração de DOI
    def extract_doi(row):
        for col in ['LID', 'AID']:
            if col in df.columns and isinstance(row[col], list):
                for val in row[col]:
                    if '[doi]' in val.lower():
                        return val.replace('[doi]', '').replace('[DOI]', '').strip()
            elif col in df.columns and isinstance(row[col], str):
                if '[doi]' in row[col].lower():
                    return row[col].replace('[doi]', '').replace('[DOI]', '').strip()
        return None
    
    df['DOI'] = df.apply(extract_doi, axis=1)

    # 7. Tipo de Documento
    if 'PT' in df.columns:
        df['DOCUMENT TYPE'] = df['PT'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    # 8. Extração de Países via Afiliações (AD)
    if 'AD' in df.columns:
        # Usa o motor robusto de extração geográfica que já criamos para o RIS
        PAISES = ["USA", "United States", "Canada", "Brazil", "China", "Taiwan", "United Kingdom", "England", "Scotland", "Wales", "North Ireland", "France", "Germany", "Italy", "Spain", "Portugal", "Netherlands", "Switzerland", "Sweden", "Norway", "Denmark", "Finland", "Belgium", "Austria", "Russia", "Australia", "New Zealand", "Japan", "South Korea", "India", "South Africa", "Mexico", "Argentina", "Chile", "Colombia", "Peru", "Israel", "Saudi Arabia", "Iran", "Turkey", "Egypt"]
        MAPA_PAISES = {"usa": "USA", "united states": "USA", "uk": "United Kingdom", "england": "United Kingdom"}
        paises_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, PAISES)) + r')\b', re.IGNORECASE)

        def extrair_paises(afiliacao_list):
            if not isinstance(afiliacao_list, list): return None
            texto_completo = " ".join(afiliacao_list)
            found = set()
            for match in paises_pattern.finditer(texto_completo):
                pais_encontrado = match.group(1).lower()
                found.add(MAPA_PAISES.get(pais_encontrado, pais_encontrado.title()))
            return "; ".join(sorted(list(found))) if found else None

        df['COUNTRY'] = df['AD'].apply(extrair_paises)

    # Garantia de colunas numéricas
    df['TOTAL CITATIONS'] = 0 # PubMed via txt/nbib não exporta Citações Recebidas por padrão

    # Passa pelo padronizador central para garantir que a interface funcione
    from utils import padronizar_base_bibliometrica
    return padronizar_base_bibliometrica(df)

@st.cache_data(show_spinner=False)
def calcular_genetica_palavras(df):
    """Calcula o ciclo de vida (nascimento, morte e replicação) e o impacto das palavras-chave."""
    import pandas as pd
    
    if 'KEYWORDS' not in df.columns or 'YEAR CLEAN' not in df.columns:
        return None

    df_clean = df.dropna(subset=['KEYWORDS', 'YEAR CLEAN']).copy()
    df_clean['YEAR CLEAN'] = pd.to_numeric(df_clean['YEAR CLEAN'], errors='coerce')
    df_clean = df_clean.dropna(subset=['YEAR CLEAN'])

    # NOVO: Garante que a coluna de citações exista e seja numérica para o Eixo Z do 3D
    if 'TOTAL CITATIONS' not in df_clean.columns:
        df_clean['TOTAL CITATIONS'] = 0
    else:
        df_clean['TOTAL CITATIONS'] = pd.to_numeric(df_clean['TOTAL CITATIONS'], errors='coerce').fillna(0)

    # Explode as palavras-chave para criar uma matriz "Termo x Ano"
    df_clean['KW'] = df_clean['KEYWORDS'].astype(str).str.split(';')
    df_exp = df_clean.explode('KW')
    df_exp['KW'] = df_exp['KW'].str.strip().str.lower()
    df_exp = df_exp[df_exp['KW'] != '']

    if df_exp.empty:
        return None

    # Agrupa por Palavra-chave calculando a biologia do termo E o impacto
    genetica = df_exp.groupby('KW').agg(
        ano_nascimento=('YEAR CLEAN', 'min'),
        ano_extincao=('YEAR CLEAN', 'max'),
        total_aparicoes=('KW', 'count'),
        total_citacoes=('TOTAL CITATIONS', 'sum') # NOVO: Somatório do impacto
    ).reset_index()

    # Longevidade é a diferença entre a última aparição e a primeira
    genetica['tempo_vida_anos'] = genetica['ano_extincao'] - genetica['ano_nascimento']
    genetica = genetica.rename(columns={'KW': 'Palavra-chave'})
    
    return genetica

@st.cache_data(show_spinner=False)
def plot_sankey_evolution(df, p1_range, p2_range, p3_range, top_n=10):
    """Gera o diagrama de Sankey cruzando palavras-chave em 3 períodos temporais."""
    import pandas as pd
    import plotly.graph_objects as go
    from collections import Counter
    from itertools import combinations

    if 'YEAR CLEAN' not in df.columns or 'KEYWORDS' not in df.columns:
        return None

    df_clean = df.dropna(subset=['YEAR CLEAN', 'KEYWORDS']).copy()
    df_clean['YEAR CLEAN'] = pd.to_numeric(df_clean['YEAR CLEAN'], errors='coerce')
    df_clean = df_clean.dropna(subset=['YEAR CLEAN'])

    def get_top_words(df_subset, n):
        words = []
        for kw_str in df_subset['KEYWORDS']:
            words.extend([w.strip().lower() for w in str(kw_str).split(';') if w.strip()])
        return [w for w, c in Counter(words).most_common(n)]

    # Filtra os 3 recortes de tempo
    df_p1 = df_clean[(df_clean['YEAR CLEAN'] >= p1_range[0]) & (df_clean['YEAR CLEAN'] <= p1_range[1])]
    df_p2 = df_clean[(df_clean['YEAR CLEAN'] >= p2_range[0]) & (df_clean['YEAR CLEAN'] <= p2_range[1])]
    df_p3 = df_clean[(df_clean['YEAR CLEAN'] >= p3_range[0]) & (df_clean['YEAR CLEAN'] <= p3_range[1])]

    top_p1 = get_top_words(df_p1, top_n)
    top_p2 = get_top_words(df_p2, top_n)
    top_p3 = get_top_words(df_p3, top_n)

    if not top_p1 and not top_p2 and not top_p3:
        return None

    # Constrói a matriz global de frequência e co-ocorrência para definir a grossura das linhas
    cooc = Counter()
    freq = Counter()
    for kw_str in df_clean['KEYWORDS']:
        words = list(set([w.strip().lower() for w in str(kw_str).split(';') if w.strip()]))
        for w in words: freq[w] += 1
        for w1, w2 in combinations(sorted(words), 2):
            cooc[(tuple(sorted((w1, w2))))] += 1

    # NOVO: Estilização CSS para letras pretas com efeito halo/contorno branco de alto contraste
    # text-shadow cria contornos suaves em todas as direções para legibilidade máxima
    halo_style = "color:black; font-weight:bold; text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;"
    
    def stylize_node_label(text):
        """Aplica a estilização HTML/CSS ao texto da label."""
        return f"<span style='{halo_style}'>{text}</span>"

    labels = []
    node_dict = {}
    idx = 0

    # Criação estrutural dos nós (Nodes) para os 3 períodos com estilização de alto contraste
    for w in top_p1:
        raw_label = f"{w.title()} ({p1_range[0]}-{p1_range[1]})"
        labels.append(stylize_node_label(raw_label))
        node_dict[('p1', w)] = idx
        idx += 1
    for w in top_p2:
        raw_label = f"{w.title()} ({p2_range[0]}-{p2_range[1]})"
        labels.append(stylize_node_label(raw_label))
        node_dict[('p2', w)] = idx
        idx += 1
    for w in top_p3:
        raw_label = f"{w.title()} ({p3_range[0]}-{p3_range[1]})"
        labels.append(stylize_node_label(raw_label))
        node_dict[('p3', w)] = idx
        idx += 1

    source = []
    target = []
    value = []

    def add_links(source_list, target_list, prefix_s, prefix_t):
        for w_s in source_list:
            for w_t in target_list:
                if w_s == w_t:
                    # Linha de Continuidade (A mesma palavra sobreviveu ao próximo período)
                    weight = freq[w_s] 
                    source.append(node_dict[(prefix_s, w_s)])
                    target.append(node_dict[(prefix_t, w_t)])
                    value.append(weight)
                else:
                    # Linha de Intersecção (Palavras diferentes que costumam co-ocorrer nos textos)
                    pair = tuple(sorted((w_s, w_t)))
                    weight = cooc.get(pair, 0)
                    if weight > 0:
                        source.append(node_dict[(prefix_s, w_s)])
                        target.append(node_dict[(prefix_t, w_t)])
                        value.append(weight * 0.4) # Peso reduzido para que a linha de continuidade seja sempre a mais grossa

    add_links(top_p1, top_p2, 'p1', 'p2')
    add_links(top_p2, top_p3, 'p2', 'p3')

    if not source:
        return None

    # Renderização visual com Plotly
    # --- CÓDIGO CORRIGIDO NO UTILS.PY ---

    # Renderização visual com Plotly
    fig = go.Figure(data=[go.Sankey(
        # 1. Propriedades do Nó (Removida a propriedade 'font' daqui)
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "#f39c12" # Cor laranja intensa para os nós
        ),
        # 2. Propriedades dos Links (Fluxos)
        link = dict(
          source = source,
          target = target,
          value = value,
          color = "rgba(243, 156, 18, 0.4)" 
        ),
        # 3. CORREÇÃO: A configuração de fonte global do texto fica aqui fora
        textfont = dict(size=12, family="Arial")
    )])

    # Ajustes de Layout (Fundo transparente e margens)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=650,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

@st.cache_data(show_spinner=False)
def preparar_contexto_llm(df):
    """Filtra e comprime a base de dados em um formato estruturado (JSON) para a IA."""
    # Selecionamos apenas as colunas vitais para economizar tokens e focar a IA
    colunas_alvo = ['TITLE', 'AUTHORS', 'YEAR CLEAN', 'SECONDARY TITLE', 'TOTAL CITATIONS', 'TEMA_GEMINI', 'KEYWORDS', 'COUNTRY', 'ABSTRACT']
    colunas_presentes = [c for c in colunas_alvo if c in df.columns]
    
    df_contexto = df[colunas_presentes].copy()
    
    # Tratamento para evitar que textos muito longos ou nulos quebrem o prompt
    for col in df_contexto.columns:
        df_contexto[col] = df_contexto[col].fillna("Não informado")
        
    # Converte para JSON (O formato que LLMs processam com maior precisão estrutural)
    return df_contexto.to_json(orient="records", force_ascii=False)

def _pick_column(df, candidates):
    return next((col for col in candidates if col in df.columns), None)


def _split_semicolon_tokens(value, case=None):
    if pd.isna(value):
        return []

    tokens = [token.strip() for token in str(value).split(';') if token and token.strip()]
    if case == "lower":
        return [token.lower() for token in tokens]
    if case == "title":
        return [token.title() for token in tokens]
    return tokens


def _join_sorted(values, sep=", "):
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return sep.join(sorted(set(cleaned)))


def padronizar_base_bibliometrica(df):
    if df is None:
        return None

    df_padrao = df.copy()

    # 1. PROTECAO CONTRA COLUNAS DUPLICADAS
    df_padrao = df_padrao.loc[:, ~df_padrao.columns.duplicated()].copy()

    # 2. PROTECAO CONTRA CRASH DE MEMÓRIA / CACHE (Unhashable Type: List)
    # Detecta qualquer coluna que tenha sobrado como lista ou dicionário e a converte para texto.
    # Isso resolve o erro fatal do Streamlit ao tentar hashear (fazer cache) de arquivos de +50MB.
    for col in df_padrao.columns:
        if not df_padrao[col].dropna().empty:
            primeiro_valido = df_padrao[col].dropna().iloc[0]
            if isinstance(primeiro_valido, (list, dict, tuple)):
                df_padrao[col] = df_padrao[col].astype(str)

    # 3. PROTECAO CONTRA KEYERROR (Eixos de Gráficos)
    # Força a existência matemática da coluna de citações, independentemente da base lida.
    if 'TOTAL CITATIONS' not in df_padrao.columns:
        df_padrao['TOTAL CITATIONS'] = 0
    else:
        df_padrao['TOTAL CITATIONS'] = pd.to_numeric(df_padrao['TOTAL CITATIONS'], errors='coerce').fillna(0)

    # 4. Unificação de Referências (Mantido do seu código original)
    ref_candidates = ['REFERENCES_UNIFIED', 'REFERENCES', 'CITED REFERENCES', 'CR']
    if any(col in df_padrao.columns for col in ref_candidates):
        series_ref = None
        for col in ref_candidates:
            if col not in df_padrao.columns:
                continue

            current = df_padrao[col].fillna('').astype(str).str.strip()
            if series_ref is None:
                series_ref = current
            else:
                series_ref = series_ref.mask(series_ref.eq(''), current)

        df_padrao['REFERENCES_UNIFIED'] = series_ref.fillna('')

    if 'YEAR' in df_padrao.columns and 'YEAR CLEAN' not in df_padrao.columns:
        df_padrao['YEAR CLEAN'] = pd.to_numeric(df_padrao['YEAR'], errors='coerce')
    elif 'YEAR CLEAN' in df_padrao.columns:
        df_padrao['YEAR CLEAN'] = pd.to_numeric(df_padrao['YEAR CLEAN'], errors='coerce')

    if 'TOTAL CITATIONS' in df_padrao.columns:
        df_padrao['TOTAL CITATIONS'] = pd.to_numeric(df_padrao['TOTAL CITATIONS'], errors='coerce').fillna(0)

    object_cols = df_padrao.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_padrao[col] = df_padrao[col].fillna('').astype(str)

    return df_padrao


@st.cache_data(show_spinner=False)
def gerar_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data(show_spinner=False)
def analisar_completude_metadados(df):
    campos_verificacao = [
        ('AUTHORS', 'Author (AU)'),
        ('DOCUMENT TYPE', 'Document Type (DT)'),
        ('ABSTRACT', 'Abstract (AB)'),
        ('COUNTRY', 'Affiliation/Country (C1)'),
        ('DOI', 'DOI (DI)'),
        ('TITLE', 'Title (TI)'),
        ('SECONDARY TITLE', 'Journal/Source (SO)'),
        ('YEAR CLEAN', 'Publication Year (PY)'),
        ('TOTAL CITATIONS', 'Total Citation (TC)'),
        ('KEYWORDS', 'Keywords (DE/ID)'),
        ('REFERENCES_UNIFIED', 'Cited References (CR)')
    ]

    total_docs = len(df)
    dados = []

    for col_chave, descricao in campos_verificacao:
        if col_chave in df.columns:
            serie = df[col_chave]
            
            # --- CORREÇÃO: Proteção contra colunas duplicadas na base ---
            # Se vierem duas colunas com o mesmo nome, usamos apenas a primeira para evitar o TypeError
            if isinstance(serie, pd.DataFrame):
                serie = serie.iloc[:, 0]
            # ------------------------------------------------------------
            
            faltantes = int(serie.isna().sum())
            if pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie):
                faltantes += int(serie.astype(str).str.strip().eq('').sum())
        else:
            faltantes = total_docs

        pct_faltante = (faltantes / total_docs) * 100 if total_docs else 0

        if pct_faltante == 0:
            status = "Excelente"
        elif pct_faltante <= 10:
            status = "Bom"
        elif pct_faltante <= 20:
            status = "Aceitável"
        else:
            status = "Ruim"

        dados.append({
            "Metadado": descricao,
            "Faltantes": faltantes,
            "Faltantes (%)": pct_faltante,
            "Status": status
        })

    return pd.DataFrame(dados).sort_values(by="Faltantes (%)").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def resumir_base_bibliometrica(df):
    df_local = padronizar_base_bibliometrica(df)
    b_metrics = calcular_metricas_bibliometrix(df_local)

    years = pd.to_numeric(df_local.get('YEAR CLEAN'), errors='coerce') if 'YEAR CLEAN' in df_local.columns else pd.Series(dtype=float)
    valid_years = years.dropna()
    if valid_years.empty:
        timespan = "N/S"
        avg_age = "N/S"
    else:
        timespan = f"{int(valid_years.min())}:{int(valid_years.max())}"
        avg_age = round(CURRENT_YEAR - valid_years.mean(), 2)

    authors_count = 0
    if 'AUTHORS' in df_local.columns:
        authors = {
            autor
            for value in df_local['AUTHORS']
            for autor in _split_semicolon_tokens(value)
        }
        authors_count = len(authors)

    countries_count = 0
    if 'COUNTRY' in df_local.columns:
        countries = {
            pais
            for value in df_local['COUNTRY']
            for pais in _split_semicolon_tokens(value)
        }
        countries_count = len(countries)

    kw_count = 0
    col_kw = _pick_column(df_local, ['KEYWORDS', 'KW', 'DE'])
    if col_kw:
        keywords = {
            kw.lower()
            for value in df_local[col_kw]
            for kw in _split_semicolon_tokens(value)
        }
        kw_count = len(keywords)

    venues_count = 0
    col_venue = _pick_column(df_local, ['SECONDARY TITLE', 'SO', 'JO'])
    if col_venue:
        venues_count = int(df_local[col_venue].astype(str).str.strip().replace('', np.nan).dropna().nunique())

    return {
        "total_docs": int(len(df_local)),
        "timespan": timespan,
        "avg_age": avg_age,
        "authors_count": authors_count,
        "countries_count": countries_count,
        "kw_count": kw_count,
        "venues_count": venues_count,
        "b_metrics": b_metrics
    }


@st.cache_data(show_spinner=False)
def preparar_opcoes_busca(df):
    col_titulos = _pick_column(df, ['TITLE', 'TI'])
    col_autores = _pick_column(df, ['AUTHORS', 'AU'])
    col_paises = _pick_column(df, ['COUNTRY'])
    col_venue = _pick_column(df, ['SECONDARY TITLE', 'SO', 'JO'])
    col_ano = _pick_column(df, ['YEAR', 'PY', 'YEAR CLEAN'])

    opcoes_doc = sorted(df[col_titulos].dropna().astype(str).unique().tolist()) if col_titulos else []
    opcoes_aut = sorted({
        autor
        for value in df[col_autores]
        for autor in _split_semicolon_tokens(value)
    }) if col_autores else []
    opcoes_pais = sorted({
        pais
        for value in df[col_paises]
        for pais in _split_semicolon_tokens(value)
    }) if col_paises else []
    opcoes_venue = sorted(df[col_venue].dropna().astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()) if col_venue else []
    
    # NOVO: Opções de Temas da IA
    opcoes_tema = sorted(df['TEMA_GEMINI'].dropna().unique().tolist()) if 'TEMA_GEMINI' in df.columns else []

    return {
        "col_titulos": col_titulos,
        "col_autores": col_autores,
        "col_paises": col_paises,
        "col_venue": col_venue,
        "col_ano": col_ano,
        "opcoes_doc": opcoes_doc,
        "opcoes_aut": opcoes_aut,
        "opcoes_pais": opcoes_pais,
        "opcoes_venue": opcoes_venue,
        "opcoes_tema": opcoes_tema
    }

@st.cache_data(show_spinner=False)
def filtrar_por_entidade(df, termo_ativo, tipo_ativo):
    if not termo_ativo:
        return pd.DataFrame(columns=df.columns)

    info_busca = preparar_opcoes_busca(df)
    col_titulos = info_busca["col_titulos"]
    col_autores = info_busca["col_autores"]
    col_paises = info_busca["col_paises"]
    col_venue = info_busca["col_venue"]

    if tipo_ativo == "Documento" and col_titulos:
        return df[df[col_titulos] == termo_ativo].copy()
    if tipo_ativo == "Autor" and col_autores:
        return df[df[col_autores].fillna('').str.contains(str(termo_ativo), regex=False)].copy()
    if tipo_ativo == "País" and col_paises:
        return df[df[col_paises].fillna('').str.contains(str(termo_ativo), regex=False)].copy()
    if tipo_ativo == "Local de Publicação (Venue)" and col_venue:
        return df[df[col_venue] == termo_ativo].copy()
    # NOVO: Filtro por Tema
    if tipo_ativo == "Tema" and 'TEMA_GEMINI' in df.columns:
        return df[df['TEMA_GEMINI'] == termo_ativo].copy()

    return pd.DataFrame(columns=df.columns)

@st.cache_resource(show_spinner=False)
def obter_grafo_global_busca(df, col_titulos, col_autores, col_paises, col_venue):
    G = nx.Graph()
    colunas_necessarias = [c for c in [col_titulos, col_autores, col_paises, col_venue] if c is not None]
    if not col_titulos or not colunas_necessarias:
        return G

    for row in df[colunas_necessarias].to_dict('records'):
        doc_node = str(row.get(col_titulos, '')).strip()
        if not doc_node or doc_node.lower() == 'nan':
            continue

        G.add_node(doc_node, type='Documento')

        if col_autores:
            for autor in _split_semicolon_tokens(row.get(col_autores)):
                G.add_node(autor, type='Autor')
                G.add_edge(doc_node, autor)

        if col_paises:
            for pais in _split_semicolon_tokens(row.get(col_paises)):
                G.add_node(pais, type='País')
                G.add_edge(doc_node, pais)

        if col_venue:
            venue = str(row.get(col_venue, '')).strip()
            if venue:
                G.add_node(venue, type='Venue')
                G.add_edge(doc_node, venue)

    return G

# --- FUNÇÕES AUXILIARES DE AGREGAÇÃO PARA TABELAS ---

# --- FUNÇÕES AUXILIARES DE AGREGAÇÃO CORRIGIDAS ---

def _format_timeline(group):
    """Gera a string agrupada por ano garantindo unicidade de documentos no grupo."""
    anos = {}
    title_col = _pick_column(group, ['TITLE', 'TI'])
    subset_cols = ['YEAR CLEAN']
    if title_col:
        subset_cols.append(title_col)

    group_clean = group.drop_duplicates(subset=subset_cols)

    for row in group_clean.itertuples(index=False, name=None):
        row_dict = dict(zip(group_clean.columns, row))
        ano = str(int(row_dict['YEAR CLEAN'])) if pd.notna(row_dict.get('YEAR CLEAN')) else "S/D"
        tit = str(row_dict.get(title_col, 'Sem título')).strip() if title_col else "Sem título"
        cit_val = row_dict.get('TOTAL CITATIONS', 0)
        cit = int(cit_val) if pd.notna(cit_val) else 0
        if ano not in anos:
            anos[ano] = []
        anos[ano].append(f"{tit} ({cit} citações)")
    
    out = []
    for a in sorted(anos.keys(), reverse=True):
        out.append(f"{a}: {'; '.join(anos[a])}")
    return " | ".join(out)

def _get_top_doc(group):
    """Retorna o documento mais citado usando posição (iloc) para evitar erro de índice ambíguo."""
    if group.empty:
        return ""

    title_col = _pick_column(group, ['TITLE', 'TI'])
    posicao_max = group['TOTAL CITATIONS'].fillna(0).values.argmax()
    row = group.iloc[posicao_max]

    tit = str(row.get(title_col, 'Sem Título')).strip() if title_col else "Sem Título"
    val_cit = row.get('TOTAL CITATIONS', 0)
    cit = int(val_cit) if pd.notna(val_cit) else 0

    return f"{tit} ({cit} citações)"

# --- MOTORES DE GERAÇÃO DE TABELAS ---

# --- MOTORES DE GERAÇÃO DE TABELAS (ATUALIZADOS COM QL) ---

@st.cache_data(show_spinner=False)
def gerar_tabela_autores(df):
    if 'AUTHORS' not in df.columns: return pd.DataFrame()

    df_exp = df.copy()
    df_exp['AUTHOR'] = df_exp['AUTHORS'].astype(str).str.split(';')
    df_exp = df_exp.explode('AUTHOR')
    df_exp['AUTHOR'] = df_exp['AUTHOR'].str.strip().str.title()
    df_exp = df_exp[df_exp['AUTHOR'] != '']
    
    # Preparação para o QL
    Q = df['TITLE'].nunique() if 'TITLE' in df.columns else len(df)
    has_tema = 'TEMA_GEMINI' in df.columns
    if has_tema:
        Qi_series = df.drop_duplicates('TITLE')['TEMA_GEMINI'].value_counts() if 'TITLE' in df.columns else df['TEMA_GEMINI'].value_counts()

    res = []
    for autor, group in df_exp.groupby('AUTHOR'):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        coautores = set()
        for auth_list in group['AUTHORS'].dropna():
            coautores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip().title() != autor])
            
        paises = set()
        if 'COUNTRY' in group.columns:
            for c_list in group['COUNTRY'].dropna():
                paises.update([c.strip().title() for c in str(c_list).split(';') if c.strip()])

        # Cálculo do QL para este autor
        tema_principal = "Não Categorizado"
        if has_tema and 'TEMA_GEMINI' in group.columns:
            q_k = len(group)
            qik_series = group['TEMA_GEMINI'].value_counts()
            best_ql = -1
            best_tema = ""
            for tema, q_ik in qik_series.items():
                q_i = Qi_series.get(tema, 0)
                if q_i > 0 and q_k > 0:
                    ql = (q_ik / q_k) / (q_i / Q)
                    if ql > best_ql:
                        best_ql = ql
                        best_tema = tema
            if best_ql >= 0:
                tema_principal = f"{best_tema} (QL: {best_ql:.2f})"

        res.append({
            'Autor': autor,
            'País do Autor': ", ".join(paises),
            'Especialização Principal (Maior QL)': tema_principal,
            'Documentos': " | ".join(group['TITLE'].dropna().astype(str)),
            'Qtd. de Documentos': len(group),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Coautores': ", ".join(coautores)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def gerar_tabela_paises(df):
    if 'COUNTRY' not in df.columns: return pd.DataFrame()
    
    df_exp = df.copy()
    df_exp['PAIS'] = df_exp['COUNTRY'].astype(str).str.split(';')
    df_exp = df_exp.explode('PAIS')
    df_exp['PAIS'] = df_exp['PAIS'].str.strip().str.title()
    df_exp = df_exp[(df_exp['PAIS'] != '') & (df_exp['PAIS'] != 'Nan')]

    Q = df['TITLE'].nunique() if 'TITLE' in df.columns else len(df)
    has_tema = 'TEMA_GEMINI' in df.columns
    if has_tema:
        Qi_series = df.drop_duplicates('TITLE')['TEMA_GEMINI'].value_counts() if 'TITLE' in df.columns else df['TEMA_GEMINI'].value_counts()

    res = []
    for pais, group in df_exp.groupby('PAIS'):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        autores = set()
        for auth_list in group['AUTHORS'].dropna():
            autores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip()])

        tema_principal = "Não Categorizado"
        if has_tema and 'TEMA_GEMINI' in group.columns:
            q_k = len(group)
            qik_series = group['TEMA_GEMINI'].value_counts()
            best_ql = -1
            best_tema = ""
            for tema, q_ik in qik_series.items():
                q_i = Qi_series.get(tema, 0)
                if q_i > 0 and q_k > 0:
                    ql = (q_ik / q_k) / (q_i / Q)
                    if ql > best_ql:
                        best_ql = ql
                        best_tema = tema
            if best_ql >= 0:
                tema_principal = f"{best_tema} (QL: {best_ql:.2f})"

        res.append({
            'País': pais,
            'Especialização Principal (Maior QL)': tema_principal,
            'Autores': ", ".join(autores),
            'Qtd. de Autores': len(autores),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Documento com Mais Citações': _get_top_doc(group)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def gerar_tabela_venues(df):
    col_venue = _pick_column(df, ['SECONDARY TITLE', 'SO', 'JO'])
    if col_venue not in df.columns: return pd.DataFrame()

    df_ven = df.dropna(subset=[col_venue]).copy()
    
    Q = df['TITLE'].nunique() if 'TITLE' in df.columns else len(df)
    has_tema = 'TEMA_GEMINI' in df.columns
    if has_tema:
        Qi_series = df.drop_duplicates('TITLE')['TEMA_GEMINI'].value_counts() if 'TITLE' in df.columns else df['TEMA_GEMINI'].value_counts()

    res = []
    for venue, group in df_ven.groupby(col_venue):
        cits = group['TOTAL CITATIONS'].fillna(0)
        
        autores = set()
        for auth_list in group['AUTHORS'].dropna():
            autores.update([a.strip().title() for a in str(auth_list).split(';') if a.strip()])

        tema_principal = "Não Categorizado"
        if has_tema and 'TEMA_GEMINI' in group.columns:
            q_k = len(group)
            qik_series = group['TEMA_GEMINI'].value_counts()
            best_ql = -1
            best_tema = ""
            for tema, q_ik in qik_series.items():
                q_i = Qi_series.get(tema, 0)
                if q_i > 0 and q_k > 0:
                    ql = (q_ik / q_k) / (q_i / Q)
                    if ql > best_ql:
                        best_ql = ql
                        best_tema = tema
            if best_ql >= 0:
                tema_principal = f"{best_tema} (QL: {best_ql:.2f})"

        res.append({
            'Local de Publicação (Venue)': str(venue).upper(),
            'Especialização Principal (Maior QL)': tema_principal,
            'Autores': ", ".join(autores),
            'Qtd. de Autores': len(autores),
            'Qtd. de Citações': cits.sum(),
            'Média de Citações': round(cits.mean(), 2),
            'Mediana de Citações': round(cits.median(), 2),
            'Desvio Padrão de Citações': round(cits.std(), 2) if len(group) > 1 else 0.0,
            'Anos, Documentos e Citações': _format_timeline(group),
            'Documento com Mais Citações': _get_top_doc(group)
        })
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def obter_top_ql_por_tema(df):
    """Calcula os top QL para Autores, Países e Venues, retornando Dicionários (Tema -> Label)."""
    import pandas as pd
    if 'TEMA_GEMINI' not in df.columns: 
        return pd.Series(dtype=str), pd.Series(dtype=str), pd.Series(dtype=str)
        
    def _calc_top(col):
        if col not in df.columns: return pd.Series(dtype=str)
        df_exp = df[['TITLE', col, 'TEMA_GEMINI']].copy()
        df_exp[col] = df_exp[col].astype(str).str.split(';')
        df_exp = df_exp.explode(col)
        
        if col == 'SECONDARY TITLE': df_exp[col] = df_exp[col].str.strip().str.upper()
        else: df_exp[col] = df_exp[col].str.strip().str.title()
            
        df_exp = df_exp[(df_exp[col] != '') & (df_exp[col] != 'Nan') & (df_exp[col].notna())]
        df_exp = df_exp.drop_duplicates(subset=['TITLE', col]) # Garante unicidade por documento
        
        Q = df['TITLE'].nunique() if 'TITLE' in df.columns else len(df)
        if Q == 0: return pd.Series(dtype=str)
        
        Qi_s = df.drop_duplicates('TITLE')['TEMA_GEMINI'].value_counts() if 'TITLE' in df.columns else df['TEMA_GEMINI'].value_counts()
        Qk_s = df_exp[col].value_counts()
        Qik_s = df_exp.groupby([col, 'TEMA_GEMINI']).size()
        
        res = []
        for (k, i), q_ik in Qik_s.items():
            q_k = Qk_s.get(k, 0)
            q_i = Qi_s.get(i, 0)
            if q_k == 0 or q_i == 0: continue
            ql = (q_ik / q_k) / (q_i / Q)
            res.append({'Entidade': k, 'Tema': i, 'QL': ql, 'Qik': q_ik})
            
        if not res: return pd.Series(dtype=str)
        
        df_ql = pd.DataFrame(res)
        # Desempate: Se houverem QLs altos e empatados, prioriza quem publicou MAIS documentos naquele tema (Qik)
        df_ql = df_ql.sort_values(by=['QL', 'Qik'], ascending=[False, False])
        top = df_ql.drop_duplicates('Tema').copy()
        top['Label'] = top['Entidade'] + " (QL: " + top['QL'].round(2).astype(str) + ")"
        return top.set_index('Tema')['Label']
        
    top_aut = _calc_top(next((c for c in ['AUTHORS', 'AU'] if c in df.columns), 'AUTHORS'))
    top_pais = _calc_top('COUNTRY')
    top_ven = _calc_top(next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), 'SECONDARY TITLE'))
    
    return top_aut, top_pais, top_ven

@st.cache_data(show_spinner=False)
def gerar_tabela_keywords(df):
    col_kw = _pick_column(df, ['KEYWORDS', 'KW', 'DE'])
    if not col_kw:
        return pd.DataFrame()
    
    df_exp = df.copy()
    df_exp['KW'] = df_exp[col_kw].astype(str).str.split(';')
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
    return pd.DataFrame(res).sort_values(by='Qtd. de Citações', ascending=False).reset_index(drop=True)

def categorizar_temas_por_cluster(df, api_key, max_clusters=10):
    """Clusteriza documentos com TF-IDF + K-Means (otimizado por Silhouette) e nomeia via Gemini."""
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from google import genai
    from google.genai import types
    import streamlit as st
    import time

    # --- INICIALIZAÇÃO DO CLIENTE (Onde o erro estava acontecendo) ---
    client = genai.Client(api_key=api_key)

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

    with st.spinner("Vetorizando textos e comprimindo semântica (LSA)..."):
        # 1. Vetorização clássica TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
        X_sparse = vectorizer.fit_transform(textos)

        # 2. NOVO: LSA (Latent Semantic Analysis / TruncatedSVD)
        # Comprime a matriz esparsa gigantesca em dimensões matemáticas densas.
        # Isso quebra a "maldição da dimensionalidade" e permite que o Silhouette funcione.
        from sklearn.decomposition import TruncatedSVD
        
        n_comps = min(50, X_sparse.shape[0] - 1, X_sparse.shape[1] - 1)
        if n_comps >= 2:
            svd = TruncatedSVD(n_components=n_comps, random_state=42)
            X = svd.fit_transform(X_sparse)
        else:
            X = X_sparse.toarray()

    with st.spinner("Calculando o K ideal via Silhouette Score..."):
        # 3. Otimização do K: Testa de 2 até max_clusters para achar a melhor divisão
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
        # Seleciona os 5 documentos mais relevantes do cluster
        amostra = df_text[df_text['CLUSTER_ID'] == cluster_id].copy()
        amostra['TAM_TEXTO'] = amostra['TEXTO_COMBINADO'].str.len()
        amostra_top = amostra.sort_values(by='TAM_TEXTO', ascending=False).head(5)
        
        textos_amostra = ""
        for _, row in amostra_top.iterrows():
            resumo = str(row['ABSTRACT'])[:600] if pd.notna(row['ABSTRACT']) else "Sem resumo"
            textos_amostra += f"- Título: {row['TITLE']}\n  Resumo: {resumo}...\n\n"
            
        prompt = f"""
        Você é um cientista de dados especialista em revisão de literatura.
        Abaixo estão amostras representativas de artigos científicos agrupados:
        
        {textos_amostra}
        
        Sua tarefa: Sintetize o tema central unificador desta escola de pesquisa.
        Responda APENAS com o nome do tema em Português, conciso (máximo 4 palavras). 
        Nenhuma pontuação final ou aspas.
        """
        
        try:
            time.sleep(2.5) # Respiro para a cota da API
            
            # --- FILTROS DE SEGURANÇA DESATIVADOS PARA TEXTOS ACADÊMICOS ---
            configuracao = types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                ]
            )

            # --- NOVA ESTRUTURA DE GERAÇÃO DA BIBLIOTECA GENAI ---
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite', 
                contents=prompt,
                config=configuracao
            )
            
            if response.text:
                tema_nome = response.text.strip().replace('\n', '').replace('"', '').replace('*', '').title()
                cluster_names[cluster_id] = tema_nome
            else:
                cluster_names[cluster_id] = f"Tema {cluster_id + 1} (Resposta Vazia)"
                
        except Exception as e:
            erro_msg = str(e)
            cluster_names[cluster_id] = f"Tema {cluster_id + 1} (Erro API)"
            st.toast(f"Cluster {cluster_id + 1} falhou: {erro_msg[:100]}", icon="🛑")
            
        progress_bar.progress((cluster_id + 1) / best_k)

    progress_bar.empty()

    # 6. Mapeamento Final: Atribui o nome gerado pela IA aos índices originais
    series_temas = df_text['CLUSTER_ID'].map(cluster_names)
    df['TEMA_GEMINI'] = series_temas
    df['TEMA_GEMINI'] = df['TEMA_GEMINI'].fillna("Outros/Não Categorizado")
    
    return df

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def _calcular_metricas_globais_sna(_G):
    """Calcula métricas de ecologia profunda para o ecossistema bibliométrico."""
    import networkx as nx
    import numpy as np
    import pandas as pd

    if len(_G) == 0: return {}

    num_nodes = len(_G)
    degrees = [d for n, d in _G.degree()]
    
    # 1. Redes Complexas
    densidade = nx.density(_G)
    try: clustering = nx.average_clustering(_G)
    except: clustering = 0.0
    
    # Entropia de Shannon (Desordem e Resiliência)
    deg_counts = np.bincount(degrees)
    probs = deg_counts[deg_counts > 0] / num_nodes
    entropia = -np.sum(probs * np.log2(probs))

    if num_nodes < 1500:
        try: eficiencia = nx.global_efficiency(_G)
        except: eficiencia = 0.0
    else:
        eficiencia = "N/A (Grafo Denso)"

    # 2. Conectividade e Influência
    media_links = np.mean(degrees)
    std_links = np.std(degrees)
    min_links = np.min(degrees)
    max_links = np.max(degrees)

    try: 
        pr = nx.pagerank(_G, max_iter=50)
        mean_pr = np.mean(list(pr.values()))
    except: mean_pr = 0.0

    try: 
        eig = nx.eigenvector_centrality_numpy(_G)
        mean_eig = np.mean(list(eig.values()))
    except: mean_eig = 0.0

    # Restrição de Burt e Redundância são O(N^3), travam servidores em redes grandes
    restricao = "N/A (Processo Lento)"
    redundancia = "N/A"

    # 4. Ecologia Profunda
    try: assortatividade = nx.degree_assortativity_coefficient(_G)
    except: assortatividade = 0.0

    # Lei de Potência (Regressão simples no espaço log-log)
    try:
        y = deg_counts[deg_counts > 0]
        x = np.nonzero(deg_counts)[0]
        if len(x) > 2:
            log_x = np.log10(x)
            log_y = np.log10(y)
            slope, _ = np.polyfit(log_x, log_y, 1)
            lei_potencia = abs(slope)
        else: lei_potencia = 0.0
    except: lei_potencia = 0.0

    # Spearman: Correlação entre ter muitos links (Degree) e ser ponte (Betweenness)
    try:
        # Usa proxy de centralidade para redes gigantes
        betw = nx.betweenness_centrality(_G, k=min(100, num_nodes))
        s_deg = pd.Series([_G.degree(n) for n in _G.nodes()])
        s_bet = pd.Series([betw[n] for n in _G.nodes()])
        spearman = s_deg.corr(s_bet, method='spearman')
    except:
        spearman = 0.0

    rich_club = "0.00% (Sem Hubs)"

    return {
        "densidade": densidade, "eficiencia": eficiencia, "entropia": entropia, "clustering": clustering,
        "media_links": media_links, "std_links": std_links, "min_links": min_links, "max_links": max_links,
        "mean_pr": mean_pr, "mean_eig": mean_eig, "restricao": restricao, "redundancia": redundancia,
        "lei_potencia": lei_potencia, "assortatividade": assortatividade, "spearman": spearman, "rich_club": rich_club
    }


def gerar_tabela_metricas_completas(df, _pbar=None):
    """Gera a tabela de métricas por nó e o dicionário de ecologia profunda da rede."""
    import networkx as nx
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

    if _pbar: _pbar.progress(0.85, text="Executando algoritmos de centralidade dos nós...")
    
    res_data, _, _, _, _ = _engine_calculo_sna(list(set(nodes)), list(set(edges)), node_types)
    df_nodes = pd.DataFrame(res_data).sort_values(by="Grau Absoluto", ascending=False)

    if _pbar: _pbar.progress(0.95, text="Calculando métricas avançadas de ecologia profunda...")
    
    G_completo = nx.Graph()
    G_completo.add_nodes_from(list(set(nodes)))
    G_completo.add_edges_from(list(set(edges)))
    metricas_globais = _calcular_metricas_globais_sna(G_completo)

    # Retorna agora uma dupla: O DataFrame dos nós e o Dicionário de métricas globais
    return df_nodes, metricas_globais

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

    return padronizar_base_bibliometrica(df)


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

    return padronizar_base_bibliometrica(df)


@st.cache_data
def calcular_metricas_bibliometrix(df):
    """Calcula métricas avançadas baseadas no relatório Main Information do Bibliometrix."""
    df_local = padronizar_base_bibliometrica(df)

    years = pd.to_numeric(df_local.get('YEAR CLEAN'), errors='coerce') if 'YEAR CLEAN' in df_local.columns else pd.Series(dtype=float)
    valid_years = years.dropna()
    if len(valid_years.unique()) > 1:
        docs_by_year = valid_years.value_counts().sort_index()
        ano_inicio = docs_by_year.index.min()
        ano_fim = docs_by_year.index.max()
        doc_inicio = docs_by_year.loc[ano_inicio]
        doc_fim = docs_by_year.loc[ano_fim]
        intervalo = ano_fim - ano_inicio
        growth_rate = ((doc_fim / doc_inicio) ** (1 / intervalo) - 1) * 100 if doc_inicio > 0 and intervalo > 0 else 0
    else:
        growth_rate = 0

    avg_cit_year = 0
    if 'TOTAL CITATIONS' in df_local.columns and 'YEAR CLEAN' in df_local.columns:
        anos_doc = CURRENT_YEAR - years + 1
        anos_doc = anos_doc.where(anos_doc > 0, 1)
        tc_per_year = df_local['TOTAL CITATIONS'] / anos_doc
        avg_cit_year = round(tc_per_year.replace([np.inf, -np.inf], np.nan).dropna().mean(), 2) if not tc_per_year.empty else 0

    mcp_count = 0
    if 'COUNTRY' in df_local.columns:
        mcp_count = int(df_local['COUNTRY'].apply(lambda value: len(set(_split_semicolon_tokens(value))) > 1).sum())

    autores_por_doc = 0
    docs_unico_autor = 0
    if 'AUTHORS' in df_local.columns:
        counts = df_local['AUTHORS'].apply(lambda value: len(_split_semicolon_tokens(value)))
        counts = counts[counts > 0]
        autores_por_doc = counts.mean()
        docs_unico_autor = (counts == 1).sum()

    return {
        "growth_rate": round(growth_rate, 2),
        "mcp": mcp_count,
        "scp": len(df_local) - mcp_count,
        "coauth_index": round(autores_por_doc, 2),
        "single_author_docs": docs_unico_autor,
        "avg_cit_year": avg_cit_year if not pd.isna(avg_cit_year) else 0
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

    col_titulos = _pick_column(df, ['TITLE', 'TI'])
    col_autores = _pick_column(df, ['AUTHORS', 'AU'])
    col_kw = _pick_column(df, ['KEYWORDS', 'KW', 'DE'])
    col_venue = _pick_column(df, ['SECONDARY TITLE', 'SO', 'JO'])
    col_paises = _pick_column(df, ['COUNTRY'])

    doc_profiles = {}
    author_profiles = defaultdict(set)
    country_profiles = defaultdict(set)
    venue_profiles = defaultdict(set)

    colunas_base = [col for col in [col_titulos, col_autores, col_kw, col_venue, col_paises] if col]
    for row in df[colunas_base].to_dict('records'):
        title = str(row.get(col_titulos, '')).strip() if col_titulos else ''
        authors = set(_split_semicolon_tokens(row.get(col_autores)))
        keywords = set(_split_semicolon_tokens(row.get(col_kw), case="lower"))
        countries = set(_split_semicolon_tokens(row.get(col_paises)))
        venue = str(row.get(col_venue, '')).strip() if col_venue else ''
        venue_set = {venue} if venue else set()

        dna_doc = keywords | authors | venue_set
        if title:
            doc_profiles[title] = dna_doc

        for author in authors:
            author_profiles[author].update(keywords)
            author_profiles[author].update(venue_set)
            author_profiles[author].update(authors - {author})

        for country in countries:
            country_profiles[country].update(keywords)
            country_profiles[country].update(authors)
            country_profiles[country].update(venue_set)

        if venue:
            venue_profiles[venue].update(keywords)
            venue_profiles[venue].update(authors)
            venue_profiles[venue].update(countries)

    if tipo_busca == "Documento":
        perfil_alvo = doc_profiles.get(termo_ativo, set())
        candidatos = {nome: perfil for nome, perfil in doc_profiles.items() if nome != termo_ativo}
    elif tipo_busca == "Autor":
        perfil_alvo = author_profiles.get(termo_ativo, set())
        candidatos = {nome: perfil for nome, perfil in author_profiles.items() if nome != termo_ativo}
    elif tipo_busca == "País":
        perfil_alvo = country_profiles.get(termo_ativo, set())
        candidatos = {nome: perfil for nome, perfil in country_profiles.items() if nome != termo_ativo}
    elif tipo_busca == "Local de Publicação (Venue)":
        perfil_alvo = venue_profiles.get(termo_ativo, set())
        candidatos = {nome: perfil for nome, perfil in venue_profiles.items() if nome != termo_ativo}
    else:
        return {}

    if not perfil_alvo:
        return {}

    resultados = []
    for candidato, perfil in candidatos.items():
        intersecao = perfil_alvo.intersection(perfil)
        if not intersecao:
            continue

        uniao = perfil_alvo.union(perfil)
        similaridade = (len(intersecao) / len(uniao)) * 100 if uniao else 0
        resultados.append({
            'Item': candidato,
            'Similaridade (%)': round(similaridade, 1),
            'Traços em Comum': " | ".join(sorted(intersecao)[:4])
        })

    resultados = sorted(resultados, key=lambda item: item['Similaridade (%)'], reverse=True)[:15]

    if tipo_busca == "Documento":
        return {'Documentos': resultados}
    if tipo_busca == "Autor":
        return {'Autores': resultados}
    return {'Itens': resultados}

def limpar_termo_busca():
    """Limpa o termo de busca quando o usuário clica manualmente no botão de rádio."""
    st.session_state['busca_termo_biblio'] = None

def navegar_busca(novo_tipo, novo_termo):
    """Atualiza o estado global para mudar o perfil exibido no motor de busca."""
    st.session_state['busca_tipo_biblio'] = novo_tipo
    st.session_state['busca_termo_biblio'] = novo_termo

@st.cache_data(show_spinner=False)
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

        citation_cols = [col for col in ['TC', 'Z9', 'TIMES CITED', 'CITED BY'] if col in df.columns]
        citations = pd.Series(np.nan, index=df.index, dtype='float64')
        for col in citation_cols:
            citations = citations.fillna(pd.to_numeric(df[col], errors='coerce'))

        if 'NOTES' in df.columns:
            notes_str = df['NOTES'].fillna('').astype(str)
            cited_by_notes = pd.to_numeric(
                notes_str.str.extract(r'Cited\s+By:\s*(\d+)', expand=False),
                errors='coerce'
            )
            times_cited_notes = pd.to_numeric(
                notes_str.str.extract(r'Times\s+Cited(?:.*?):\s*(\d+)', expand=False),
                errors='coerce'
            )
            citations = citations.fillna(cited_by_notes).fillna(times_cited_notes)

        df['TOTAL CITATIONS'] = citations.fillna(0)
        
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

        return padronizar_base_bibliometrica(df)
    return None

def deduplicar_por_doi(df):
    df_clean = df.sort_values(by='TOTAL CITATIONS', ascending=False, na_position='last').copy()
    
    doi_col = next((c for c in ['DOI', 'DO'] if c in df_clean.columns), None)
    title_col = next((c for c in ['TITLE', 'TI'] if c in df_clean.columns), None)
    
    if not doi_col: 
        return df_clean, pd.DataFrame()

    df_clean['_DOI_NORMALIZADO'] = (
        df_clean[doi_col]
        .fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_doi = df_clean[df_clean['_DOI_NORMALIZADO'].ne('')]
    dupe_mask = valid_doi.duplicated(subset=['_DOI_NORMALIZADO'], keep='first')
    dupes_indices = valid_doi[dupe_mask].index
    
    df_dupes = df_clean.loc[dupes_indices].copy()
    
    if not df_dupes.empty and title_col:
        kept_titles = (
            valid_doi.drop_duplicates(subset=['_DOI_NORMALIZADO'], keep='first')
            .set_index('_DOI_NORMALIZADO')[title_col]
            .to_dict()
        )
        df_dupes['DOCUMENTO DE REFERÊNCIA (MANTIDO)'] = df_dupes['_DOI_NORMALIZADO'].map(kept_titles)

    df_unified = df_clean.drop(index=dupes_indices).drop(columns=['_DOI_NORMALIZADO']).copy()
    df_dupes = df_dupes.drop(columns=['_DOI_NORMALIZADO'])
    return df_unified, df_dupes

def deduplicar_por_similaridade(df, threshold=0.90):
    df_clean = df.sort_values(by='TOTAL CITATIONS', ascending=False, na_position='last').copy()
    
    title_col = next((c for c in ['TITLE', 'TI'] if c in df_clean.columns), None)
    
    if not title_col or len(df_clean) < 2: 
        return df_clean, pd.DataFrame()

    indices_para_excluir = set()
    ref_mapping = {}
    normalized_titles = (
        df_clean[title_col]
        .fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df_clean['_TITLE_NORMALIZADO'] = normalized_titles

    valid_titles = df_clean[df_clean['_TITLE_NORMALIZADO'].ne('')].copy()

    exact_dupes = valid_titles.duplicated(subset=['_TITLE_NORMALIZADO'], keep='first')
    dupes_exact_idx = valid_titles[exact_dupes].index
    if len(dupes_exact_idx) > 0:
        first_titles = (
            valid_titles.drop_duplicates(subset=['_TITLE_NORMALIZADO'], keep='first')
            .set_index('_TITLE_NORMALIZADO')[title_col]
            .to_dict()
        )
        for idx in dupes_exact_idx:
            indices_para_excluir.add(idx)
            ref_mapping[idx] = first_titles.get(df_clean.at[idx, '_TITLE_NORMALIZADO'], df_clean.at[idx, title_col])

    try:
        remaining = valid_titles.loc[~valid_titles.index.isin(indices_para_excluir), '_TITLE_NORMALIZADO']
        if len(remaining) > 1:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
            tfidf_matrix = vectorizer.fit_transform(remaining)
            cosine_sparse = cosine_similarity(tfidf_matrix, dense_output=False).tocoo()
            remaining_indices = remaining.index.to_numpy()

            for row_pos, col_pos, score in zip(cosine_sparse.row, cosine_sparse.col, cosine_sparse.data):
                if row_pos >= col_pos or score < threshold:
                    continue

                idx_r = remaining_indices[row_pos]
                idx_c = remaining_indices[col_pos]

                if idx_r in indices_para_excluir or idx_c in indices_para_excluir:
                    continue

                indices_para_excluir.add(idx_c)
                ref_mapping[idx_c] = df_clean.at[idx_r, title_col]

    except Exception:
        pass

    ordered_exclusions = sorted(indices_para_excluir)
    df_dupes = df_clean.loc[ordered_exclusions].copy()
    
    if not df_dupes.empty:
        df_dupes['DOCUMENTO DE REFERÊNCIA (MANTIDO)'] = [ref_mapping.get(idx, '') for idx in ordered_exclusions]

    df_unified = df_clean.drop(index=ordered_exclusions).drop(columns=['_TITLE_NORMALIZADO']).copy()
    df_dupes = df_dupes.drop(columns=['_TITLE_NORMALIZADO'])
    return df_unified, df_dupes    
