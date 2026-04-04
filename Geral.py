import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_agraph import agraph, Config
import plotly.graph_objects as go
from utils import gerar_mapa_tematico, gerar_nuvem_echarts, processar_excel_wos, processar_csv_scopus, calcular_metricas_bibliometrix, gerar_tabela_metricas_completas, calcular_similares_biblio, limpar_termo_busca, navegar_busca, process_multiple_ris, criar_grafo_e_metricas, deduplicar_por_doi, deduplicar_por_similaridade
from pyecharts import options as opts
from pyecharts.charts import WordCloud as PyechartsWordCloud
from streamlit_echarts import st_pyecharts
from streamlit_echarts import st_echarts
from pyecharts.commons.utils import JsCode
import json
import plotly.express as px
import io
import os
from collections import Counter
import networkx as nx


st.set_page_config(page_title="Simetrics", page_icon="🧬", layout="wide")


st.title("🧬 Simetrics - Análise Bibliométrica e Cientométrica")

def create_kpi_card(title, value, color="#f0f2f6", text_color="#31333F", subtitle=""):
    st.markdown(f"""
    <div style="background-color: {color}; padding: 15px; border-radius: 8px; color: {text_color}; border: 1px solid #e0e0e0; text-align: left; height: 110px; margin-bottom: 15px;">
        <p style="margin: 0; font-size: 13px; font-weight: 500; opacity: 0.8;">{title}</p>
        <h2 style="margin: 5px 0 2px 0; font-size: 26px; font-weight: bold;">{value}</h2>
        <p style="margin: 0; font-size: 11px; opacity: 0.7;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# Inicialização segura do session_state
if 'df_geral' not in st.session_state:
    st.session_state['df_geral'] = None

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None

# Esta lógica garante que se for None ou se não existir, vira um DataFrame
if 'df_duplicados' not in st.session_state or st.session_state['df_duplicados'] is None:
    st.session_state['df_duplicados'] = pd.DataFrame()

with st.sidebar:
    st.image("simetrics - logo.png", use_container_width=True)

    st.header("1. Envio de Arquivos")
    
    # --- NOVO: GUIA DE FORMATOS SUPORTADOS ---
    with st.expander("ℹ️ Formatos e Bases Suportadas", expanded=False):
        st.markdown("""
        | Extensão | Base de Dados Sugerida |
        | :--- | :--- |
        | **.ris** | SciELO, WoS, Scopus, Mendeley |
        | **.csv** | Scopus (Exportação Direta) |
        | **.xls / .xlsx**| Web of Science (Full Record) |
        
        **Dica:** Para Excel da WoS, certifique-se de exportar com todas as colunas (Full Record e Cited References).
        """)
    
    # Adicionamos extensões de Excel
    uploaded_files = st.file_uploader(
        "Selecione arquivos RIS, CSV ou Excel", 
        type=['ris', 'csv', 'xls', 'xlsx'], 
        accept_multiple_files=True
    )
    
    db_mapping = {}
    if uploaded_files:
        st.header("2. Atribuição de Base")
        for f in uploaded_files:
            ext = f.name.lower()
            # Sugestão inteligente baseada na extensão
            if ext.endswith('.csv'): def_idx = 0 # Scopus
            elif ext.endswith(('.xls', '.xlsx')): def_idx = 1 # Web of Science
            else: def_idx = 3 # Outra
            
            db_mapping[f.name] = st.selectbox(
                f"{f.name}", 
                options=["Scopus", "Web of Science", "SciELO", "Outra"], 
                index=def_idx,
                key=f"db_{f.name}"
            )
            
        if st.button("Processar e Integrar", type="primary"):
            
            # 1. Inicializa a barra de progresso
            pbar_load = st.progress(0, text="Iniciando integração de dados...")
            
            list_dfs = []
            total_files = len(uploaded_files)
            
            for i, f in enumerate(uploaded_files):
                # 2. Atualiza a barra para cada arquivo processado
                progresso_atual = (i + 1) / total_files
                pbar_load.progress(progresso_atual, text=f"Integrando {f.name} ({i+1}/{total_files})")
                
                ext = f.name.lower()
                if ext.endswith('.csv'):
                    df_temp = processar_csv_scopus(f)
                elif ext.endswith(('.xls', '.xlsx')):
                    df_temp = processar_excel_wos(f)
                else:
                    # Processamento RIS existente
                    df_temp = process_multiple_ris([f], {f.name: db_mapping[f.name]})
                
                if df_temp is not None:
                    df_temp['BASE DE DADOS'] = db_mapping[f.name]
                    list_dfs.append(df_temp)

            # 3. Remove a barra de progresso ao finalizar o loop
            pbar_load.empty()

            if list_dfs:
                with st.spinner("Consolidando estrutura final..."):
                    df_raw = pd.concat(list_dfs, ignore_index=True)

                    # --- NOVO: UNIFICAÇÃO DE SEGURANÇA ---
                    # Caso existam colunas antigas (REFERENCES ou CITED REFERENCES), unifica em uma só
                    cols_para_unificar = ['REFERENCES', 'CITED REFERENCES', 'CR']
                    if 'REFERENCES_UNIFIED' not in df_raw.columns:
                        df_raw['REFERENCES_UNIFIED'] = ""
                        
                    for col in cols_para_unificar:
                        if col in df_raw.columns:
                            # Preenche a unificada com os dados da coluna encontrada, caso a unificada esteja vazia
                            df_raw['REFERENCES_UNIFIED'] = df_raw['REFERENCES_UNIFIED'].fillna(df_raw[col])
                            df_raw['REFERENCES_UNIFIED'] = df_raw['REFERENCES_UNIFIED'].replace("", df_raw[col])
                    
                    # Remove as colunas duplicadas para limpar a memória e a tabela
                    df_raw = df_raw.drop(columns=[c for c in cols_para_unificar if c in df_raw.columns], errors='ignore')
                    # -------------------------------------
                    
                    # Garante que o ano seja numérico para evitar erros nos gráficos
                    if 'YEAR CLEAN' not in df_raw.columns and 'YEAR' in df_raw.columns:
                        df_raw['YEAR CLEAN'] = pd.to_numeric(df_raw['YEAR'], errors='coerce')
                    
                    st.session_state['df_original'] = df_raw.copy()
                    st.session_state['df_geral'] = df_raw
                    st.session_state['df_duplicados'] = pd.DataFrame()
                    st.success(f"Sucesso! {len(df_raw)} documentos integrados.")
                    st.rerun()
            else:
                st.error("Não foi possível extrair dados dos arquivos selecionados.")
            
            # --- NOVO: MODO DE DEMONSTRAÇÃO ---
    st.markdown("---")
    st.subheader("🌟 Modo de Demonstração")
    st.caption("Não tem arquivos agora? Explore o Simetrics com dados de exemplo.")
    
    if st.button("🚀 Carregar Ecossistema de Exemplo", help="Carrega automaticamente bases pré-configuradas (Scopus, WoS e SciELO) da pasta do projeto."):
       
        # Lista dos arquivos que devem estar na pasta raiz
        arquivos_demo = ["scopus.ris", "wos.ris", "scielo.ris"]
        mapping_demo = {
            "scopus.ris": "Scopus", 
            "wos.ris": "Web of Science", 
            "scielo.ris": "SciELO"
        }
        
        list_mock_files = []
        
        for nome in arquivos_demo:
            if os.path.exists(nome):
                with open(nome, "rb") as f:
                    content = f.read()
                    # Criamos um "mock" de UploadedFile para o processador de RIS
                    mock_file = io.BytesIO(content)
                    mock_file.name = nome
                    list_mock_files.append(mock_file)
            else:
                st.sidebar.warning(f"Atenção: O arquivo '{nome}' não foi encontrado na raiz.")

        if list_mock_files:
            with st.spinner("Tecendo rede de demonstração..."):
                # Reutilizamos a sua função existente do utils.py
                
                df_demo = process_multiple_ris(list_mock_files, mapping_demo)
                
                if df_demo is not None:
                    # Atualiza o estado da aplicação com os dados de exemplo
                    st.session_state['df_original'] = df_demo.copy()
                    st.session_state['df_geral'] = df_demo
                    st.session_state['df_duplicados'] = pd.DataFrame()
                    st.success("Exemplo carregado com sucesso!")
                    st.rerun()

if st.session_state['df_geral'] is not None:
    df = st.session_state['df_geral']
    
    tab_main, tab_grafos, tab_search = st.tabs(["📊 Informações Principais", "🕸️ Redes e Grafos de Conhecimento","🔍 Motor de Busca"])
    
    with tab_main:
        st.subheader("Resumo Estrutural da Amostra")

        # Interface de Deduplicação
        with st.expander("🛠️ Ferramenta de Limpeza: Remover Documentos Duplicados", expanded=True):
            st.write("Escolha o método de deduplicação. A base será atualizada nos gráficos e tabelas.")
            
            # Controle de Threshold para a Similaridade
            threshold = st.slider("Limiar de Similaridade do Título (Apenas para o Botão 2):", min_value=0.70, max_value=1.00, value=0.90, step=0.01)
            
            c_btn1, c_btn2, c_btn3 = st.columns(3)
            
            with c_btn1:
                if st.button("1. Deduplicar por DOI Exato"):
                    with st.spinner("Buscando DOIs..."):
                        df_limpo, df_dupes = deduplicar_por_doi(st.session_state['df_geral'])
                        st.session_state['df_geral'] = df_limpo
                        # ANEXA os novos duplicados aos já existentes
                        st.session_state['df_duplicados'] = pd.concat([st.session_state['df_duplicados'], df_dupes], ignore_index=True)
                        st.rerun()

            with c_btn2:
                if st.button("2. Deduplicar por Similaridade"):
                    with st.spinner("Calculando similaridade..."):
                        df_limpo, df_dupes = deduplicar_por_similaridade(st.session_state['df_geral'], threshold)
                        st.session_state['df_geral'] = df_limpo
                        # ANEXA os novos duplicados aos já existentes
                        st.session_state['df_duplicados'] = pd.concat([st.session_state['df_duplicados'], df_dupes], ignore_index=True)
                        st.rerun()

            with c_btn3:
                if st.button("🔄 Reverter Base"):
                    st.session_state['df_geral'] = st.session_state['df_original'].copy()
                    st.session_state['df_duplicados'] = pd.DataFrame() # Limpa a lista de excluídos
                    st.rerun()

            # Avisos de Sucesso ou Erro
            if st.session_state.get('df_duplicados') is not None:
                qtd_removidos = len(st.session_state['df_duplicados'])
                if qtd_removidos > 0:
                    st.error(f"🗑️ Foram detectados e removidos **{qtd_removidos} documentos duplicados** nesta ação.")
                else:
                    st.success("🗃️ Deduplique por DOI ou por similaridade")

            if st.session_state['df_duplicados'] is not None and not st.session_state['df_duplicados'].empty:
                st.markdown("### 🚨 Relatório Permanente de Documentos Excluídos")
            
                dupes = st.session_state['df_duplicados']
                # Definindo colunas para exibição
                cols_exibicao = [c for c in ['TITLE', 'TI', 'DOCUMENTO DE REFERÊNCIA (MANTIDO)', 'BASE DE DADOS', 'DOI'] if c in dupes.columns]
            
                st.dataframe(dupes[cols_exibicao], use_container_width=True, hide_index=True)
            
                # Botão para baixar apenas os excluídos (útil para o anexo da metodologia PRISMA)
                csv_dupes = dupes.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar Relatório de Excluídos (CSV)", data=csv_dupes, file_name='documentos_excluidos.csv')

            st.write("")

        # =========================================================
        # --- NOVO: PANORAMA DE COMPLETUDE DOS METADADOS ---
        # =========================================================
        st.markdown("##### 🗂️ Qualidade e Completude dos Metadados")
        st.caption(f"Análise de dados faltantes em **{len(df)} documentos**. Metadados incompletos podem reduzir a precisão das redes de conhecimento.")
        
        # Mapeamento dos campos principais para análise
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
        
        dados_completude = []
        tot_docs = len(df)
        
        for col_chave, descricao in campos_verificacao:
            if col_chave in df.columns:
                # Conta valores nulos (NaN) e strings vazias/só com espaços
                faltantes = df[col_chave].isna().sum() + (df[col_chave].astype(str).str.strip() == '').sum()
            else:
                # Se a coluna nem existir na base, 100% dos dados faltam
                faltantes = tot_docs
                
            pct_faltante = (faltantes / tot_docs) * 100 if tot_docs > 0 else 0
            
            # Regras de Status (Inspiradas no Bibliometrix)
            if pct_faltante == 0:
                status = "Excelente"
            elif pct_faltante <= 10:
                status = "Bom"
            elif pct_faltante <= 20:
                status = "Aceitável"
            else:
                status = "Ruim"
                
            dados_completude.append({
                "Metadado": descricao,
                "Faltantes": int(faltantes),
                "Faltantes (%)": pct_faltante,
                "Status": status
            })
            
        df_comp = pd.DataFrame(dados_completude).sort_values(by="Faltantes (%)")
        
        # Função para aplicar cores estilo "Semáforo" no Pandas
        def colorir_status(val):
            if val == 'Excelente': return 'background-color: #28a745; color: white; font-weight: bold;'
            elif val == 'Bom': return 'background-color: #82e0aa; color: black; font-weight: bold;'
            elif val == 'Aceitável': return 'background-color: #f1c40f; color: black; font-weight: bold;'
            else: return 'background-color: #e74c3c; color: white; font-weight: bold;'
            
        # Aplica o estilo na coluna de Status e formata a porcentagem
        tabela_estilizada = df_comp.style.map(colorir_status, subset=['Status']).format({'Faltantes (%)': "{:.2f}%"})
        
        # Exibe a tabela no Streamlit
        st.dataframe(tabela_estilizada, use_container_width=True, hide_index=True)
        st.divider()

        # =========================================================
        # --- INTELIGÊNCIA ARTIFICIAL: CATEGORIZAÇÃO TEMÁTICA ---
        # =========================================================
        st.markdown("##### 🤖 Inteligência Artificial: Categorização de Temas")
        
        with st.expander("Identificar Escolas de Pesquisa via Machine Learning", expanded=True):
            api_key_valida = False
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                api_key_valida = True
            except KeyError:
                st.error("Chave 'GEMINI_API_KEY' não encontrada.")
            
            # --- VERIFICAÇÃO DE DADOS EXISTENTES ---
            # Importante: Buscamos a coluna diretamente no session_state para garantir a exibição
            if 'TEMA_GEMINI' in st.session_state['df_original'].columns:
                st.success("✅ O corpus foi categorizado com sucesso!")
                
                # Exibe a tabela de resumo sempre que a coluna existir
                df_resumo = st.session_state['df_original']['TEMA_GEMINI'].value_counts().reset_index()
                df_resumo.columns = ['Escola Temática (IA)', 'Documentos']
                
                st.markdown("###### 📊 Resumo da Distribuição")
                st.dataframe(
                    df_resumo.style.bar(subset=['Documentos'], color='#82c2c2'),
                    use_container_width=True, 
                    hide_index=True
                )
                
                if st.button("Refazer Categorização (Limpar Temas)"):
                    st.session_state['df_original'] = st.session_state['df_original'].drop(columns=['TEMA_GEMINI'])
                    st.rerun()
            
            elif api_key_valida:
                st.info("⚡ O algoritmo Silhouette identificará os grupos e o Gemini nomeará cada escola.")
                if st.button("Executar Mapeamento Temático", type="primary"):
                    from utils import categorizar_temas_por_cluster
                    
                    # CORREÇÃO CRÍTICA: Processamos e salvamos diretamente no df_original
                    # Isso garante que a alteração seja global em todas as abas
                    df_processado = categorizar_temas_por_cluster(st.session_state['df_original'], api_key)
                    st.session_state['df_original'] = df_processado
                    
                    st.toast("Temas gerados com sucesso!", icon="🤖")
                    st.rerun()

        # --- 1. CÁLCULO DE TODAS AS MÉTRICAS ---        
        total_docs = len(df)
        b_metrics = calcular_metricas_bibliometrix(df)
        
        # Período e Idade Média
        timespan = f"{int(df['YEAR CLEAN'].min())}:{int(df['YEAR CLEAN'].max())}" if 'YEAR CLEAN' in df.columns and pd.notna(df['YEAR CLEAN'].min()) else "N/S"
        avg_age = round(2026 - df['YEAR CLEAN'].mean(), 2) if 'YEAR CLEAN' in df.columns else "N/S"
        
        # Autores Únicos
        authors_count = 0
        if 'AUTHORS' in df.columns:
            flat_auths = [a.strip() for sublist in df['AUTHORS'].dropna().astype(str).str.split(';') for a in sublist if a.strip()]
            authors_count = len(set(flat_auths))

        # Países Únicos
        countries_count = 0
        if 'COUNTRY' in df.columns:
            flat_countries = [c.strip() for sublist in df['COUNTRY'].dropna().astype(str).str.split(';') for c in sublist if c.strip()]
            countries_count = len(set(flat_countries))

        # Palavras-Chave Únicas
        kw_count = 0
        col_kw = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None)
        if col_kw:
            flat_kw = [k.strip().lower() for sublist in df[col_kw].dropna().astype(str).str.split(';') for k in sublist if k.strip()]
            kw_count = len(set(flat_kw))

        # Locais de Publicação (Venues)
        venues_count = 0
        col_v = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
        if col_v:
            venues_count = df[col_v].dropna().nunique()

        # --- 2. EXIBIÇÃO DOS CARDS EM TRÊS LINHAS SIMÉTRICAS ---
        
        # Linha 1: Massa Crítica
        k1, k2, k3, k4 = st.columns(4)
        with k1: create_kpi_card("Período", timespan)
        with k2: create_kpi_card("Total de Documentos", total_docs)
        with k3: create_kpi_card("Autores Únicos", authors_count, "#2E86C1", "white")
        with k4: create_kpi_card("Países", countries_count, "#2E86C1", "white")

        # Linha 2: Dinâmica e Impacto (As Novas Métricas)
        k5, k6, k7, k8 = st.columns(4)
        with k5: create_kpi_card("Taxa de Crescimento Anual", f"{b_metrics['growth_rate']}%", "#E67E22", "white")
        with k6: create_kpi_card("Citações Médias/Ano/Doc", b_metrics['avg_cit_year'], "#E67E22", "white")
        with k7: create_kpi_card("Locais de Publicação (Venues)", venues_count, "#138D75", "white")
        with k8: create_kpi_card("Palavras-Chave", kw_count, "#138D75", "white")

        # Linha 3: Destaque de Maturidade
        st.write("")
        m_col = st.columns([1, 2, 1]) # Centralizando o card de idade
        with m_col[1]:
            create_kpi_card("Média de Idade dos Documentos", f"{avg_age} anos", "#2E86C1", "white")

        st.divider()
        col_meta, col_coll = st.columns(2)

        with col_meta:
            st.markdown("##### 📄 Conteúdo e Tipologia")
            
            # Tabela de Tipos de Documentos (IDêntica ao Bibliometrix)
            col_tipo = next((c for c in df.columns if str(c).upper() in ['TYPE OF REFERENCE', 'DOCUMENT TYPE', 'DT']), None)
            if col_tipo:
                dt_counts = df[col_tipo].value_counts().reset_index()
                dt_counts.columns = ['Tipo de Documento', 'Quantidade']
                st.dataframe(dt_counts, use_container_width=True, hide_index=True)
            

        with col_coll:
            st.markdown("##### 🤝 Autoria e Colaboração")
            
            # Dados de Colaboração em Tabela
            coll_data = pd.DataFrame({
                "Métrica": ["Documentos de Autor Único", "Índice de Coautoria", "Publicações Multi-País (MCP)", "Publicações Mono-País (SCP)"],
                "Valor": [b_metrics['single_author_docs'], b_metrics['coauth_index'], b_metrics['mcp'], b_metrics['scp']]
            })
            st.dataframe(coll_data, use_container_width=True, hide_index=True)
            
            # Ratio de Internacionalização
            intl_ratio = (b_metrics['mcp'] / len(df)) * 100 if len(df) > 0 else 0
            st.progress(intl_ratio / 100, text=f"Índice de Colaboração Internacional: {intl_ratio:.2f}%")

        st.divider()

        col_graf_1, col_graf_2 = st.columns(2)
        
        with col_graf_1:
            st.markdown("##### Dinâmica de Produção e Impacto")
            
            # 1. Seleção de Métrica (O novo Dropdown)
            metrica_dyn = st.selectbox(
                "Métrica do Eixo Y:",
                ["Quantidade de Produções", "Média de Citações por Ano"],
                key="sel_metrica_dinamica"
            )

            # 2. Identificação da coluna de tipo de documento
            nomes_comuns_tipo = [
                'TYPE', 'DT', 'DOCUMENT TYPE', 'TY', 'TIPO', 
                'TIPO DE DOCUMENTO', 'TYPE OF REFERENCE', 'REFERENCE TYPE'
            ]
            col_tipo_doc = next((c for c in df.columns if str(c).strip().upper() in nomes_comuns_tipo), None)
            
            # 3. Seletor de visualização (Radio)
            opcoes_prod = ["Volume Geral"]
            if col_tipo_doc:
                opcoes_prod.append("Separado por Tipo de Documento")
            
            modo_prod = st.radio("Visualização:", opcoes_prod, horizontal=True, key="prod_dyn_mode")
            
            # 4. Lógica de Processamento
            if 'YEAR CLEAN' in df.columns:
                df_prod = df.dropna(subset=['YEAR CLEAN']).copy()
                df_prod['YEAR CLEAN'] = pd.to_numeric(df_prod['YEAR CLEAN'], errors='coerce')
                df_prod = df_prod.dropna(subset=['YEAR CLEAN'])
                
                # Definimos o que será calculado
                label_y = "Documentos" if metrica_dyn == "Quantidade de Produções" else "Média de Citações"
                aggr_func = 'size' if metrica_dyn == "Quantidade de Produções" else 'mean'
                col_calc = col_tipo_doc if metrica_dyn == "Quantidade de Produções" else 'TOTAL CITATIONS'

                if modo_prod == "Volume Geral":
                    if metrica_dyn == "Quantidade de Produções":
                        df_plot = df_prod.groupby('YEAR CLEAN').size().reset_index(name=label_y)
                    else:
                        df_plot = df_prod.groupby('YEAR CLEAN')['TOTAL CITATIONS'].mean().reset_index(name=label_y)
                    
                    df_plot.columns = ['Ano', label_y]
                    fig_year = px.line(
                        df_plot.sort_values('Ano'), 
                        x='Ano', y=label_y, 
                        markers=True, 
                        color_discrete_sequence=['#1273B9'],
                        title=f"{metrica_dyn} (Total)"
                    )
                else:
                    # Separado por Tipo
                    df_prod[col_tipo_doc] = df_prod[col_tipo_doc].fillna("Não Especificado")
                    if metrica_dyn == "Quantidade de Produções":
                        df_plot = df_prod.groupby(['YEAR CLEAN', col_tipo_doc]).size().reset_index(name=label_y)
                    else:
                        df_plot = df_prod.groupby(['YEAR CLEAN', col_tipo_doc])['TOTAL CITATIONS'].mean().reset_index(name=label_y)
                    
                    df_plot.columns = ['Ano', 'Tipo', label_y]
                    fig_year = px.line(
                        df_plot.sort_values('Ano'), 
                        x='Ano', y=label_y, 
                        color='Tipo', 
                        markers=True,
                        title=f"{metrica_dyn} por Categoria"
                    )
                
                # Ajustes Estéticos
                fig_year.update_layout(
                    xaxis=dict(title="Ano", tickmode='linear', dtick=1),
                    yaxis=dict(title=label_y),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.warning("Coluna 'YEAR CLEAN' necessária para esta análise.")

        with col_graf_2:
            st.markdown("##### Distribuição por Base de Dados")
            chart_type = st.radio("Selecione o tipo de gráfico:", ["Barras", "Donut", "Pizza"], horizontal=True)
            db_counts = df['BASE DE DADOS'].value_counts().reset_index()
            db_counts.columns = ['Base', 'Quantidade']
            if chart_type == "Barras": fig_db = px.bar(db_counts, x='Base', y='Quantidade', color='Base', color_discrete_sequence=px.colors.qualitative.Pastel)
            elif chart_type == "Donut": fig_db = px.pie(db_counts, names='Base', values='Quantidade', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            else: fig_db = px.pie(db_counts, names='Base', values='Quantidade', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_db, use_container_width=True)

        st.divider()

        col_auth, col_docs = st.columns(2)
        
        with col_auth:
            st.markdown("##### 👥 Top 20 Autores")
            metric_author = st.radio(
                "Métrica para o Ranking de Autores:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações por Doc."], 
                horizontal=True, key="sel_auth_metric"
            )
            
            if 'AUTHORS' in df.columns:
                # Otimização Vetorizada
                df_auth_temp = df[['AUTHORS', 'TOTAL CITATIONS']].copy()
                df_auth_temp['TOTAL CITATIONS'] = pd.to_numeric(df_auth_temp['TOTAL CITATIONS'], errors='coerce').fillna(0)
                df_auth_temp['Autor'] = df_auth_temp['AUTHORS'].astype(str).str.split(';')
                
                df_auth_expanded = df_auth_temp.explode('Autor')
                df_auth_expanded['Autor'] = df_auth_expanded['Autor'].str.strip()
                df_auth_expanded = df_auth_expanded[df_auth_expanded['Autor'] != '']
                df_auth_expanded['Documentos'] = 1
                
                # Agrupamento usando o nome real da coluna 'TOTAL CITATIONS'
                res_auth = df_auth_expanded.groupby('Autor').agg({
                    'Documentos': 'sum', 
                    'TOTAL CITATIONS': 'sum'
                }).reset_index()
                
                # Renomeia para 'Citações' apenas APÓS o cálculo
                res_auth.rename(columns={'TOTAL CITATIONS': 'Citações'}, inplace=True)
                res_auth['Média'] = (res_auth['Citações'] / res_auth['Documentos']).round(2)
                
                if metric_author == "Qtd. de Documentos":
                    top_authors = res_auth.nlargest(20, 'Documentos')
                    fig_auth = px.bar(top_authors, x='Documentos', y='Autor', orientation='h', color='Documentos', color_continuous_scale='Blues')
                elif metric_author == "Total de Citações":
                    top_authors = res_auth.nlargest(20, 'Citações')
                    fig_auth = px.bar(top_authors, x='Citações', y='Autor', orientation='h', color='Citações', color_continuous_scale='Oranges')
                else: # Média de Citações
                    top_authors = res_auth.nlargest(20, 'Média')
                    fig_auth = px.bar(top_authors, x='Média', y='Autor', orientation='h', color='Média', color_continuous_scale='GnBu')
                
                fig_auth.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_auth, use_container_width=True)

        with col_docs:
            st.markdown("##### 📄 Top 20 Documentos Mais Citados")
            title_col = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
            
            # Adicionamos uma verificação: a coluna existe e tem pelo menos um número que não seja nulo?
            if title_col and 'TOTAL CITATIONS' in df.columns and df['TOTAL CITATIONS'].notna().any():
                try:
                    # O nlargest agora funcionará porque garantimos o tipo no utils.py
                    top_d = df.nlargest(20, 'TOTAL CITATIONS').copy()
                    top_d['Título Curto'] = top_d[title_col].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else x)
                    
                    fig_d = px.bar(top_d, x='TOTAL CITATIONS', y='Título Curto', orientation='h', 
                                    color='TOTAL CITATIONS', color_continuous_scale='Reds')
                    fig_d.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_d, use_container_width=True)
                except Exception as e:
                    st.warning("Não foi possível gerar o ranking de citações com os dados disponíveis.")
            else:
                st.info("ℹ️ Nenhuma informação de citação encontrada nos documentos para gerar este ranking.")


        # --- LINHA 3 DE GRÁFICOS: TOP PAÍSES E TOP VENUES LADO A LADO ---
        col_pais, col_venue = st.columns(2)
        
        with col_pais:
            st.markdown("##### 🌍 Top 20 Países Mais Produtivos")
            metric_country = st.radio(
                "Métrica de Países:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações"], 
                horizontal=True, key="sel_pais"
            )
            
            if 'COUNTRY' in df.columns:
                # Otimização Vetorizada
                df_country_temp = df[['COUNTRY', 'TOTAL CITATIONS']].copy()
                df_country_temp['TOTAL CITATIONS'] = pd.to_numeric(df_country_temp['TOTAL CITATIONS'], errors='coerce').fillna(0)
                df_country_temp['País'] = df_country_temp['COUNTRY'].astype(str).str.split(';')
                
                df_country_expanded = df_country_temp.explode('País')
                df_country_expanded['País'] = df_country_expanded['País'].str.strip()
                df_country_expanded = df_country_expanded[df_country_expanded['País'] != '']
                df_country_expanded['Documentos'] = 1
                
                # Agrupamento usando o nome real da coluna 'TOTAL CITATIONS'
                res_country = df_country_expanded.groupby('País').agg({
                    'Documentos': 'sum', 
                    'TOTAL CITATIONS': 'sum'
                }).reset_index()
                
                res_country.rename(columns={'TOTAL CITATIONS': 'Citações'}, inplace=True)
                res_country['Média'] = (res_country['Citações'] / res_country['Documentos']).round(2)
                
                if metric_country == "Qtd. de Documentos":
                    top_c = res_country.nlargest(20, 'Documentos')
                    fig_c = px.bar(top_c, x='Documentos', y='País', orientation='h', color='Documentos', color_continuous_scale='Viridis')
                elif metric_country == "Total de Citações":
                    top_c = res_country.nlargest(20, 'Citações')
                    fig_c = px.bar(top_c, x='Citações', y='País', orientation='h', color='Citações', color_continuous_scale='Plasma')
                else: 
                    top_c = res_country.nlargest(20, 'Média')
                    fig_c = px.bar(top_c, x='Média', y='País', orientation='h', color='Média', color_continuous_scale='YlGnBu')
                
                fig_c.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_c, use_container_width=True)

        with col_venue:
            st.markdown("##### 🏢 Top 20 Locais de Publicação (Venues)")
            
            metric_venue = st.radio(
                "Métrica de Locais:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações"], 
                horizontal=True, key="sel_venue"
            )
            
            # Procura a coluna de Venues baseada nos padrões comuns
            col_venue_name = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
            
            if col_venue_name:
                venue_data = []
                df_with_venue = df.dropna(subset=[col_venue_name])
                
                for _, row in df_with_venue.iterrows():
                    v_name = str(row[col_venue_name]).strip()
                    cit = row['TOTAL CITATIONS'] if pd.notna(row['TOTAL CITATIONS']) else 0
                    if v_name:
                        venue_data.append({'Venue': v_name, 'Documentos': 1, 'Citações': cit})
                
                if venue_data:
                    df_venue_expanded = pd.DataFrame(venue_data)
                    res_venue = df_venue_expanded.groupby('Venue').agg({
                        'Documentos': 'sum', 
                        'Citações': 'sum'
                    }).reset_index()
                    res_venue['Média'] = (res_venue['Citações'] / res_venue['Documentos']).round(2)
                    
                    # Corta nomes muito longos de revistas/conferências para o gráfico ficar bonito
                    res_venue['Venue Curta'] = res_venue['Venue'].apply(lambda x: str(x)[:40] + "..." if len(str(x)) > 40 else x)
                    
                    if metric_venue == "Qtd. de Documentos":
                        top_v = res_venue.nlargest(20, 'Documentos')
                        fig_v = px.bar(top_v, x='Documentos', y='Venue Curta', orientation='h', color='Documentos', color_continuous_scale='Teal')
                    elif metric_venue == "Total de Citações":
                        top_v = res_venue.nlargest(20, 'Citações')
                        fig_v = px.bar(top_v, x='Citações', y='Venue Curta', orientation='h', color='Citações', color_continuous_scale='Aggrnyl')
                    else: 
                        top_v = res_venue.nlargest(20, 'Média')
                        fig_v = px.bar(top_v, x='Média', y='Venue Curta', orientation='h', color='Média', color_continuous_scale='Tealgrn')
                    
                    fig_v.update_layout(yaxis={'categoryorder':'total ascending'})
                    # Adiciona tooltip completo para quando passar o mouse
                    fig_v.update_traces(customdata=top_v['Venue'], hovertemplate='<b>%{customdata}</b><br>Valor: %{x}<extra></extra>')
                    
                    st.plotly_chart(fig_v, use_container_width=True)
            else:
                st.info("ℹ️ Nenhuma coluna de Local de Publicação (ex: SECONDARY TITLE, SO, JO) foi encontrada nos dados.")

       # --- BLOCO: TOP 20 PALAVRAS-CHAVE ---
        st.markdown("##### ☁️ Top 20 Palavras-chave")
        
        col_kw_opt, col_kw_fig = st.columns([1, 3])

        with col_kw_opt:
            # ATUALIZAÇÃO: De selectbox para radio
            metric_kw = st.radio(
                "Métrica de Ranking (Keywords):",
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações"],
                key="sel_metric_kw_radio" # Alterado a key para evitar conflitos
            )
            st.info("💡 **Dica:** O gradiente lateral ajuda a identificar visualmente a distância de impacto entre o termo principal e os demais.")

        with col_kw_fig:
            with st.spinner("Mapeando léxico de impacto..."):
                from utils import plot_top_keywords_metric
                fig_kw = plot_top_keywords_metric(df, metric_kw, top_n=20)
                
                if fig_kw:
                    st.plotly_chart(fig_kw, use_container_width=True)
                else:
                    st.warning("Não foram encontradas palavras-chave válidas.")

        # --- BLOCO: COLABORAÇÃO ENTRE PAÍSES ---
        st.divider()
        st.markdown("##### 🌍 Redes de Colaboração Internacional")
        st.caption("Investigue como a produção científica se articula geopoliticamente. O grafo circular evidencia as parcerias diretas, enquanto o mapa-múndi revela as pontes intercontinentais.")

        col_circ, col_map = st.columns(2)

        with col_circ:
            with st.spinner("Desenhando matriz circular de países..."):
                from utils import plot_circular_collaboration
                fig_circ = plot_circular_collaboration(df, top_n=30)
                if fig_circ:
                    st.plotly_chart(fig_circ, use_container_width=True)
                else:
                    st.warning("Não há dados de países suficientes para formar redes de colaboração.")

        with col_map:
            with st.spinner("Renderizando mapa-múndi interativo..."):
                from utils import plot_map_collaboration
                fig_map = plot_map_collaboration(df, top_n=30)
                if fig_map:
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.info("Aguardando conexões geográficas...")

        # --- LINHA 4 DE GRÁFICOS: NUVEM DE PALAVRAS ---
        st.divider()
        st.markdown("##### ☁️ Nuvem de Palavras")
        
        col_wc_sel, col_wc_img = st.columns([1, 3])
        
        with col_wc_sel:
            st.write("")
            fonte_nuvem = st.selectbox(
                "Fonte de dados para a Nuvem:",
                ["Títulos", "Palavras-chave", "Resumo (Abstract)", "Título + Resumo + Palavras-chave"],
                key="sel_wordcloud"
            )

            # NOVO SELETOR DE ESTILO:
            estilo_fonte = st.selectbox(
                "Estilo da Fonte:",
                ["Arial", "Verdana", "Courier New", "Comic Sans MS", "Impact", "Poppins"],
                index=0,
                key="sel_font_style"
            )

            tema_cor = st.selectbox(
                "Paleta de Cores:",
                ["Oceano", "Fogo", "Floresta", "Cyberpunk", "Acadêmico"],
                index=0,
                key="sel_word_palette"
            )

            # Dicionário de Hex-Codes para as paletas
            paletas = {
                "Oceano": ["#0077b6", "#00b4d8", "#90e0ef", "#03045e", "#023e8a"],
                "Fogo": ["#ff4d00", "#ff8c00", "#ff0000", "#fad02c", "#e85d04"],
                "Floresta": ["#2d6a4f", "#40916c", "#1b4332", "#74c69d", "#95d5b2"],
                "Cyberpunk": ["#f72585", "#7209b7", "#3a0ca3", "#4361ee", "#4cc9f0"],
                "Acadêmico": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
            }
            
            paleta_escolhida = paletas[tema_cor]
            
            # --- LÓGICA DE MAPEAMENTO ATUALIZADA ---
            # Identifica as colunas reais presentes no seu DataFrame
            c_t = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
            c_k = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None)
            c_a = next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)

            if fonte_nuvem == "Título + Resumo + Palavras-chave":
                # Filtra apenas as colunas que existem na base carregada
                cols_para_unir = [c for c in [c_t, c_k, c_a] if c]
                # Cria uma coluna temporária unindo os textos com espaços
                df['WC_COMBINADO'] = df[cols_para_unir].fillna('').agg(' '.join, axis=1)
                coluna_escolhida = 'WC_COMBINADO'
            else:
                mapa_colunas = {
                    "Títulos": c_t,
                    "Palavras-chave": c_k,
                    "Resumo (Abstract)": c_a
                }
                coluna_escolhida = mapa_colunas.get(fonte_nuvem)
            
            st.info(f"As 'Stopwords' em inglês são removidas automaticamente para destacar termos conceituais.")

        with col_wc_img:
            if coluna_escolhida and coluna_escolhida in df.columns:
                with st.spinner("Pintando palavras e ajustando tipografia..."):
                    
                    wc_opcoes = gerar_nuvem_echarts(
                        df, 
                        coluna_escolhida, 
                        fonte=estilo_fonte, 
                        paleta=paleta_escolhida
                    )
                    
                    if wc_opcoes:
                        st_echarts(
                            options=wc_opcoes, 
                            height="550px", 
                            key=f"wc_final_{estilo_fonte}_{fonte_nuvem}_{tema_cor}"
                        )
                    else:
                        st.warning("Não há texto suficiente nesta coluna para gerar a nuvem.")
            else:
                st.warning(f"A coluna de {fonte_nuvem} não foi encontrada nos dados.")

        # =========================================================
        # --- NOVO BLOCO: MAPAS CONCEITUAIS 2D E 3D ---
        # =========================================================
        st.divider()
        st.markdown("##### 🧠 Estrutura Intelectual (Mapa Conceitual PCA)")
        st.caption("A Análise de Componentes Principais (PCA) reduz a base a dimensões espaciais. Termos agrupados na mesma cor pertencem à mesma escola temática (K-Means Clustering).")

        col_config_mapa, col_mapas_graf = st.columns([1, 4])
        
        with col_config_mapa:
            top_termos_mapa = st.slider("Quantidade de Termos:", min_value=20, max_value=100, value=50, step=10, key='sl_mapa_termos', help="Foca a matemática nos termos mais frequentes para evitar ruído.")
            num_clusters = st.slider("Qtd. de Escolas (Clusters):", min_value=2, max_value=8, value=3, step=1, key='sl_mapa_clusters')
            st.info("💡 **Dica 3D:** Arraste o gráfico tridimensional para rotacionar, descobrir profundidade e ver quais termos atuam como 'ponte' entre os clusters principais.")
            
        with col_mapas_graf:
            with st.spinner("Extraindo matriz semântica e calculando dimensionalidade..."):
                from utils import gerar_mapas_conceituais
                fig_2d, fig_3d = gerar_mapas_conceituais(df, top_n_words=top_termos_mapa, n_clusters=num_clusters)
                
                if fig_2d and fig_3d:
                    c2d, c3d = st.columns(2)
                    with c2d:
                        st.markdown("**Projeção Plana (2D)**")
                        st.plotly_chart(fig_2d, use_container_width=True)
                    with c3d:
                        st.markdown("**Projeção Imersiva (3D)**")
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Não há palavras-chave válidas ou diversidade suficiente na amostra para processar a topologia matemática.")

        # =========================================================
        # --- AUTORES: PRODUÇÃO NO TEMPO E LEI DE LOTKA ---
        # =========================================================
        st.divider()
        st.markdown("##### ✍️ Dinâmica de Autoria e Produtividade Científica")
        
        col_lotka, col_prod = st.columns([1, 1.2]) # Produção ganha levemente mais espaço
        
        with col_lotka:
            st.caption("A **Lei de Lotka** compara a distribuição teórica da produtividade (linha vermelha tracejada) com o que ocorre na sua base (linha azul). É esperado que muitos autores publiquem apenas um artigo, e pouquíssimos publiquem muitos.")
            with st.spinner("Calculando constante da Lei de Lotka..."):
                from utils import plot_lotkas_law
                fig_lotka = plot_lotkas_law(df)
                if fig_lotka:
                    st.plotly_chart(fig_lotka, use_container_width=True)
                else:
                    st.info("Dados insuficientes para gerar a Lei de Lotka.")
                    
        with col_prod:
            st.caption("O tamanho da bolha indica o volume de documentos publicados no ano, e a cor indica o volume total de citações recebidas. A linha horizontal marca o início e fim da atuação.")
            with st.spinner("Mapeando constância acadêmica..."):
                from utils import plot_author_production_over_time
                fig_prod = plot_author_production_over_time(df, top_n=15)
                if fig_prod:
                    st.plotly_chart(fig_prod, use_container_width=True)
                else:
                    st.info("Dados insuficientes para mapear a produção ao longo do tempo.")


        # --- HISTORIÓGRAFO (REDE DIRETA DE CITAÇÕES) ---
        st.divider()
        st.markdown("##### 🕰️ Histórico de Citações Diretas (Historiograph)")
        st.caption("Mapeamento cronológico dos documentos mais influentes da amostra. Os links representam o fluxo hereditário de ideias (quando um documento mais recente cita o autor principal de um documento seminal anterior).")

        col_hist_sel, col_hist_graf = st.columns([1, 3])
        
        # Estatísticas e Tabela Bruta (Da base já limpa, caso tenha rodado a limpeza)
        if 'TOTAL CITATIONS' in df.columns:
            stats_df = df['TOTAL CITATIONS'].dropna()
            if not stats_df.empty:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Média de Citações", f"{stats_df.mean():.2f}")
                e2.metric("Mediana", f"{stats_df.median():.2f}")
                e3.metric("Desvio Padrão", f"{stats_df.std():.2f}")
                e4.metric("Máximo", f"{stats_df.max():.0f}")

        with col_hist_sel:
            st.write("")
            top_n_hist = st.slider(
                "Documentos Analisados:", 
                min_value=10, max_value=100, value=30, step=10,
                key="slider_historiograph",
                help="Aumente para visualizar a malha histórica completa. Valores muito altos podem poluir visualmente o gráfico."
            )
            st.info("💡 **Leitura:** O eixo X é a linha do tempo. Documentos maiores são os mais citados da rede, servindo como os pilares do ecossistema de conhecimento.")
            
        with col_hist_graf:
            with st.spinner("Desenhando linha do tempo das citações..."):
                from utils import gerar_historiograph
                fig_hist = gerar_historiograph(df, top_n=top_n_hist)
                
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("Não foi possível traçar a rede histórica. Certifique-se de que sua base contém a coluna de 'Referências' (Cited References).")

        # --- LINHA 5 DE GRÁFICOS: MAPA TEMÁTICO (BIBLIOMETRIX STYLE) ---
        st.divider()
        st.markdown("##### 🗺️ Mapa Temático")
        st.caption("Esta visualização agrupa conceitos em 'comunidades' e os divide em quadrantes com base na sua **Densidade** (força dos laços internos do tema) e **Centralidade** (força dos laços do tema com outras áreas).")
        
        col_mapa_sel, col_mapa_graf = st.columns([1, 3])
        
        with col_mapa_sel:
            st.write("")
            fonte_mapa = st.selectbox(
                "Dados para Análise Temática:",
                ["Palavras-chave", "Títulos", "Resumo (Abstract)"],
                key="sel_mapa_tematico"
            )
            
            n_termos_mapa = st.slider(
                "Nº de Termos Processados:", 
                min_value=50, max_value=300, value=150, step=25,
                help="Mais termos geram redes mais complexas e demoradas. Menos termos focam apenas no essencial."
            )
            
            mapa_colunas_tematico = {
                "Títulos": next((c for c in ['TITLE', 'TI'] if c in df.columns), None),
                "Palavras-chave": next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None),
                "Resumo (Abstract)": next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)
            }
            coluna_mapa_escolhida = mapa_colunas_tematico[fonte_mapa]
            
            st.info("💡 **Dica de Leitura:**\n\n**Temas Motores:** Estruturados e essenciais.\n**Temas Básicos:** Gerais/transversais.\n**Nicho:** Muito especializados.\n**Emergentes:** Novos ou desaparecendo.")

        with col_mapa_graf:
            if coluna_mapa_escolhida and coluna_mapa_escolhida in df.columns:
                with st.spinner("Extraindo comunidades de conhecimento e calculando coordenadas..."):
                    
                    fig_mapa = gerar_mapa_tematico(df, coluna_texto=coluna_mapa_escolhida, n_palavras=n_termos_mapa)
                    
                    if fig_mapa:
                        st.plotly_chart(fig_mapa, use_container_width=True)
                    else:
                        st.warning("Volume de texto insuficiente ou sem padrões de co-ocorrência claros para gerar os quadrantes.")
            else:
                st.warning(f"Coluna de {fonte_mapa} não encontrada na base de dados.")

        # =========================================================
        # MÓDULO DE TABELAS E ESTATÍSTICAS DETALHADAS
        # =========================================================            
        st.markdown("### 📋 Tabelas Analíticas e Estatísticas")
        st.caption("Navegue pelas abas abaixo para investigar os dados agregados por diferentes dimensões do ecossistema científico.")
        
        df = st.session_state['df_original']
        
        # Criação das Abas
        aba_geral, aba_autores, aba_paises, aba_venues, aba_keywords = st.tabs([
            "📚 Tabela Geral", 
            "✍️ Autores", 
            "🌍 Países", 
            "🏛️ Fontes (Venues)", 
            "🔑 Palavras-chave"
        ])
        
        # --- ABA 1: TABELA GERAL ---
        with aba_geral:
            cols = [c for c in df.columns if c != 'YEAR CLEAN']
            prioridades = ['TEMA_GEMINI', 'TOTAL CITATIONS', 'TITLE', 'AUTHORS']
            for col_prioritaria in reversed(prioridades):
                if col_prioritaria in cols:
                    cols.insert(0, cols.pop(cols.index(col_prioritaria)))
            
            st.dataframe(df[cols], use_container_width=True)
            csv_geral = df[cols].to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Base Geral (CSV)", data=csv_geral, file_name='base_simetrics_geral.csv', key='dl_geral')

        # --- ABA 2: AUTORES ---
        with aba_autores:
            with st.spinner("Compilando perfil estatístico dos autores..."):
                from utils import gerar_tabela_autores
                df_autores = gerar_tabela_autores(df)
                if not df_autores.empty:
                    st.dataframe(df_autores, use_container_width=True, hide_index=True)
                    csv_autores = df_autores.to_csv(index=False).encode('utf-8')
                    st.download_button("Baixar Tabela de Autores (CSV)", data=csv_autores, file_name='simetrics_autores.csv', key='dl_aut')
                else:
                    st.info("Não há dados de autores suficientes para gerar esta tabela.")

        # --- ABA 3: PAÍSES ---
        with aba_paises:
            with st.spinner("Compilando estatísticas geopolíticas..."):
                from utils import gerar_tabela_paises
                df_paises = gerar_tabela_paises(df)
                if not df_paises.empty:
                    st.dataframe(df_paises, use_container_width=True, hide_index=True)
                    csv_paises = df_paises.to_csv(index=False).encode('utf-8')
                    st.download_button("Baixar Tabela de Países (CSV)", data=csv_paises, file_name='simetrics_paises.csv', key='dl_pai')
                else:
                    st.info("Não há dados de países (COUNTRY) suficientes para gerar esta tabela.")

        # --- ABA 4: FONTES / VENUES ---
        with aba_venues:
            with st.spinner("Agrupando periódicos e conferências..."):
                from utils import gerar_tabela_venues
                df_venues = gerar_tabela_venues(df)
                if not df_venues.empty:
                    st.dataframe(df_venues, use_container_width=True, hide_index=True)
                    csv_venues = df_venues.to_csv(index=False).encode('utf-8')
                    st.download_button("Baixar Tabela de Fontes/Venues (CSV)", data=csv_venues, file_name='simetrics_venues.csv', key='dl_ven')
                else:
                    st.info("Não há dados de fontes de publicação (SECONDARY TITLE) suficientes para gerar esta tabela.")

        # --- ABA 5: PALAVRAS-CHAVE ---
        with aba_keywords:
            with st.spinner("Calculando impacto do léxico..."):
                from utils import gerar_tabela_keywords
                df_keywords = gerar_tabela_keywords(df)
                if not df_keywords.empty:
                    st.dataframe(df_keywords, use_container_width=True, hide_index=True)
                    csv_keywords = df_keywords.to_csv(index=False).encode('utf-8')
                    st.download_button("Baixar Tabela de Palavras-chave (CSV)", data=csv_keywords, file_name='simetrics_keywords.csv', key='dl_kw')
                else:
                    st.info("Não há palavras-chave (KEYWORDS) suficientes para gerar esta tabela.")

    # === ABA 2: REDES E GRAFOS ===
    with tab_grafos:
        st.subheader("Mapeamento do Ecossistema de Conhecimento")
        col_opcoes, col_grafo = st.columns([1, 3])
        
        with col_opcoes:
            # 1. Adicionamos a opção no Selectbox
            tipo_grafo = st.selectbox(
                "Mapear:", 
                ["Rede de Coautoria", "Coocorrência de Palavras-chave", "Rede de Cocitação"]
            )
            
            top_n_nodes = st.slider("Top N Nós:", 10, 150, 50, 5)
            metric_for_size = st.selectbox(
                "Basear tamanho do nó em:", 
                ["Tamanho Fixo", "Grau Absoluto", "Centralidade (Eigen)", "Betweenness", "Closeness"]
            )
            
            # 2. Lógica de busca da coluna alvo atualizada
            coluna_alvo = None
            
            if tipo_grafo == "Rede de Coautoria":
                coluna_alvo = "AUTHORS"
            elif tipo_grafo == "Rede de Cocitação":
                # Busca variações comuns de nomes de colunas de referências
                for col in ['REFERENCES', 'CITED REFERENCES']:
                    if col in df.columns: 
                        coluna_alvo = col
                        break
            else: # Coocorrência de Palavras-chave
                for col in ['KEYWORDS', 'KW', 'DE']:
                    if col in df.columns: 
                        coluna_alvo = col
                        break

            # Feedback visual caso a coluna não exista na base carregada
            if not coluna_alvo:
                st.warning(f"⚠️ A coluna necessária para '{tipo_grafo}' não foi encontrada na sua base de dados.")

        with col_grafo:
            if coluna_alvo and coluna_alvo in df.columns:
                with st.spinner("Calculando topologia SNA..."):
                    
                    # ATUALIZAÇÃO: A função agora retorna 5 elementos (incluindo o G_obj)
                    from utils import criar_grafo_e_metricas, plot_grafo_estatico
                    nodes, edges, df_nodes, net_metrics, G_obj = criar_grafo_e_metricas(df, coluna_alvo, top_n_nodes, metric_for_size)
                    
                    if len(nodes) > 0:
                        config = Config(
                            width="100%", 
                            height=700, 
                            directed=False, 
                            hierarchical=False,
                            navigationButtons=True, 
                            interaction={
                                "hover": True, 
                                "zoomView": True, 
                                "dragView": True,
                                "navigationButtons": True 
                            },
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            stabilization=True,
                            edges={
                                "smooth": {
                                    "enabled": True,
                                    "type": "dynamic" 
                                }
                            }
                        )
                        
                        # 1. Grafo Dinâmico (Agora com Tooltips ao passar o mouse!)
                        agraph(nodes=nodes, edges=edges, config=config)
                        
                        # 2. Grafo Estático (Exportável para Tese/Artigos)
                        st.markdown("### 📸 Instantâneo da Rede (Estilo VOSviewer)")
                        st.caption("Esta versão estática agrupa os nós em cores por **comunidade** (Algoritmo de Louvain). Clique com o botão direito para salvar a imagem em alta resolução.")
                        
                        with st.spinner("Desenhando instantâneo de alta resolução..."):
                            titulo_grafo = f"Rede Estática: {tipo_grafo}"
                            fig_estatica = plot_grafo_estatico(G_obj, titulo=titulo_grafo)
                            if fig_estatica:
                                st.pyplot(fig_estatica)
                            else:
                                st.info("Não foi possível gerar a versão estática.")

                    else: st.warning("Sem conexões suficientes.")
            else: st.warning("Coluna não encontrada.")

        st.divider()
        st.markdown("### 📊 Tabela de Nós e Métricas SNA (Rede Heterogênea)")
        st.caption("Esta tabela integra Autores, Documentos, Países e Venues, permitindo comparar a influência transversal de diferentes entidades. Por exigir alto processamento, o cálculo é feito sob demanda.")
        
        # Cria um espaço na memória da sessão para guardar a tabela gerada
        if "tabela_sna_completa" not in st.session_state:
            st.session_state["tabela_sna_completa"] = None

        # Botão de Ação
        if st.button("🚀 Iniciar Cálculo da Tabela SNA Completa", type="primary"):
            pbar_sna = st.progress(0, text="Calculando centralidades de todo o ecossistema...")
            
            # Executa o cálculo e guarda na memória da sessão
            st.session_state["tabela_sna_completa"] = gerar_tabela_metricas_completas(df, _pbar=pbar_sna)
            
            pbar_sna.empty() # Remove a barra ao terminar

        # Renderização da Tabela (Ocorre se a tabela já existir na memória)
        if st.session_state["tabela_sna_completa"] is not None:
            df_metricas = st.session_state["tabela_sna_completa"]
            
            if not df_metricas.empty:
                # Filtro rápido por tipo para o usuário
                tipos_disponiveis = ["Todos"] + sorted(df_metricas["Tipo"].unique().tolist())
                filtro_tipo = st.selectbox("Filtrar por Tipo de Item:", tipos_disponiveis)

                df_filtrado = df_metricas if filtro_tipo == "Todos" else df_metricas[df_metricas["Tipo"] == filtro_tipo]

                # Exibição da Tabela com a configuração correta de colunas
                st.dataframe(
                    df_filtrado,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Item": st.column_config.TextColumn("Item"),
                        "Tipo": st.column_config.TextColumn("Tipo"),
                        "Grau Absoluto": st.column_config.NumberColumn("Grau Absoluto", format="%d"),
                        "Grau Centralidade": st.column_config.NumberColumn("Grau Centralidade", format="%.4f"),
                        "Betweenness": st.column_config.NumberColumn("Betweenness", format="%.4f"),
                        "Closeness": st.column_config.NumberColumn("Closeness", format="%.4f")
                    }
                )
                
                st.download_button(
                    "📥 Baixar Relatório SNA (CSV)",
                    data=df_filtrado.to_csv(index=False).encode('utf-8'),
                    file_name="metricas_sna_ecossistema.csv"
                )
            else:
                st.warning("Não há dados suficientes para calcular a rede heterogênea.")
        else:
            st.info("A tabela está em modo de espera. Clique no botão azul acima para iniciar o processamento topológico.")

    # Inicializa o estado de busca se não existir
    if 'busca_tipo_biblio' not in st.session_state:
        st.session_state['busca_tipo_biblio'] = "Documento"
    if 'busca_termo_biblio' not in st.session_state:
        st.session_state['busca_termo_biblio'] = None

    # =========================================================
    # --- ABA DO MOTOR DE BUSCA ---
    # =========================================================
    with tab_search:
        st.header("🔍 Motor de Busca e Dossiê Científico")
        st.caption("Investigue as entidades do ecossistema e descubra sua influência topológica na rede.")

        # 1. Preparação das Listas Únicas
        col_titulos = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
        col_autores = next((c for c in ['AUTHORS', 'AU'] if c in df.columns), None)
        col_paises = next((c for c in ['COUNTRY'] if c in df.columns), None)
        col_venue = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
        col_ano = next((c for c in ['YEAR', 'PY'] if c in df.columns), None)

        opcoes_doc = df[col_titulos].dropna().unique().tolist() if col_titulos else []
        
        autores_raw = df[col_autores].dropna().tolist() if col_autores else []
        opcoes_aut = sorted(list(set([a.strip() for sub in autores_raw for a in str(sub).split(';') if a.strip()])))
        
        paises_raw = df[col_paises].dropna().tolist() if col_paises else []
        opcoes_pais = sorted(list(set([p.strip() for sub in paises_raw for p in str(sub).split(';') if p.strip()])))
        
        opcoes_venue = sorted(df[col_venue].dropna().unique().tolist()) if col_venue else []

        # 2. Interface de Busca
        opcoes_busca = ["Documento", "Autor", "País", "Local de Publicação (Venue)"]
        
        tipo_busca = st.radio(
            "Procurar por Entidade:", 
            opcoes_busca, 
            horizontal=True, 
            key="busca_tipo_biblio", 
            on_change=limpar_termo_busca 
        )

        if st.session_state['busca_tipo_biblio'] == "Documento": opcoes_lista = opcoes_doc
        elif st.session_state['busca_tipo_biblio'] == "Autor": opcoes_lista = opcoes_aut
        elif st.session_state['busca_tipo_biblio'] == "País": opcoes_lista = opcoes_pais
        elif st.session_state['busca_tipo_biblio'] == "Local de Publicação (Venue)": opcoes_lista = opcoes_venue

        termo_selecionado = st.selectbox(
            "Selecione ou digite para pesquisar:", 
            sorted(opcoes_lista), 
            index=sorted(opcoes_lista).index(st.session_state['busca_termo_biblio']) if st.session_state['busca_termo_biblio'] in opcoes_lista else None, 
            placeholder="Explore o ecossistema..."
        )

        if termo_selecionado != st.session_state['busca_termo_biblio']:
            st.session_state['busca_termo_biblio'] = termo_selecionado
            st.rerun()

        termo_ativo = st.session_state['busca_termo_biblio']
        tipo_ativo = st.session_state['busca_tipo_biblio']

        # --- ARQUITETURA ULTRARRÁPIDA: Grafo Global em Memória ---
        # Usamos cache_resource porque grafos são objetos complexos de rede, não apenas dados tubulares.
        @st.cache_resource 
        def obter_grafo_global(df_dados):
            G = nx.Graph()
            colunas_necessarias = [c for c in [col_titulos, col_autores, col_paises, col_venue] if c is not None]
            records = df_dados[colunas_necessarias].to_dict('records')
            
            for row in records:
                doc_node = str(row.get(col_titulos, ''))
                if not doc_node or doc_node == 'nan': continue
                G.add_node(doc_node, type='Documento')
                
                if col_autores and pd.notna(row.get(col_autores)):
                    for a in [x.strip() for x in str(row[col_autores]).split(';') if x.strip()]:
                        G.add_node(a, type='Autor')
                        G.add_edge(doc_node, a)
                if col_paises and pd.notna(row.get(col_paises)):
                    for p in [x.strip() for x in str(row[col_paises]).split(';') if x.strip()]:
                        G.add_node(p, type='País')
                        G.add_edge(doc_node, p)
                if col_venue and pd.notna(row.get(col_venue)):
                    venue = str(row[col_venue]).strip()
                    G.add_node(venue, type='Venue')
                    G.add_edge(doc_node, venue)
            return G

        # 3. Construção do Perfil e Métricas SNA
        if termo_ativo:
            col_info, col_sna = st.columns([2, 1])
            
            # Recupera a rede global instantaneamente
            grafo_global = obter_grafo_global(df)
            
            def calcular_sna_instantaneo(G, termo):
                """Calcula métricas localizadas baseadas no Grafo Global pré-carregado."""
                if termo not in G: return None
                
                grau_abs = G.degree(termo)
                # Cálculo de centralidade de grau isolado (super rápido: grau / nós possíveis)
                total_nos = len(G)
                cent_grau = grau_abs / (total_nos - 1) if total_nos > 1 else 0
                
                # Extrai apenas os vizinhos diretos (radius=1) para a intermediação, economizando processamento pesado
                ego_net = nx.ego_graph(G, termo, radius=1) 
                betw = nx.betweenness_centrality(ego_net).get(termo, 0)
                clos = nx.closeness_centrality(ego_net).get(termo, 0)
                
                return {"Grau Absoluto": grau_abs, "Centralidade Grau": cent_grau, "Betweenness": betw, "Closeness": clos}

            # --- RENDERIZAÇÃO DO PERFIL (COLUNA ESQUERDA) ---
            with col_info:
                st.info(f"**{tipo_ativo}:** {termo_ativo}")
                
                if tipo_ativo == "Documento":
                    doc = df[df[col_titulos] == termo_ativo].iloc[0]
                    
                    ano = doc[col_ano] if col_ano and pd.notna(doc[col_ano]) else 'N/A'
                    citacoes = doc['TOTAL CITATIONS'] if 'TOTAL CITATIONS' in df.columns and pd.notna(doc['TOTAL CITATIONS']) else 0
                    doi = doc['DOI'] if 'DOI' in doc and pd.notna(doc['DOI']) else None
                    
                    st.write(f"**Ano de Publicação:** {ano} | **Citações Recebidas:** {citacoes}")
                    if doi: st.markdown(f"🔗 **Link DOI:** [https://doi.org/{doi}](https://doi.org/{doi})")
                    
                    if col_venue and pd.notna(doc[col_venue]):
                        venue_nome = doc[col_venue]
                        st.write("**Publicado em:**")
                        st.button(f"🏢 {venue_nome}", key=f"btn_nav_venue_{hash(venue_nome)}", on_click=navegar_busca, args=("Local de Publicação (Venue)", venue_nome))
                        
                    if col_autores and pd.notna(doc[col_autores]):
                        st.write("**Rede de Autoria (clique para ver perfil):**")
                        autores_doc = [a.strip() for a in str(doc[col_autores]).split(';') if a.strip()]
                        for i, a in enumerate(autores_doc):
                            st.button(f"👤 {a}", key=f"btn_nav_aut_doc_{hash(a)}_{i}", on_click=navegar_busca, args=("Autor", a))
                    
                    col_abstract = next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)
                    if col_abstract and pd.notna(doc[col_abstract]):
                        with st.expander("Ler Resumo (Abstract)"):
                            st.write(doc[col_abstract])
                            
                elif tipo_ativo == "Autor":
                    docs_autor = df[df[col_autores].fillna('').str.contains(termo_ativo, regex=False)]
                    total_citacoes = docs_autor['TOTAL CITATIONS'].sum() if 'TOTAL CITATIONS' in docs_autor.columns else 0
                    st.write(f"**Impacto Total (Citações):** {total_citacoes}")
                    
                    # Coautores
                    parceiros = []
                    for _, r in docs_autor.iterrows():
                        if pd.notna(r[col_autores]):
                            parceiros.extend([a.strip() for a in str(r[col_autores]).split(';') if a.strip() and a.strip() != termo_ativo])
                    top_parceiros = Counter(parceiros).most_common(5)
                    
                    if top_parceiros:
                        st.write("**🤝 Principais Coautores (clique para ver perfil):**")
                        for i, (p, qtd) in enumerate(top_parceiros):
                            st.button(f"🤝 {p} ({qtd} docs)", key=f"btn_nav_coaut_{hash(p)}_{i}", on_click=navegar_busca, args=("Autor", p))
                    
                    with st.expander(f"📚 Documentos Publicados ({len(docs_autor)})"):
                        for i, (_, r) in enumerate(docs_autor.iterrows()):
                            titulo_doc = r[col_titulos]
                            st.button(f"📄 {titulo_doc} ({r.get(col_ano, 'N/A')})", key=f"btn_nav_doc_aut_{hash(titulo_doc)}_{i}", on_click=navegar_busca, args=("Documento", titulo_doc))

                elif tipo_ativo == "País":
                    docs_pais = df[df[col_paises].fillna('').str.contains(termo_ativo, regex=False)]
                    total_citacoes = docs_pais['TOTAL CITATIONS'].sum() if 'TOTAL CITATIONS' in docs_pais.columns else 0
                    st.write(f"**Impacto do País (Citações):** {total_citacoes}")
                    
                    with st.expander(f"📚 Ver Documentos Associados ({len(docs_pais)})"):
                        for i, (_, r) in enumerate(docs_pais.iterrows()):
                            titulo_doc = r[col_titulos]
                            st.button(f"📄 {titulo_doc}", key=f"btn_nav_doc_pais_{hash(titulo_doc)}_{i}", on_click=navegar_busca, args=("Documento", titulo_doc))

                elif tipo_ativo == "Local de Publicação (Venue)":
                    docs_venue = df[df[col_venue] == termo_ativo]
                    total_citacoes = docs_venue['TOTAL CITATIONS'].sum() if 'TOTAL CITATIONS' in docs_venue.columns else 0
                    st.write(f"**Citações Acumuladas nesta Fonte:** {total_citacoes}")
                    
                    with st.expander(f"📚 Ver Documentos Publicados Aqui ({len(docs_venue)})"):
                        for i, (_, r) in enumerate(docs_venue.iterrows()):
                            titulo_doc = r[col_titulos]
                            st.button(f"📄 {titulo_doc}", key=f"btn_nav_doc_venue_{hash(titulo_doc)}_{i}", on_click=navegar_busca, args=("Documento", titulo_doc))

            # --- RENDERIZAÇÃO DAS MÉTRICAS SNA (COLUNA DIREITA) ---
            with col_sna:
                st.markdown("##### 🕸️ Métricas Topológicas (SNA)")
                
                # Botão sob demanda
                if st.button(f"⚙️ Calcular SNA para: {termo_ativo}", key=f"btn_calc_sna_{hash(termo_ativo)}", use_container_width=True):
                    with st.spinner("Analisando centralidade na rede..."):
                        metricas_sna = calcular_sna_instantaneo(grafo_global, termo_ativo)
                        
                        if metricas_sna:
                            st.metric("Grau Absoluto (Conexões)", metricas_sna['Grau Absoluto'], help="Total de conexões diretas desta entidade na rede.")
                            st.metric("Centralidade de Grau", f"{metricas_sna['Centralidade Grau']:.4f}", help="Proporção da rede com a qual esta entidade se conecta diretamente.")
                            st.metric("Betweenness Centrality", f"{metricas_sna['Betweenness']:.4f}", help="Capacidade de agir como 'ponte' no fluxo de informação entre outros nós.")
                            st.metric("Closeness Centrality", f"{metricas_sna['Closeness']:.4f}", help="O quão perto (em saltos) esta entidade está de todas as outras na rede.")
                        else:
                            st.warning("Métricas isoladas/não aplicáveis.")
                else:
                    st.info("Aperte o botão acima para calcular o impacto topológico (Grau, Betweenness, Closeness) desta entidade específica no ecossistema.")

            st.markdown("---")

            # =========================================================
            # ABAS DO DOSSIÊ (HISTÓRICO, NUVEM E SIMILARES)
            # =========================================================
            tab_hist, tab_nuvem, tab_similares = st.tabs(["📈 Evolução Histórica", "☁️ Lexicometria", "🔗 Itens Semelhantes"])

            if not termo_ativo:
                subset_df = pd.DataFrame(columns=df.columns)
            elif tipo_ativo == "Documento": 
                subset_df = df[df[col_titulos] == termo_ativo] if col_titulos else pd.DataFrame(columns=df.columns)
            elif tipo_ativo == "Autor": 
                subset_df = df[df[col_autores].fillna('').str.contains(str(termo_ativo), regex=False)] if col_autores else pd.DataFrame(columns=df.columns)
            elif tipo_ativo == "País": 
                subset_df = df[df[col_paises].fillna('').str.contains(str(termo_ativo), regex=False)] if col_paises else pd.DataFrame(columns=df.columns)
            elif tipo_ativo == "Local de Publicação (Venue)": 
                subset_df = df[df[col_venue] == termo_ativo] if col_venue else pd.DataFrame(columns=df.columns)
            else:
                subset_df = pd.DataFrame(columns=df.columns)

            # --- ABA 1: HISTÓRICO AVANÇADO ---
            with tab_hist:
                st.markdown(f"**Produção e Impacto ao Longo do Tempo**")
                
                nomes_comuns_tipo = [
                    'TYPE', 'DT', 'DOCUMENT TYPE', 'TY', 'TIPO', 
                    'TIPO DE DOCUMENTO', 'TYPE OF REFERENCE', 'REFERENCE TYPE'
                ]
                
                col_tipo_doc = next((c for c in df.columns if str(c).strip().upper() in nomes_comuns_tipo), None)
                
                opcoes_visao = ["Visão Geral"]
                if col_tipo_doc: 
                    opcoes_visao.append("Separado por Tipo de Documento")
                
                visao_hist = st.radio(
                    "Análise Histórica:", 
                    opcoes_visao, 
                    horizontal=True, 
                    key=f"rad_hist_{hash(termo_ativo)}"
                )
                
                if col_ano and not subset_df.empty:
                    df_ano = subset_df.copy()
                    df_ano[col_ano] = pd.to_numeric(df_ano[col_ano], errors='coerce')
                    df_ano = df_ano.dropna(subset=[col_ano])
                    
                    if not df_ano.empty:
                    
                        if visao_hist == "Visão Geral":
                            if 'TOTAL CITATIONS' in df_ano.columns:
                                hist_data = df_ano.groupby(col_ano).agg(Volume=(col_titulos, 'count'), Citacoes=('TOTAL CITATIONS', 'sum')).reset_index()
                            else:
                                hist_data = df_ano.groupby(col_ano).size().reset_index(name='Volume')
                                hist_data['Citacoes'] = 0
                                
                            # CORREÇÃO ARROW: Forçamos o tipo Inteiro para Volume para impedir o colapso do PyArrow
                            hist_data['Volume'] = pd.to_numeric(hist_data['Volume'], errors='coerce').fillna(0).astype(int)
                                
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Bar(x=hist_data[col_ano], y=hist_data['Volume'], name="Documentos", marker_color="#2a9d8f"))
                            fig_hist.add_trace(go.Scatter(x=hist_data[col_ano], y=hist_data['Citacoes'], name="Citações", mode='lines+markers', yaxis='y2', line=dict(color="#e76f51", width=3)))
                            fig_hist.update_layout(
                                title="Volume de Publicações vs Citações no Tempo",
                                xaxis=dict(title="Ano", tickmode='linear', dtick=1),
                                yaxis=dict(title=dict(text="Volume de Documentos", font=dict(color="#2a9d8f")), tickfont=dict(color="#2a9d8f")),
                                yaxis2=dict(title=dict(text="Total de Citações", font=dict(color="#e76f51")), tickfont=dict(color="#e76f51"), overlaying='y', side='right'),
                                template="plotly_white", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                            df_ano[col_tipo_doc] = df_ano[col_tipo_doc].fillna("Desconhecido")
                            hist_data = df_ano.groupby([col_ano, col_tipo_doc]).size().reset_index(name='Volume')
                            
                            # CORREÇÃO ARROW: Forçamos o tipo Inteiro para Volume 
                            hist_data['Volume'] = pd.to_numeric(hist_data['Volume'], errors='coerce').fillna(0).astype(int)
                            
                            fig_hist = px.bar(hist_data, x=col_ano, y='Volume', color=col_tipo_doc, title="Volume de Documentos por Tipo e Ano", template="plotly_white")
                            fig_hist.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                            st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("Não há dados temporais válidos para gerar o histórico.")
                else:
                    st.info("A coluna de Ano não está disponível para esta análise.")

            # --- ABA 2: NUVEM DE PALAVRAS CUSTOMIZÁVEL ---
            with tab_nuvem:
                st.markdown(f"**Assinatura Semântica do Perfil**")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    fonte_txt = st.selectbox("Composição do Texto:", ["Tudo Combinado", "Apenas Títulos", "Apenas Palavras-chave", "Apenas Resumo"], key=f"src_wc_{hash(termo_ativo)}")
                with c2:
                    estilo_txt = st.selectbox("Tipografia:", ["Arial", "Verdana", "Courier New", "Georgia", "Impact", "Trebuchet MS"], key=f"font_wc_{hash(termo_ativo)}")
                with c3:
                    tema_cor = st.selectbox("Paleta:", ["Oceano", "Fogo", "Floresta", "Cyberpunk", "Acadêmico"], index=4, key=f"pal_wc_{hash(termo_ativo)}")
                    
                paletas_dict = {
                    "Oceano": ["#0077b6", "#00b4d8", "#90e0ef", "#03045e", "#023e8a"],
                    "Fogo": ["#ff4d00", "#ff8c00", "#ff0000", "#fad02c", "#e85d04"],
                    "Floresta": ["#2d6a4f", "#40916c", "#1b4332", "#74c69d", "#95d5b2"],
                    "Cyberpunk": ["#f72585", "#7209b7", "#3a0ca3", "#4361ee", "#4cc9f0"],
                    "Acadêmico": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
                }

                textos_para_juntar = []
                col_ab = next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)
                col_kw = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None)
                
                # CORREÇÃO 2: Verificamos se a coluna está em subset_df.columns antes de acessá-la
                if ("Tudo" in fonte_txt or "Título" in fonte_txt) and col_titulos in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_titulos].dropna().astype(str)))
                if ("Tudo" in fonte_txt or "Palavras-chave" in fonte_txt) and col_kw in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_kw].dropna().astype(str).str.replace(';', ' ')))
                if ("Tudo" in fonte_txt or "Resumo" in fonte_txt) and col_ab in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_ab].dropna().astype(str)))

                texto_final_vetorizado = " ".join(textos_para_juntar)
                
                subset_df_wc = pd.DataFrame({'TEXTO_COMBINADO': [texto_final_vetorizado]})

                with st.spinner("Gerando nuvem de palavras específica..."):
                  
                    
                    wc_opcoes = gerar_nuvem_echarts(
                        subset_df_wc, 
                        coluna='TEXTO_COMBINADO', 
                        fonte=estilo_txt, 
                        paleta=paletas_dict[tema_cor]
                    )
                    
                    if wc_opcoes:
                        st_echarts(options=wc_opcoes, height="450px", key=f"wc_{hash(termo_ativo)}_{fonte_txt}_{tema_cor}")
                    else:
                        st.warning("Texto insuficiente nos documentos desta entidade para gerar a nuvem semântica.")

            # --- ABA 3: CÁLCULO DE SIMILARIDADE ---
            with tab_similares:
                st.markdown(f"**Recomendação Topológica (Itens mais próximos de {termo_ativo})**")
                st.caption("A proximidade é calculada pelo **Índice de Jaccard**, que mede a sobreposição estrutural de palavras-chave, coautorias e locais de publicação na sua base bibliométrica.")
                
                with st.spinner("Calculando similaridade vetorial na rede..."):
                    similares = calcular_similares_biblio(termo_ativo, tipo_ativo, df)
                    
                def render_tabela_similares(lista_dados, titulo_coluna_item, tipo_nav):
                    if not lista_dados:
                        st.info(f"Nenhum item com forte correlação encontrado.")
                        return
                        
                    df_sim = pd.DataFrame(lista_dados)[['Item', 'Similaridade (%)', 'Traços em Comum']]
                    df_sim = df_sim.rename(columns={'Item': titulo_coluna_item})
                    
                    st.dataframe(
                        df_sim, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Similaridade (%)": st.column_config.ProgressColumn("Similaridade (%)", min_value=0, max_value=100, format="%.1f%%")
                        }
                    )
                    
                    with st.expander(f"Navegar para os perfis ({titulo_coluna_item})"):
                        for idx, row in df_sim.iterrows():
                            item_nome = row[titulo_coluna_item]
                            st.button(f"Ir para: {item_nome}", key=f"btn_sim_{tipo_nav}_{hash(item_nome)}_{idx}", on_click=navegar_busca, args=(tipo_nav, item_nome))

                if not similares or all(len(v) == 0 for v in similares.values()):
                    st.warning("Este item possui conexões muito isoladas do resto do sistema para que vizinhos próximos sejam calculados com precisão.")
                else:
                    if tipo_ativo == 'Documento':
                        st.markdown("##### 📄 Documentos Semelhantes")
                        render_tabela_similares(similares.get('Documentos', []), "Documento", "Documento")
                    elif tipo_ativo == 'Autor':
                        st.markdown("##### ✍️ Autores com Perfil Semelhante")
                        render_tabela_similares(similares.get('Autores', []), "Autor", "Autor")
                    elif tipo_ativo in ['País', 'Local de Publicação (Venue)']:
                        st.markdown(f"##### 🔗 Entidades Semelhantes ({tipo_ativo})")
                        render_tabela_similares(similares.get('Itens', []), tipo_ativo, tipo_ativo)