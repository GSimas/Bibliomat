import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_agraph import agraph, Config
from streamlit_echarts import st_echarts
import io
import os
from collections import Counter
import networkx as nx
from google import genai
if "SECRETS_TOML" in os.environ:
    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/secrets.toml", "w", encoding="utf-8") as f:
        f.write(os.environ["SECRETS_TOML"])
from streamlit_gsheets import GSheetsConnection
from utils import (
    processar_pubmed,
    processar_cochrane,
    analisar_completude_metadados,
    calcular_metricas_bibliometrix,
    calcular_similares_biblio,
    criar_grafo_e_metricas,
    deduplicar_por_doi,
    deduplicar_por_similaridade,
    filtrar_por_entidade,
    gerar_csv_bytes,
    gerar_mapa_tematico,
    gerar_nuvem_echarts,
    gerar_tabela_metricas_completas,
    limpar_termo_busca,
    navegar_busca,
    obter_grafo_global_busca,
    padronizar_base_bibliometrica,
    preparar_opcoes_busca,
    process_multiple_ris,
    processar_csv_scopus,
    processar_excel_wos,
    resumir_base_bibliometrica
)


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


def get_gemini_api_key():
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
        return secret_key or None
    except Exception:
        return None

def resetar_estado_derivado():
    st.session_state['df_duplicados'] = pd.DataFrame()
    st.session_state['tabela_sna_completa'] = None
    st.session_state['metricas_globais_sna'] = None


st.session_state.setdefault('df_geral', None)
st.session_state.setdefault('df_original', None)
st.session_state.setdefault('df_duplicados', pd.DataFrame())
st.session_state.setdefault('tabela_sna_completa', None)
st.session_state.setdefault('mostrar_descritivo', True)


# ==========================================================
# --- DESCRITIVO INICIAL DO SIMETRICS ---
# ==========================================================
if st.session_state.get('mostrar_descritivo', True) and st.session_state['df_geral'] is None:
    with st.container():
        st.markdown("""
        ### 📚 O que é o Simetrics?
        
        **Simetrics** é uma plataforma avançada de análise bibliométrica e cientométrica que transforma dados brutos de pesquisa acadêmica em inteligência estratégica. Desenvolvida para pesquisadores, gestores de ciência e tomadores de decisão, o Simetrics oferece uma visão 360° do ecossistema científico.
        
        ---
        
        #### 🎯 Funcionalidades Principais
        
        **1. 📊 Análise Bibliométrica Completa**
        - Visualize a dinâmica de produção científica ao longo do tempo
        - Identifique tendências, ciclos e períodos de crescimento
        - Compare métricas de impacto por base de dados, país, autor e venue
        - Análise estatística avançada com distribuições (boxplots) e outliers
        
        **2. 🤖 Inteligência Artificial & Categorização**
        - Mapeamento automático de escolas temáticas via Machine Learning (Gemini AI)
        - Agrupamento semântico de documentos sem classificação manual pré-existente
        - Identificação do Quociente Locacional (QL) para descobrir especialização vocacional
        - Análise da genética das ideias: mortalidade, fecundidade e longevidade do conhecimento
        
        **3. 🕸️ Redes de Conhecimento (SNA)**
        - **Rede de Coautoria:** Visualize colaborações entre pesquisadores
        - **Rede de Cocitação:** Descubra documentos seminais que fundamentam a pesquisa
        - **Coocorrência de Palavras-chave:** Mapeie a estrutura conceitual do campo
        - **Análise Heterogênea:** Integre autores, documentos, países e venues em um único grafo
        - Métricas avançadas: Centralidade, Betweenness, Closeness, Lei de Potência
        
        **4. 🔍 Motor de Busca Inteligente**
        - Investigue qualquer entidade: documentos, autores, países, venues ou temas
        - Visualize histórico temporal e evolução de interesse
        - Descubra especialização temática (QL) e lideranças por tema
        - Encontre documentos semelhantes via similaridade de conteúdo
        - Lexicometria customizável: nuvem de palavras com múltiplas fontes e estilos
        
        **5. 📈 Ecologia Profunda & Topologia**
        - Análise da Lei de Potência (comumente observada em redes científicas)
        - Correlação de Spearman: identifique inovação (Brokers) vs hierarquias
        - Assortatividade: redes expansivas vs fechadas (endógenas)
        - Diâmetro, Densidade e Clusters: entenda a saúde global da rede
        
        **6. 🗺️ Mapeamentos Conceituais**
        - **PCA 2D e 3D:** Redução dimensional interativa com clustering
        - **Mapa Temático (4 Quadrantes):** Classificação por Densidade vs Centralidade
        - **Sankey Temporal:** Acompanhe a evolução de termos em 3 períodos
        - **Historiograph:** Linha do tempo de citações diretas
        - **Colaboração Internacional:** Redes circulares e mapas-múndi interativos
        
        **7. 🎛️ Ferramentas de Limpeza & Qualidade**
        - Deduplicação por DOI exato e similaridade de título
        - Análise de completude de metadados (semáforo: Excelente/Bom/Aceitável/Crítico)
        - Relatório permanente de documentos excluídos (para metodologia PRISMA)
        - Controle granular de thresholds
        
        **8. 🤖 Assistente Científico (Chatbot)**
        - Converse com sua base de dados em linguagem natural
        - Peça recomendações de leitura sobre temas específicos
        - Identifique especialistas e principais pesquisadores
        - Descubra os melhores venues para submeter seu artigo
        - Baseado em Google Gemini com contexto sincronizado
        
        **9. 📥 Importação Flexível**
        - Suporte para múltiplas bases: **Scopus**, **Web of Science**, **SciELO**, **PubMed**, **Cochrane**
        - Formatos aceitos: RIS, CSV, Excel, TXT/NBIB
        - Integração automática de múltiplos arquivos
        - Modo Demonstração: explore com dados de exemplo pré-carregados
        
        ---
        
        #### 💡 Casos de Uso
        
        ✅ **Pesquisadores:** Realize revisões sistemáticas e mapeamentos de literatura com rigor cientométrico
        
        ✅ **Gestores de CT&I:** Monitore a ciência produzida, identifique clusters de pesquisa e aloque recursos
        
        ✅ **Editores & Revisores:** Avalie qualidade, impacto e relevância de manuscritos submetidos
        
        ✅ **Policy Makers:** Tome decisões baseadas em dados sobre investimento em P&D
        
        ✅ **Estudantes de Pós-Graduação:** Estruture sua revisão bibliográfica e descubra gaps de pesquisa
        
        ---
        
        #### 🚀 Como Começar
        
        1️⃣ **Importe seus dados** pelo painel à esquerda (suporta RIS, CSV, Excel, TXT/NBIB)
        
        2️⃣ **Escolha a base de dados** para cada arquivo (Scopus, WoS, SciELO, PubMed, Cochrane)
        
        3️⃣ **Clique em \"Processar e Integrar\"** para consolidar e padronizar
        
        4️⃣ **Explore as 4 abas principais:**
           - 📊 **Informações Principais:** Métricas globais, rankings e distribuições
           - 🕸️ **Redes e Grafos:** SNA, coautoria, cocitação e análise heterogênea
           - 🔍 **Motor de Busca:** Dossiês, perfis e evolução de entidades
           - 🤖 **Assistente Científico:** Chat com sua base de dados
        
        ---
        
        #### ⚙️ Notas Técnicas
        
        - **Limite:** Até 10.000 documentos por processamento
        - **Latência:** Análises de redes grandes podem levar minutos (visualizações são processadas sob demanda)
        - **Qualidade de dados:** Metadados incompletos podem afetar precisão de co-ocorrência e clustering
        - **IA Responsável:** Sempre verificar interpretações do Gemini com os dados brutos
        
        ---
        
        ✨ **Pronto para explorar?** Suba um arquivo e descubra os padrões escondidos em sua pesquisa!
        """)
    st.divider()

with st.sidebar:
    st.image("simetrics - logo.png", width='stretch')

    st.header("1. Envio de Arquivos")
    
    # --- GUIA DE FORMATOS SUPORTADOS ATUALIZADO ---
    with st.expander("ℹ️ Formatos e Bases Suportadas", expanded=False):
        st.markdown("""
        | Extensão | Base de Dados |
        | :--- | :--- |
        | **.ris** | SciELO, WoS, Scopus, Mendeley, Cochrane |
        | **.csv** | Scopus (Exportação Direta), Cochrane |
        | **.xls / .xlsx**| Web of Science (Full Record) |
        | **.txt / .nbib**| PubMed (Formato Medline) |
        
        **Dicas Importantes:**
        * O limite de processamento é 10000 linhas, ou seja, 10000 documentos
        * **WoS:** Para Excel, certifique-se de exportar com 'Full Record and Cited References'.
        * **PubMed:** Use a opção de exportação 'Format: PubMed' no site da NCBI.
        """)
    
    # Uploader configurado para aceitar todos os formatos do ecossistema Simetrics
    uploaded_files = st.file_uploader(
        "Selecione arquivos RIS, CSV, Excel, TXT ou NBIB", 
        type=['ris', 'csv', 'xls', 'xlsx', 'txt', 'nbib'], 
        accept_multiple_files=True
    )
    
    db_mapping = {}
    if uploaded_files:
        st.header("2. Atribuição de Base")
        for f in uploaded_files:
            ext = f.name.lower()
            
            # Sugestão inteligente baseada na extensão e nome do arquivo
            if ext.endswith('.csv'):
                if 'cochrane' in f.name.lower(): def_idx = 5
                else: def_idx = 0 # Scopus
            elif ext.endswith(('.xls', '.xlsx')): def_idx = 1 # Web of Science
            elif ext.endswith(('.txt', '.nbib')): def_idx = 3 # PubMed
            elif ext.endswith('.ris'):
                if 'cochrane' in f.name.lower(): def_idx = 5
                else: def_idx = 4 # Outra (Mendeley, SciELO)
            else: def_idx = 4
            
            db_mapping[f.name] = st.selectbox(
                f"{f.name}", 
                options=["Scopus", "Web of Science", "SciELO", "PubMed", "Cochrane", "Outra"], # "Cochrane" adicionada (Índice 5)
                index=def_idx,
                key=f"db_{f.name}"
            )
            
        if st.button("Processar e Integrar", type="primary"):
            st.session_state['mostrar_descritivo'] = False
            
            # 1. Inicializa a barra de progresso
            pbar_load = st.progress(0, text="Iniciando integração de dados...")
            
            list_dfs = []
            total_files = len(uploaded_files)
            
            for i, f in enumerate(uploaded_files):
                # 2. Atualiza a barra para cada arquivo processado
                progresso_atual = (i + 1) / total_files
                pbar_load.progress(progresso_atual, text=f"Integrando {f.name} ({i+1}/{total_files})")
                
                ext = f.name.lower()
                base_escolhida = db_mapping[f.name]
                
                # MOTOR DE ROTEAMENTO (Prioriza a escolha do usuário no Dropdown)
                if base_escolhida == "Cochrane":
                    df_temp = processar_cochrane(f, f.name)
                elif base_escolhida == "PubMed" or ext.endswith(('.txt', '.nbib')):
                    df_temp = processar_pubmed(f)
                elif ext.endswith('.csv') and base_escolhida != "Cochrane":
                    # Se for CSV e não for Cochrane, assumimos que é padrão Scopus
                    df_temp = processar_csv_scopus(f)
                elif ext.endswith(('.xls', '.xlsx')):
                    df_temp = processar_excel_wos(f)
                else:
                    # Fallback padrão para RIS genéricos (WoS, Mendeley, SciELO)
                    df_temp = process_multiple_ris([f], {f.name: base_escolhida})
                
                if df_temp is not None:
                    df_temp['BASE DE DADOS'] = db_mapping[f.name]
                    list_dfs.append(df_temp)

            # 3. Remove a barra de progresso ao finalizar o loop
            pbar_load.empty()

            if list_dfs:
                with st.spinner("Consolidando estrutura final..."):
                    df_raw = padronizar_base_bibliometrica(pd.concat(list_dfs, ignore_index=True))

                    st.session_state['df_original'] = df_raw.copy()
                    st.session_state['df_geral'] = df_raw.copy()
                    resetar_estado_derivado()
                    st.success(f"Sucesso! {len(df_raw)} documentos integrados.")
                    st.rerun()
            else:
                st.error("Não foi possível extrair dados dos arquivos selecionados.")
            
            # --- NOVO: MODO DE DEMONSTRAÇÃO ---
    st.markdown("---")
    st.subheader("🌟 Modo de Demonstração")
    st.caption("Não tem arquivos agora? Explore o Simetrics com dados de exemplo.")
    
    if st.button("🚀 Carregar Arquivos de Exemplo", help="Carrega automaticamente bases pré-configuradas (Scopus, WoS e SciELO) da pasta do projeto."):
        st.session_state['mostrar_descritivo'] = False
       
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
                    df_demo = padronizar_base_bibliometrica(df_demo)
                    # Atualiza o estado da aplicação com os dados de exemplo
                    st.session_state['df_original'] = df_demo.copy()
                    st.session_state['df_geral'] = df_demo.copy()
                    resetar_estado_derivado()
                    st.success("Exemplo carregado com sucesso!")
                    st.rerun()

if st.session_state['df_geral'] is not None:
    df = st.session_state['df_geral']
    
    tab_main, tab_grafos, tab_search, tab_chat, tab_form = st.tabs(["📊 Informações Principais", "🕸️ Redes e Grafos de Conhecimento","🔍 Motor de Busca","🤖 Assistente Científico", "📝 Formulário de Feedback"])
    
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
                        st.session_state['df_duplicados'] = pd.concat([st.session_state['df_duplicados'], df_dupes], ignore_index=True)
                        st.session_state['tabela_sna_completa'] = None
                        st.rerun()

            with c_btn2:
                if st.button("2. Deduplicar por Similaridade"):
                    with st.spinner("Calculando similaridade..."):
                        df_limpo, df_dupes = deduplicar_por_similaridade(st.session_state['df_geral'], threshold)
                        st.session_state['df_geral'] = df_limpo
                        st.session_state['df_duplicados'] = pd.concat([st.session_state['df_duplicados'], df_dupes], ignore_index=True)
                        st.session_state['tabela_sna_completa'] = None
                        st.rerun()

            with c_btn3:
                if st.button("🔄 Reverter Base"):
                    st.session_state['df_geral'] = st.session_state['df_original'].copy()
                    resetar_estado_derivado()
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
            
                st.dataframe(dupes[cols_exibicao], width='stretch', hide_index=True)
            
                # Botão para baixar apenas os excluídos (útil para o anexo da metodologia PRISMA)
                csv_dupes = gerar_csv_bytes(dupes)
                st.download_button("Baixar Relatório de Excluídos (CSV)", data=csv_dupes, file_name='documentos_excluidos.csv')

            st.write("")

        # =========================================================
        # --- NOVO: PANORAMA DE COMPLETUDE DOS METADADOS ---
        # =========================================================
        st.markdown("##### 🗂️ Qualidade e Completude dos Metadados")
        st.caption(f"Análise de dados faltantes em **{len(df)} documentos**. Metadados incompletos podem reduzir a precisão das redes de conhecimento.")
        
        df_comp = analisar_completude_metadados(df)
        
        # Função para aplicar cores estilo "Semáforo" no Pandas
        def colorir_status(val):
            if val == 'Excelente': return 'background-color: #28a745; color: white; font-weight: bold;'
            elif val == 'Bom': return 'background-color: #82e0aa; color: black; font-weight: bold;'
            elif val == 'Aceitável': return 'background-color: #f1c40f; color: black; font-weight: bold;'
            else: return 'background-color: #e74c3c; color: white; font-weight: bold;'
            
        # Aplica o estilo na coluna de Status e formata a porcentagem
        tabela_estilizada = df_comp.style.map(colorir_status, subset=['Status']).format({'Faltantes (%)': "{:.2f}%"})
        
        # Exibe a tabela no Streamlit
        st.dataframe(tabela_estilizada, width='stretch', hide_index=True)
        st.divider()

        # =========================================================
        # --- INTELIGÊNCIA ARTIFICIAL: CATEGORIZAÇÃO TEMÁTICA ---
        # =========================================================
        st.markdown("##### 🤖 Inteligência Artificial: Categorização de Temas")
        
        with st.expander("Identificar Escolas de Pesquisa via Machine Learning", expanded=True):
            api_key = get_gemini_api_key()
            
            if not api_key:
                api_key_valida = False
                st.warning("⚠️ Chave 'GEMINI_API_KEY' não encontrada nas variáveis de ambiente. O mapeamento via IA está temporariamente desativado.")
            else:
                api_key_valida = True
            
            if 'TEMA_GEMINI' in st.session_state['df_geral'].columns:
                st.success("✅ O corpus foi categorizado com sucesso!")
                
                df_resumo = st.session_state['df_geral']['TEMA_GEMINI'].value_counts().reset_index()
                df_resumo.columns = ['Escola Temática (IA)', 'Documentos']
                
                # --- NOVO: Integração do Quociente Locacional (QL) ---
                from utils import obter_top_ql_por_tema
                top_a, top_p, top_v = obter_top_ql_por_tema(st.session_state['df_geral'])
                
                df_resumo['Autor Top QL'] = df_resumo['Escola Temática (IA)'].map(top_a).fillna("-")
                df_resumo['País Top QL'] = df_resumo['Escola Temática (IA)'].map(top_p).fillna("-")
                df_resumo['Venue Top QL'] = df_resumo['Escola Temática (IA)'].map(top_v).fillna("-")
                
                st.markdown("###### 📊 Resumo da Distribuição e Especialização Vocacional (QL)")
                st.dataframe(
                    df_resumo.style.bar(subset=['Documentos'], color='#82c2c2'),
                    width='stretch', 
                    hide_index=True
                )
                
                if st.button("Refazer Categorização (Limpar Temas)"):
                    st.session_state['df_geral'] = st.session_state['df_geral'].drop(columns=['TEMA_GEMINI'])
                    st.rerun()
            
            elif api_key_valida:
                st.info("⚡ O algoritmo Silhouette identificará os grupos e o Gemini nomeará cada escola.")
                # --- NOVO: SLIDER DE CONTROLE DE GRANULARIDADE ---
                st.write("")
                max_temas = st.slider(
                    "Teto Máximo de Temas (Granularidade):", 
                    min_value=3, max_value=20, value=10, step=1, 
                    help="O algoritmo tentará achar o número perfeito de clusters, mas nunca ultrapassará este limite. Use limites altos para nichos específicos ou baixos para macro-áreas."
                )
                
                if st.button("Executar Mapeamento Temático", type="primary"):
                    from utils import categorizar_temas_por_cluster
                    
                    # Passamos o limite escolhido pelo usuário para o motor do utils
                    df_processado = categorizar_temas_por_cluster(
                        st.session_state['df_geral'], 
                        api_key, 
                        max_clusters=max_temas
                    )
                    st.session_state['df_geral'] = df_processado
                    
                    st.toast("Temas gerados com sucesso!", icon="🤖")
                    st.rerun()

        resumo_base = resumir_base_bibliometrica(df)
        total_docs = resumo_base["total_docs"]
        b_metrics = resumo_base["b_metrics"]
        timespan = resumo_base["timespan"]
        avg_age = resumo_base["avg_age"]
        authors_count = resumo_base["authors_count"]
        countries_count = resumo_base["countries_count"]
        kw_count = resumo_base["kw_count"]
        venues_count = resumo_base["venues_count"]

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
                st.dataframe(dt_counts, width='stretch', hide_index=True)
            

        with col_coll:
            st.markdown("##### 🤝 Autoria e Colaboração")
            
            # Dados de Colaboração em Tabela
            coll_data = pd.DataFrame({
                "Métrica": ["Documentos de Autor Único", "Índice de Coautoria", "Publicações Multi-País (MCP)", "Publicações Mono-País (SCP)"],
                "Valor": [b_metrics['single_author_docs'], b_metrics['coauth_index'], b_metrics['mcp'], b_metrics['scp']]
            })
            st.dataframe(coll_data, width='stretch', hide_index=True)
            
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
                st.plotly_chart(fig_year, width='stretch')
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
            st.plotly_chart(fig_db, width='stretch')

        st.divider()

        # =========================================================
        # --- GLOSSÁRIO DE MÉTRICAS ---
        # =========================================================
        st.write("")
        with st.expander("📖 Glossário de Métricas Cientométricas"):
            st.markdown("""
            Estas métricas avaliam não apenas a quantidade de publicações, mas a **consistência** e o **impacto** das entidades na rede de conhecimento:
            
            * **Quantidade de Documentos:** Volume total de produções científicas associadas à entidade.
            * **Total de Citações:** Soma bruta de todas as citações recebidas. Pode ser distorcida por um único artigo "viral".
            * **Média de Citações:** Total de citações dividido pelo número de documentos.
            * **Índice h:** Uma entidade tem índice *h* se *h* dos seus artigos têm pelo menos *h* citações cada. Ex: Um autor com Índice h=5 possui pelo menos 5 artigos com 5 ou mais citações. Mede produtividade e impacto de forma equilibrada.
            * **Índice g:** Dá mais peso aos artigos altamente citados, compensando uma limitação do Índice h. O índice é *g* se a soma das citações dos *g* artigos mais citados for maior ou igual a *g²*.
            * **Índice i10:** Número exato de publicações que possuem 10 ou mais citações (métrica popularizada pelo Google Scholar).
            * **Índice m:** É o Índice h dividido pelo número de anos de atuação (ano atual menos o ano da primeira publicação). Revela a "velocidade" de impacto, permitindo comparar veteranos e pesquisadores iniciantes de forma justa.
            """)
        st.write("")

        col_auth, col_docs = st.columns(2)
        
        # --- TOP 20 AUTORES ---
        with col_auth:
            st.markdown("##### 👥 Top 20 Autores")
            
            # Novo Selectbox com todas as métricas
            metric_author = st.selectbox(
                "Métrica para o Ranking de Autores:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações", "Índice h", "Índice g", "Índice i10", "Índice m"], 
                key="sel_auth_metric"
            )
            
            if 'AUTHORS' in df.columns:
                df_auth_temp = df[['AUTHORS', 'TOTAL CITATIONS', 'YEAR CLEAN']].copy()
                df_auth_temp['TOTAL CITATIONS'] = pd.to_numeric(df_auth_temp['TOTAL CITATIONS'], errors='coerce').fillna(0).astype(int)
                df_auth_temp['YEAR CLEAN'] = pd.to_numeric(df_auth_temp['YEAR CLEAN'], errors='coerce')
                df_auth_temp['Autor'] = df_auth_temp['AUTHORS'].astype(str).str.split(';')
                
                df_auth_expanded = df_auth_temp.explode('Autor')
                df_auth_expanded['Autor'] = df_auth_expanded['Autor'].str.strip()
                df_auth_expanded = df_auth_expanded[(df_auth_expanded['Autor'] != '') & (df_auth_expanded['Autor'] != 'Nan')]
                
                res_list = []
                for autor, group in df_auth_expanded.groupby('Autor'):
                    cits = group['TOTAL CITATIONS'].sort_values(ascending=False).tolist()
                    docs = len(cits)
                    total_cits = sum(cits)
                    
                    h_idx = sum(c >= i + 1 for i, c in enumerate(cits))
                    g_idx = 0
                    soma = 0
                    for i, c in enumerate(cits):
                        soma += c
                        if soma >= (i + 1)**2: g_idx = i + 1
                        else: break
                        
                    anos = group['YEAR CLEAN'].dropna().tolist()
                    m_idx = round(h_idx / ((2026 - min(anos)) + 1), 3) if anos else 0.0
                        
                    res_list.append({
                        'Autor': autor, 'Qtd. de Documentos': docs, 'Total de Citações': total_cits, 
                        'Média de Citações': round(total_cits / docs, 2) if docs > 0 else 0, 
                        'Índice h': h_idx, 'Índice g': g_idx, 'Índice i10': sum(1 for c in cits if c >= 10), 'Índice m': m_idx
                    })
                
                res_auth = pd.DataFrame(res_list)
                top_authors = res_auth.nlargest(20, metric_author)
                
                fig_auth = px.bar(top_authors, x=metric_author, y='Autor', orientation='h', color=metric_author, color_continuous_scale='Blues')
                fig_auth.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_auth, width='stretch')


        with col_docs:
            st.markdown("##### 📄 Top 20 Documentos Mais Citados")
            title_col = next((c for c in ['TITLE', 'TI'] if c in df.columns), None)
            
            # Adicionamos uma verificação: a coluna existe e tem pelo menos um número que não seja nulo?
            if title_col and 'TOTAL CITATIONS' in df.columns and df['TOTAL CITATIONS'].notna().any():
                try:
                    # O nlargest agora funcionará porque garantimos o tipo no utils.py
                    top_d = df.nlargest(20, 'TOTAL CITATIONS').copy()
                    top_d['Título Curto'] = top_d[title_col].apply(lambda x: str(x)[:35] + "..." if len(str(x)) > 35 else x)
                    
                    fig_d = px.bar(top_d, x='TOTAL CITATIONS', y='Título Curto', orientation='h', 
                                    color='TOTAL CITATIONS', color_continuous_scale='Reds')
                    fig_d.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_d, width='stretch')
                except Exception as e:
                    st.warning("Não foi possível gerar o ranking de citações com os dados disponíveis.")
            else:
                st.info("ℹ️ Nenhuma informação de citação encontrada nos documentos para gerar este ranking.")


        # --- LINHA 3 DE GRÁFICOS: TOP PAÍSES E TOP VENUES LADO A LADO ---
        col_pais, col_venue = st.columns(2)
        
        # --- TOP 20 PAÍSES ---
        with col_pais:
            st.markdown("##### 🌍 Top 20 Países Mais Produtivos")
            metric_country = st.selectbox(
                "Métrica de Países:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações", "Índice h", "Índice g", "Índice i10", "Índice m"], 
                key="sel_pais"
            )
            
            if 'COUNTRY' in df.columns:
                df_country_temp = df[['COUNTRY', 'TOTAL CITATIONS', 'YEAR CLEAN']].copy()
                df_country_temp['TOTAL CITATIONS'] = pd.to_numeric(df_country_temp['TOTAL CITATIONS'], errors='coerce').fillna(0).astype(int)
                df_country_temp['YEAR CLEAN'] = pd.to_numeric(df_country_temp['YEAR CLEAN'], errors='coerce')
                df_country_temp['País'] = df_country_temp['COUNTRY'].astype(str).str.split(';')
                
                df_country_expanded = df_country_temp.explode('País')
                df_country_expanded['País'] = df_country_expanded['País'].str.strip()
                df_country_expanded = df_country_expanded[(df_country_expanded['País'] != '') & (df_country_expanded['País'] != 'Nan')]
                
                res_list = []
                for pais, group in df_country_expanded.groupby('País'):
                    cits = group['TOTAL CITATIONS'].sort_values(ascending=False).tolist()
                    docs = len(cits)
                    total_cits = sum(cits)
                    
                    h_idx = sum(c >= i + 1 for i, c in enumerate(cits))
                    g_idx = 0
                    soma = 0
                    for i, c in enumerate(cits):
                        soma += c
                        if soma >= (i + 1)**2: g_idx = i + 1
                        else: break
                        
                    anos = group['YEAR CLEAN'].dropna().tolist()
                    m_idx = round(h_idx / ((2026 - min(anos)) + 1), 3) if anos else 0.0
                        
                    res_list.append({
                        'País': pais, 'Qtd. de Documentos': docs, 'Total de Citações': total_cits, 
                        'Média de Citações': round(total_cits / docs, 2) if docs > 0 else 0, 
                        'Índice h': h_idx, 'Índice g': g_idx, 'Índice i10': sum(1 for c in cits if c >= 10), 'Índice m': m_idx
                    })
                
                res_country = pd.DataFrame(res_list)
                top_c = res_country.nlargest(20, metric_country)
                
                fig_c = px.bar(top_c, x=metric_country, y='País', orientation='h', color=metric_country, color_continuous_scale='Viridis')
                fig_c.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_c, width='stretch')

        # --- TOP 20 VENUES ---
        with col_venue:
            st.markdown("##### 🏢 Top 20 Locais de Publicação (Venues)")
            
            metric_venue = st.selectbox(
                "Métrica de Locais:", 
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações", "Índice h", "Índice g", "Índice i10", "Índice m"], 
                key="sel_venue"
            )
            
            col_venue_name = next((c for c in ['SECONDARY TITLE', 'SO', 'JO'] if c in df.columns), None)
            
            if col_venue_name:
                df_with_venue = df[[col_venue_name, 'TOTAL CITATIONS', 'YEAR CLEAN']].copy()
                df_with_venue['Venue'] = df_with_venue[col_venue_name].astype(str).str.strip()
                df_with_venue['TOTAL CITATIONS'] = pd.to_numeric(df_with_venue['TOTAL CITATIONS'], errors='coerce').fillna(0).astype(int)
                df_with_venue['YEAR CLEAN'] = pd.to_numeric(df_with_venue['YEAR CLEAN'], errors='coerce')
                df_with_venue = df_with_venue[df_with_venue['Venue'] != '']

                if not df_with_venue.empty:
                    res_list = []
                    for venue, group in df_with_venue.groupby('Venue'):
                        cits = group['TOTAL CITATIONS'].sort_values(ascending=False).tolist()
                        docs = len(cits)
                        total_cits = sum(cits)
                        
                        h_idx = sum(c >= i + 1 for i, c in enumerate(cits))
                        g_idx = 0
                        soma = 0
                        for i, c in enumerate(cits):
                            soma += c
                            if soma >= (i + 1)**2: g_idx = i + 1
                            else: break
                            
                        anos = group['YEAR CLEAN'].dropna().tolist()
                        m_idx = round(h_idx / ((2026 - min(anos)) + 1), 3) if anos else 0.0
                            
                        venue_curta = str(venue)[:35] + "..." if len(str(venue)) > 35 else str(venue)
                        
                        res_list.append({
                            'Venue Completa': venue, 'Venue Curta': venue_curta, 'Qtd. de Documentos': docs, 'Total de Citações': total_cits, 
                            'Média de Citações': round(total_cits / docs, 2) if docs > 0 else 0, 
                            'Índice h': h_idx, 'Índice g': g_idx, 'Índice i10': sum(1 for c in cits if c >= 10), 'Índice m': m_idx
                        })
                    
                    res_venue = pd.DataFrame(res_list)
                    top_v = res_venue.nlargest(20, metric_venue)
                    
                    fig_v = px.bar(top_v, x=metric_venue, y='Venue Curta', orientation='h', color=metric_venue, color_continuous_scale='Teal')
                    fig_v.update_layout(yaxis={'categoryorder':'total ascending'})
                    fig_v.update_traces(customdata=top_v['Venue Completa'], hovertemplate='<b>%{customdata}</b><br>Valor: %{x}<extra></extra>')
                    
                    st.plotly_chart(fig_v, width='stretch')
            else:
                st.info("ℹ️ Nenhuma coluna de Local de Publicação (ex: SECONDARY TITLE, SO, JO) foi encontrada nos dados.")


       # --- BLOCO: TOP 20 PALAVRAS-CHAVE ---
        st.markdown("##### ☁️ Top 20 Palavras-chave")
        
        col_kw_opt, col_kw_fig = st.columns([1, 3])

        with col_kw_opt:
            metric_kw = st.selectbox(
                "Métrica de Ranking (Keywords):",
                ["Qtd. de Documentos", "Total de Citações", "Média de Citações", "Índice h", "Índice g", "Índice i10", "Índice m"],
                key="sel_metric_kw_radio" 
            )
            st.info("💡 **Dica:** O gradiente lateral ajuda a identificar visualmente a distância de impacto entre o termo principal e os demais.")

        with col_kw_fig:
            with st.spinner("Mapeando léxico de impacto..."):
                from utils import plot_top_keywords_metric
                fig_kw = plot_top_keywords_metric(df, metric_kw, top_n=20)
                
                if fig_kw:
                    st.plotly_chart(fig_kw, width='stretch')
                else:
                    st.warning("Não foram encontradas palavras-chave válidas.")
       
        # =========================================================
        # --- DISTRIBUIÇÃO ESTATÍSTICA (BOXPLOT) ---
        # =========================================================
        st.divider()
        st.markdown("##### 📊 Distribuição Estatística Comparativa (Boxplot)")
        st.caption("Analise a consistência e descubra *outliers* selecionando até 5 grupos. O Boxplot revela a mediana e a dispersão dos dados, enquanto os pontos mostram cada entidade individualmente.")

        col_box_opt, col_box_fig = st.columns([1, 3])

        with col_box_opt:
            st.write("")
            entidade_box = st.selectbox(
                "Comparar grupos por (Eixo X):", 
                ["Países", "Palavras-chave", "Temas (IA)"],
                key="sel_box_entidade"
            )

            # 1. Prepara e explode os dados baseados na escolha do usuário
            df_box = df.copy()
            coluna_valida = True
            
            if entidade_box == "Países":
                if 'COUNTRY' in df_box.columns:
                    df_box['Item_Box'] = df_box['COUNTRY'].astype(str).str.split(';')
                else: coluna_valida = False
            elif entidade_box == "Palavras-chave":
                col_kw = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df_box.columns), None)
                if col_kw:
                    df_box['Item_Box'] = df_box[col_kw].astype(str).str.split(';')
                else: coluna_valida = False
            elif entidade_box == "Temas (IA)":
                if 'TEMA_GEMINI' in df_box.columns:
                    df_box['Item_Box'] = df_box['TEMA_GEMINI'].apply(lambda x: [x] if pd.notna(x) else [])
                else: coluna_valida = False

            if coluna_valida:
                # Explode as listas para separar os múltiplos países/palavras de um mesmo artigo
                df_box = df_box.explode('Item_Box')
                df_box['Item_Box'] = df_box['Item_Box'].astype(str).str.strip().str.title()
                df_box = df_box[(df_box['Item_Box'] != '') & (df_box['Item_Box'] != 'Nan') & (df_box['Item_Box'] != 'None')]

                # Pega os top 300 mais frequentes para manter a interface rápida
                top_itens_box = df_box['Item_Box'].value_counts().head(300).index.tolist()
                
                itens_selecionados = st.multiselect(
                    "Selecione até 5 itens:", 
                    options=top_itens_box,
                    max_selections=5,
                    key="sel_box_itens",
                    help="A lista exibe os 300 itens mais frequentes da base para comparação."
                )

                # ATUALIZAÇÃO: Selectbox com 6 métricas específicas
                metrica_box = st.selectbox(
                    "Métrica (Eixo Y):", 
                    [
                        "Quantidade de documentos por autor",
                        "Quantidade de documentos por ano",
                        "Quantidade de citações por documento",
                        "Quantidade de citações por autor",
                        "Quantidade de citações por ano"
                    ],
                    key="sel_box_metrica"
                )
                
                # ATUALIZAÇÃO: Opção de escala Logarítmica
                escala_log = st.checkbox("Escala Logarítmica (Eixo Y)", value=False, key="chk_box_log", help="Ideal para lidar com outliers extremos de citações.")
            else:
                st.warning(f"A coluna de {entidade_box} não está disponível nesta base de dados. Faça o mapeamento temático com IA")
                itens_selecionados = []

        with col_box_fig:
            # 2. Lógica de Renderização do Gráfico
            if not coluna_valida:
                st.info("Aguardando dados válidos...")
            elif not itens_selecionados:
                st.info("👈 Selecione de 1 a 5 itens no painel ao lado para gerar o gráfico de distribuição.")
            else:
                with st.spinner("Agrupando dados e calculando dispersão estatística..."):
                    df_plot = df_box[df_box['Item_Box'].isin(itens_selecionados)].copy()
                    
                    # Garante que citações seja numérico caso exista
                    if 'TOTAL CITATIONS' in df_plot.columns:
                        df_plot['TOTAL CITATIONS'] = pd.to_numeric(df_plot['TOTAL CITATIONS'], errors='coerce').fillna(0)
                    else:
                        df_plot['TOTAL CITATIONS'] = 0
                        
                    df_agg = None
                    y_col = None
                    hover_data = None
                    y_label = ""
                    
                    # LÓGICA DE AGRUPAMENTO (6 MÉTRICAS)
                        
                    if metrica_box == "Quantidade de documentos por autor":
                        if 'AUTHORS' in df_plot.columns:
                            df_exp = df_plot.assign(AUTHOR=df_plot['AUTHORS'].astype(str).str.split(';')).explode('AUTHOR')
                            df_exp['AUTHOR'] = df_exp['AUTHOR'].str.strip()
                            df_exp = df_exp[df_exp['AUTHOR'] != '']
                            df_agg = df_exp.groupby(['Item_Box', 'AUTHOR']).size().reset_index(name='Qtd')
                            y_col = 'Qtd'
                            hover_data = ['AUTHOR']
                            y_label = "Qtd. Documentos"
                        else:
                            st.warning("Coluna de Autores não encontrada.")
                            
                    elif metrica_box == "Quantidade de documentos por ano":
                        if 'YEAR CLEAN' in df_plot.columns:
                            df_plot['YEAR CLEAN'] = pd.to_numeric(df_plot['YEAR CLEAN'], errors='coerce')
                            df_plot = df_plot.dropna(subset=['YEAR CLEAN'])
                            df_agg = df_plot.groupby(['Item_Box', 'YEAR CLEAN']).size().reset_index(name='Qtd')
                            y_col = 'Qtd'
                            hover_data = ['YEAR CLEAN']
                            y_label = "Qtd. Documentos"
                        else:
                            st.warning("Coluna de Ano não encontrada.")
                            
                    elif metrica_box == "Quantidade de citações por documento":
                        df_agg = df_plot.copy()
                        y_col = 'TOTAL CITATIONS'
                        hover_data = ['TITLE'] if 'TITLE' in df_agg.columns else None
                        y_label = "Citações por Documento"
                        
                    elif metrica_box == "Quantidade de citações por autor":
                        if 'AUTHORS' in df_plot.columns:
                            df_exp = df_plot.assign(AUTHOR=df_plot['AUTHORS'].astype(str).str.split(';')).explode('AUTHOR')
                            df_exp['AUTHOR'] = df_exp['AUTHOR'].str.strip()
                            df_exp = df_exp[df_exp['AUTHOR'] != '']
                            df_agg = df_exp.groupby(['Item_Box', 'AUTHOR'])['TOTAL CITATIONS'].sum().reset_index(name='Citações')
                            y_col = 'Citações'
                            hover_data = ['AUTHOR']
                            y_label = "Citações Acumuladas"
                        else:
                            st.warning("Coluna de Autores não encontrada.")
                            
                    elif metrica_box == "Quantidade de citações por ano":
                        if 'YEAR CLEAN' in df_plot.columns:
                            df_plot['YEAR CLEAN'] = pd.to_numeric(df_plot['YEAR CLEAN'], errors='coerce')
                            df_plot = df_plot.dropna(subset=['YEAR CLEAN'])
                            df_agg = df_plot.groupby(['Item_Box', 'YEAR CLEAN'])['TOTAL CITATIONS'].sum().reset_index(name='Citações')
                            y_col = 'Citações'
                            hover_data = ['YEAR CLEAN']
                            y_label = "Citações Acumuladas"
                        else:
                            st.warning("Coluna de Ano não encontrada.")

                    # 3. Desenho do Gráfico Plotly
                    if df_agg is not None and not df_agg.empty and y_col:
                        fig_box = px.box(
                            df_agg, 
                            x='Item_Box', 
                            y=y_col, 
                            color='Item_Box',
                            points='all', # Pontos visíveis mostrando cada ocorrência da distribuição
                            hover_data=hover_data,
                            labels={'Item_Box': entidade_box, y_col: y_label},
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            log_y=escala_log # Aplica a escala algorítmica caso ativada
                        )
                        
                        fig_box.update_layout(
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=450,
                            margin=dict(l=20, r=20, t=20, b=20),
                            xaxis=dict(title=None)
                        )
                        st.plotly_chart(fig_box, width='stretch', theme="streamlit")

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
                    st.plotly_chart(fig_circ, width='stretch')
                else:
                    st.warning("Não há dados de países suficientes para formar redes de colaboração.")

        with col_map:
            with st.spinner("Renderizando mapa-múndi interativo..."):
                from utils import plot_map_collaboration
                fig_map = plot_map_collaboration(df, top_n=30)
                if fig_map:
                    st.plotly_chart(fig_map, width='stretch')
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
            df_wc = df

            if fonte_nuvem == "Título + Resumo + Palavras-chave":
                cols_para_unir = [c for c in [c_t, c_k, c_a] if c]
                if cols_para_unir:
                    df_wc = pd.DataFrame({'WC_COMBINADO': df[cols_para_unir].fillna('').agg(' '.join, axis=1)})
                    coluna_escolhida = 'WC_COMBINADO'
                else:
                    coluna_escolhida = None
            else:
                mapa_colunas = {
                    "Títulos": c_t,
                    "Palavras-chave": c_k,
                    "Resumo (Abstract)": c_a
                }
                coluna_escolhida = mapa_colunas.get(fonte_nuvem)
            
            st.info(f"As 'Stopwords' em inglês são removidas automaticamente para destacar termos conceituais.")

        with col_wc_img:
            if coluna_escolhida and coluna_escolhida in df_wc.columns:
                with st.spinner("Pintando palavras e ajustando tipografia..."):
                    
                    wc_opcoes = gerar_nuvem_echarts(
                        df_wc, 
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
        # --- FLUXO DE EVOLUÇÃO TEMÁTICA (SANKEY) ---
        # =========================================================
        st.divider()
        st.markdown("##### 🌊 Fluxo de Evolução Temática (Sankey)")
        st.caption("Acompanhe como as palavras-chave surgem, se conectam e evoluem ao longo de três períodos distintos. As linhas espessas representam a continuidade do termo, enquanto as linhas cruzadas indicam forte co-ocorrência histórica.")

        if 'YEAR CLEAN' in df.columns and df['YEAR CLEAN'].notna().any():
            ano_min = int(df['YEAR CLEAN'].min())
            ano_max = int(df['YEAR CLEAN'].max())

            if ano_max > ano_min:
                # Divide o tempo total em 3 terços para os valores padrão iniciais
                terco = max(1, (ano_max - ano_min) // 3)
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    p1_range = st.slider("Período 1:", min_value=ano_min, max_value=ano_max, value=(ano_min, ano_min + terco - 1), key="sankey_p1")
                with col_p2:
                    p2_range = st.slider("Período 2:", min_value=ano_min, max_value=ano_max, value=(ano_min + terco, ano_min + 2*terco - 1), key="sankey_p2")
                with col_p3:
                    p3_range = st.slider("Período 3:", min_value=ano_min, max_value=ano_max, value=(ano_min + 2*terco, ano_max), key="sankey_p3")

                top_n_sankey = st.slider("Qtd. Palavras-chave Principais por Período:", min_value=3, max_value=25, value=10, step=1, key="sankey_kw_slider")

                with st.spinner("Calculando fluxos e conectando nós temporais..."):
                    from utils import plot_sankey_evolution
                    fig_sankey = plot_sankey_evolution(df, p1_range, p2_range, p3_range, top_n=top_n_sankey)

                    if fig_sankey:
                        st.plotly_chart(fig_sankey, width='stretch')
                    else:
                        st.warning("Não há palavras-chave suficientes nestas faixas de tempo para gerar o fluxo.")
            else:
                st.info("A base de dados atual compreende apenas um único ano. A divisão em períodos evolutivos requer dados de múltiplos anos.")
        else:
            st.warning("A coluna 'Ano' (YEAR CLEAN) é obrigatória para rastrear a evolução temporal.")

        # =========================================================
        # --- NOVO BLOCO: GENÉTICA DAS IDEIAS (EVOLUÇÃO) ---
        # =========================================================
        st.divider()
        st.markdown("##### 🧬 Genética das Ideias")
        st.caption("Análise do ciclo de vida e taxa de replicação das palavras-chave como unidades de conhecimento no ecossistema científico.")

        with st.spinner("Decodificando DNA acadêmico..."):
            from utils import calcular_genetica_palavras
            df_gen = calcular_genetica_palavras(df)

        if df_gen is not None and not df_gen.empty:
            col_gen1, col_gen2 = st.columns([1, 1.5])

            with col_gen1:
                st.markdown("**Fecundidade vs. Mortalidade**")
                st.caption("Termos mortos vs. Sobreviventes.")

                mortos = len(df_gen[df_gen['total_aparicoes'] == 1])
                vivos = len(df_gen[df_gen['total_aparicoes'] > 1])

                fig_mortalidade = go.Figure(data=[go.Pie(
                    labels=['Mortos (1 aparição)', 'Sobreviventes (>1 aparição)'],
                    values=[mortos, vivos],
                    hole=.6,
                    marker_colors=['#E74C3C', '#2ECC71']
                )])
                fig_mortalidade.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=350
                )
                st.plotly_chart(fig_mortalidade, width='stretch')

                with st.expander("👁️ Ver Catálogo Completo", expanded=False):
                    st.write("**🟢 Sobreviventes (Top Fecundidade)**")
                    df_vivos = df_gen[df_gen['total_aparicoes'] > 1].sort_values('total_aparicoes', ascending=False)
                    st.dataframe(df_vivos[['Palavra-chave', 'total_aparicoes']].rename(columns={'total_aparicoes': 'Nº de Aparições'}), width='stretch', hide_index=True, height=250)

                    st.write("**🔴 Mortos (Cemitério de Ideias)**")
                    df_mortos = df_gen[df_gen['total_aparicoes'] == 1]
                    amostra_mortos = df_mortos.sample(min(100, len(df_mortos))) if len(df_mortos) > 0 else df_mortos
                    st.dataframe(amostra_mortos[['Palavra-chave', 'ano_nascimento']].rename(columns={'ano_nascimento': 'Ano'}), width='stretch', hide_index=True, height=250)

            with col_gen2:
                st.markdown("**Tempo de Meia-Vida do Conhecimento (Longevidade)**")
                st.caption("Analisa a 'idade' de sobrevivência. Palavras com vida longa indicam pilares estruturais da pesquisa.")

                min_aparicoes_long = st.slider("Filtrar por nº mínimo de replicações:", min_value=2, max_value=50, value=2, key="slider_longevidade_kw")

                df_long_filtrado = df_gen[df_gen['total_aparicoes'] >= min_aparicoes_long].copy()

                if not df_long_filtrado.empty:
                    fig_longevidade = px.scatter(
                        df_long_filtrado,
                        x="ano_nascimento",
                        y="tempo_vida_anos",
                        size="total_aparicoes",
                        color="ano_extincao",
                        hover_name="Palavra-chave",
                        color_continuous_scale='Plasma',
                        labels={
                            "ano_nascimento": "Ano de Nascimento (1ª Aparição)",
                            "tempo_vida_anos": "Longevidade (Anos de Sobrevivência)",
                            "total_aparicoes": "Total de Réplicas (Fecundidade)",
                            "ano_extincao": "Ano da Última Aparição",
                            "Palavra-chave": "Termo Acadêmico"
                        }
                    )
                    
                    # --- ATUALIZAÇÃO: Contorno nas bolhas para alto contraste no Modo Claro ---
                    fig_longevidade.update_traces(
                        marker=dict(line=dict(width=1, color='rgba(128, 128, 128, 0.7)'))
                    )

                    fig_longevidade.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=450,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    
                    # Força o eixo X a usar inteiros (anos)
                    fig_longevidade.update_xaxes(tickformat="d")
                    
                    st.plotly_chart(fig_longevidade, width='stretch', theme="streamlit")
                else:
                    st.info("Nenhuma palavra-chave encontrada com essa taxa de replicação mínima.")
        
        else:
            st.warning("Dados insuficientes para calcular a genética das ideias (Faltam anos de publicação ou palavras-chave).")

        # =========================================================
        # --- MAPAS CONCEITUAIS 2D E 3D ---
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
                        st.plotly_chart(fig_2d, width='stretch')
                    with c3d:
                        st.markdown("**Projeção Imersiva (3D)**")
                        st.plotly_chart(fig_3d, width='stretch')
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
                    st.plotly_chart(fig_lotka, width='stretch')
                else:
                    st.info("Dados insuficientes para gerar a Lei de Lotka.")
                    
        with col_prod:
            st.caption("O tamanho da bolha indica o volume de documentos publicados no ano, e a cor indica o volume total de citações recebidas. A linha horizontal marca o início e fim da atuação.")
            with st.spinner("Mapeando constância acadêmica..."):
                from utils import plot_author_production_over_time
                fig_prod = plot_author_production_over_time(df, top_n=15)
                if fig_prod:
                    st.plotly_chart(fig_prod, width='stretch')
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
                    st.plotly_chart(fig_hist, width='stretch')
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
                        st.plotly_chart(fig_mapa, width='stretch')
                    else:
                        st.warning("Volume de texto insuficiente ou sem padrões de co-ocorrência claros para gerar os quadrantes.")
            else:
                st.warning(f"Coluna de {fonte_mapa} não encontrada na base de dados.")

        # =========================================================
        # MÓDULO DE TABELAS, ESTATÍSTICAS E EXPORTAÇÕES
        # =========================================================            
        st.markdown("### 📋 Tabelas Analíticas e Exportações")
        st.caption("Investigue os dados nas abas abaixo ou baixe todo o dossiê consolidado e os gráficos gerados nesta página.")
        
        df = st.session_state['df_geral']
        
        with st.spinner("Compilando dados estatísticos para exportação..."):
            from utils import gerar_tabela_autores, gerar_tabela_paises, gerar_tabela_venues, gerar_tabela_keywords
            import io
            import zipfile

            # 1. Base Geral formatada
            df_geral_export = df.copy()
            cols_geral = [c for c in df_geral_export.columns if c != 'YEAR CLEAN']
            for col_prioritaria in reversed(['TEMA_GEMINI', 'TOTAL CITATIONS', 'TITLE', 'AUTHORS']):
                if col_prioritaria in cols_geral:
                    cols_geral.insert(0, cols_geral.pop(cols_geral.index(col_prioritaria)))
            df_geral_export = df_geral_export[cols_geral]

            # 2. Sub-bases
            df_autores = gerar_tabela_autores(df)
            df_paises = gerar_tabela_paises(df)
            df_venues = gerar_tabela_venues(df)
            df_keywords = gerar_tabela_keywords(df)

            # 3. Criando o Excel Multi-abas em memória
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_geral_export.to_excel(writer, index=False, sheet_name='Base Geral')
                if not df_autores.empty: df_autores.to_excel(writer, index=False, sheet_name='Autores')
                if not df_paises.empty: df_paises.to_excel(writer, index=False, sheet_name='Países')
                if not df_venues.empty: df_venues.to_excel(writer, index=False, sheet_name='Venues')
                if not df_keywords.empty: df_keywords.to_excel(writer, index=False, sheet_name='Palavras-chave')
            excel_data = excel_buffer.getvalue()

        # 4. Captura Automática dos Gráficos Gerados Acima (Varredura de Memória)
        # 4. Captura Automática dos Gráficos Gerados Acima
        graficos_gerados = {}
        for nome_var, nome_arq in [
            ('fig_year', '01_Dinamica_Tempo'), ('fig_db', '02_Distribuicao_Bases'),
            ('fig_auth', '03_Top_Autores'), ('fig_d', '04_Top_Documentos'),
            ('fig_c', '05_Top_Paises'), ('fig_v', '06_Top_Venues'),
            ('fig_kw', '07_Top_Keywords'), ('fig_box', '08_Boxplot'),
            ('fig_circ', '09_Colaboracao_Circular'), ('fig_map', '10_Colaboracao_Mapa'),
            ('fig_sankey', '11_Sankey'), ('fig_mortalidade', '12_Fecundidade'),
            ('fig_longevidade', '13_Longevidade'), ('fig_2d', '14_PCA_2D'),
            ('fig_3d', '15_PCA_3D'), ('fig_lotka', '16_Lei_Lotka'),
            ('fig_prod', '17_Producao_Autores'), ('fig_hist', '18_Historiograph'),
            ('fig_mapa', '19_Mapa_Tematico')
        ]:
            if nome_var in locals() and locals()[nome_var] is not None:
                graficos_gerados[nome_arq] = locals()[nome_var]

        # --- FUNÇÃO DE ZIP COM AJUSTE DE MARGENS (FIX) ---
        def gerar_zip_graficos():
            zip_buffer = io.BytesIO()
            sucessos = 0
            erros_log = []
            
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for nome, fig in graficos_gerados.items():
                    try:
                        if hasattr(fig, 'to_image'):
                            # --- AJUSTE DE LAYOUT PARA EXPORTAÇÃO ---
                            # Criamos uma cópia temporária para não estragar o visual da tela
                            fig_export = go.Figure(fig)
                            
                            fig_export.update_layout(
                                # 1. Força o cálculo automático de espaço para os nomes
                                yaxis=dict(automargin=True),
                                # 2. Adiciona um recuo extra na esquerda (l) para segurança
                                margin=dict(l=150, r=50, t=80, b=50),
                                # 3. Garante que o fundo seja branco para relatórios
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            
                            # 4. Aumentamos a largura (width) para 1600 para dar mais "ar" aos textos
                            img_bytes = fig_export.to_image(
                                format="png", 
                                width=1600, 
                                height=1000, 
                                scale=2
                            )
                            
                            zip_file.writestr(f"{nome}.png", img_bytes)
                            sucessos += 1
                    except Exception as e:
                        erros_log.append(f"{nome}: {str(e)}")
                        
            return zip_buffer.getvalue(), sucessos, erros_log

        # 5. Painel de Exportação Master
        st.write("")
        col_exp_1, col_exp_2 = st.columns(2)

        with col_exp_1:
            st.download_button(
                label="📥 Baixar Dossiê Completo (Excel Multi-abas)",
                data=excel_data,
                file_name="simetrics_dossie_completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )

        with col_exp_2:
            # --- NOVO BOTÃO DE IMAGENS COM TRATAMENTO DE ERROS ---
            if "zip_graficos" not in st.session_state:
                if st.button("🖼️ Compilar Imagens dos Gráficos (ZIP)", use_container_width=True):
                    with st.spinner("Renderizando PNGs em alta resolução (Isso leva de 5 a 15 segundos)..."):
                        zip_data, qtd_sucesso, lista_erros = gerar_zip_graficos()
                        
                        if qtd_sucesso > 0:
                            st.session_state["zip_graficos"] = zip_data
                            if len(lista_erros) > 0:
                                st.warning(f"⚠️ {qtd_sucesso} imagens geradas. Algumas falharam: {lista_erros[0]}")
                            st.rerun() # Atualiza a tela para mostrar o botão de download
                        else:
                            st.error("❌ Falha crítica ao gerar as imagens.")
                            st.info(f"Detalhe do erro: {lista_erros[0] if lista_erros else 'Nenhum gráfico Plotly compatível encontrado na tela.'}")
            else:
                st.download_button(
                    label="⬇️ Clique aqui para Baixar as Imagens (ZIP)",
                    data=st.session_state["zip_graficos"],
                    file_name="simetrics_graficos_imagens.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )

        st.caption("ℹ️ *A Nuvem de Palavras e as Redes SNA Dinâmicas não são exportadas pelo botão ZIP pois rodam ativamente no navegador (JS). Você pode baixá-las clicando com o botão direito diretamente sobre elas.*")
        st.write("")
        
        # 6. Renderização Limpa das Abas
        aba_geral, aba_autores, aba_paises, aba_venues, aba_keywords = st.tabs([
            "📚 Tabela Geral", "✍️ Autores", "🌍 Países", "🏛️ Fontes (Venues)", "🔑 Palavras-chave"
        ])
        
        with aba_geral:
            st.dataframe(df_geral_export, width='stretch')
        with aba_autores:
            if not df_autores.empty: st.dataframe(df_autores, width='stretch', hide_index=True)
            else: st.info("Não há dados suficientes de autores para gerar esta tabela.")
        with aba_paises:
            if not df_paises.empty: st.dataframe(df_paises, width='stretch', hide_index=True)
            else: st.info("Não há dados suficientes de países (COUNTRY) para gerar esta tabela.")
        with aba_venues:
            if not df_venues.empty: st.dataframe(df_venues, width='stretch', hide_index=True)
            else: st.info("Não há dados suficientes de fontes de publicação para gerar esta tabela.")
        with aba_keywords:
            if not df_keywords.empty: st.dataframe(df_keywords, width='stretch', hide_index=True)
            else: st.info("Não há palavras-chave (KEYWORDS) suficientes para gerar esta tabela.")

    # === ABA 2: REDES E GRAFOS ===
    with tab_grafos:
        st.subheader("Mapeamento do Ecossistema de Conhecimento (Em breve...)")
        
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
                for col in ['REFERENCES_UNIFIED', 'REFERENCES', 'CITED REFERENCES', 'CR']:
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
        st.markdown("### 📊 Ecologia Profunda e Tabela de Nós SNA (Rede Heterogênea)")
        st.caption("Esta área integra Autores, Documentos, Países e Venues, extraindo tanto a saúde topológica global da rede quanto a influência de cada nó individualmente.")
        
        # Cria um espaço na memória da sessão para guardar os dados gerados
        if "tabela_sna_completa" not in st.session_state:
            st.session_state["tabela_sna_completa"] = None
            st.session_state["metricas_globais_sna"] = None

        # Botão de Ação
        if st.button("🚀 Iniciar Cálculo da Rede SNA Completa", type="primary"):
            pbar_sna = st.progress(0, text="Calculando centralidades de todo o ecossistema...")
            
            # Executa o cálculo e guarda as duas variáveis
            df_sna, globais_sna = gerar_tabela_metricas_completas(df, _pbar=pbar_sna)
            st.session_state["tabela_sna_completa"] = df_sna
            st.session_state["metricas_globais_sna"] = globais_sna
            
            pbar_sna.empty() # Remove a barra ao terminar

        # Renderização do Painel Global e Tabela (Ocorre se os dados existirem na memória)
        if st.session_state["tabela_sna_completa"] is not None:
            df_metricas = st.session_state["tabela_sna_completa"]
            globais = st.session_state["metricas_globais_sna"]
            
            # --- RENDERIZAÇÃO DO PAINEL DASHBOARD GLOBAL ---
            if globais:
                st.markdown("#### 🧬 Métricas de Redes Complexas")
                g1, g2, g3, g4 = st.columns(4)
                with g1: create_kpi_card("Densidade da Rede", f"{globais['densidade']:.5f}")
                with g2: create_kpi_card("Eficiência Global", f"{globais['eficiencia']:.4f}" if isinstance(globais['eficiencia'], float) else globais['eficiencia'])
                with g3: create_kpi_card("Entropia (H)", f"{globais['entropia']:.2f} bits")
                with g4: create_kpi_card("Clustering Médio", f"{globais['clustering']:.4f}")
                
                with st.expander("📊 Estatísticas de Conectividade & Influência (Médias)"):
                    st.markdown("**Conectividade (Links por Nó)**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Média de Links", f"{globais['media_links']:.2f}")
                    c2.metric("Desvio Padrão", f"{globais['std_links']:.2f}")
                    c3.metric("Mínimo", globais['min_links'])
                    c4.metric("Máximo", globais['max_links'])
                    
                    st.markdown("**Influência Estrutural**")
                    i1, i2, i3, i4 = st.columns(4)
                    i1.metric("PageRank Médio", f"{globais['mean_pr']:.6f}")
                    i2.metric("Eigenvector Médio", f"{globais['mean_eig']:.6f}")
                    i3.metric("Restrição (Burt)", globais['restricao'])
                    i4.metric("Redundância", globais['redundancia'])

                st.markdown("#### 🧬 Métricas de Ecologia Profunda (SNA Avançado)")
                e1, e2, e3, e4 = st.columns(4)
                
                # Regras de negócio visuais do Simetrics
                status_lei = "Saudável" if 2 < globais['lei_potencia'] < 3 else "Atípico"
                with e1: create_kpi_card("Lei de Potência (γ)", f"{globais['lei_potencia']:.2f}", subtitle=f"↑ {status_lei}")
                
                status_spearman = "Inovação (Brokers)" if globais['spearman'] < 0.8 else "Hierárquica"
                with e2: create_kpi_card("Correlação Spearman (ρ)", f"{globais['spearman']:.2f}", subtitle=f"↑ {status_spearman}")
                
                status_assort = "Expansiva" if globais['assortatividade'] < 0 else "Fechada (Endógena)"
                with e3: create_kpi_card("Assortatividade (r)", f"{globais['assortatividade']:.2f}", subtitle=f"↑ {status_assort}")
                
                with e4: create_kpi_card("Coeficiente Rich-Club (Φ)", globais['rich_club'], subtitle="↑ Hubs Isolados")

                # Glossário
                with st.expander("📖 Glossário de Métricas SNA"):
                    st.markdown("""
                    * **Densidade:** Proporção de conexões reais em relação ao total possível. Redes acadêmicas costumam ser naturalmente esparsas (baixa densidade).
                    * **Eficiência Global:** Quão rápido a informação viaja pela rede. Valores altos indicam forte conectividade e fluidez de ideias entre diferentes campos.
                    * **Entropia (H):** Grau de desordem estrutural da rede. Redes com alta entropia são mais orgânicas e resilientes à remoção de nós críticos.
                    * **Clustering Médio:** Tendência de os vizinhos de um autor também estarem conectados entre si (formação de "panelinhas" ou colegiados isolados).
                    * **Lei de Potência (γ):** Define se a rede é *Scale-Free* (muitos pesquisadores com poucos links orbitando poucos gigantes). Valores entre 2 e 3 indicam redes naturais saudáveis.
                    * **Correlação Spearman (ρ):** Mede a dependência entre quem tem mais links e quem atua como ponte de conhecimento. Valores baixos indicam a presença forte de *Brokers* (inovadores transversais).
                    * **Assortatividade (r):** Se pesquisadores de grande renome preferem colaborar apenas com outros gigantes (r > 0, fechada) ou se eles nutrem autores menores (r < 0, expansiva).
                    """)
                
                st.write("")

            # --- RENDERIZAÇÃO DA TABELA ---
            if not df_metricas.empty:
                st.markdown("#### 📄 Tabela de Métricas Individuais")
                tipos_disponiveis = ["Todos"] + sorted(df_metricas["Tipo"].unique().tolist())
                filtro_tipo = st.selectbox("Filtrar por Tipo de Item:", tipos_disponiveis)

                df_filtrado = df_metricas if filtro_tipo == "Todos" else df_metricas[df_metricas["Tipo"] == filtro_tipo]

                st.dataframe(
                    df_filtrado,
                    width='stretch',
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
                    data=gerar_csv_bytes(df_filtrado),
                    file_name="metricas_sna_ecossistema.csv"
                )
            else:
                st.warning("Não há dados suficientes para calcular a rede heterogênea.")
        else:
            st.info("A topologia global está em modo de espera. Clique no botão azul acima para extrair o DNA da sua rede.")

        
    # Inicializa o estado de busca se não existir
    if 'busca_tipo_biblio' not in st.session_state:
        st.session_state['busca_tipo_biblio'] = "Documento"
    if 'busca_termo_biblio' not in st.session_state:
        st.session_state['busca_termo_biblio'] = None

    # =========================================================
    # --- ABA DO MOTOR DE BUSCA ---
    # =========================================================
    
    # =========================================================
    # --- ABA DO MOTOR DE BUSCA ---
    # =========================================================
    with tab_search:
        st.header("🔍 Motor de Busca e Dossiê Científico")
        st.caption("Investigue as entidades do ecossistema e descubra sua influência topológica na rede.")

        info_busca = preparar_opcoes_busca(df)
        col_titulos = info_busca["col_titulos"]
        col_autores = info_busca["col_autores"]
        col_paises = info_busca["col_paises"]
        col_venue = info_busca["col_venue"]
        col_ano = info_busca["col_ano"]
        
        opcoes_doc = info_busca["opcoes_doc"]
        opcoes_aut = info_busca["opcoes_aut"]
        opcoes_pais = info_busca["opcoes_pais"]
        opcoes_venue = info_busca["opcoes_venue"]
        
        # Identifica e prepara as opções para Palavras-chave
        col_kw = next((c for c in ['KEYWORDS', 'KW', 'DE'] if c in df.columns), None)
        opcoes_kw = []
        if col_kw:
            kw_series = df[col_kw].dropna().astype(str).str.split(';').explode().str.strip().str.title()
            opcoes_kw = sorted(list(set(kw_series[kw_series != ''])))

        # 2. Interface de Busca
        opcoes_busca = ["Documento", "Autor", "País", "Local de Publicação (Venue)", "Palavra-chave", "Tema"] 
        
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
        elif st.session_state['busca_tipo_biblio'] == "Tema": opcoes_lista = info_busca.get("opcoes_tema", [])
        elif st.session_state['busca_tipo_biblio'] == "Palavra-chave": opcoes_lista = opcoes_kw

        # Verificação de Segurança para IA (Temas) e Palavras-chave
        if tipo_busca == "Tema" and 'TEMA_GEMINI' not in df.columns:
            st.warning("⚠️ **Nenhuma escola temática foi identificada nesta base de dados.**")
            st.info("Vá até a aba **📊 Informações Principais** e execute o mapeamento via Inteligência Artificial.")
            termo_selecionado = None
        elif tipo_busca == "Palavra-chave" and not col_kw:
            st.warning("⚠️ **Nenhuma coluna de palavras-chave foi encontrada nesta base de dados.**")
            termo_selecionado = None
        else:
            termo_selecionado = st.selectbox(
                "Selecione ou digite para pesquisar:",
                opcoes_lista,
                index=opcoes_lista.index(st.session_state['busca_termo_biblio']) if st.session_state['busca_termo_biblio'] in opcoes_lista else None,
                placeholder="Explore o ecossistema..."
            )

        if termo_selecionado != st.session_state['busca_termo_biblio']:
            st.session_state['busca_termo_biblio'] = termo_selecionado
            st.rerun()

        termo_ativo = st.session_state['busca_termo_biblio']
        tipo_ativo = st.session_state['busca_tipo_biblio']

        # --- FUNÇÃO: Obter Subset Seguro (Incluindo Keywords) ---
        def obter_docs_subset(df_global, termo, tipo):
            if tipo == "Palavra-chave" and col_kw:
                def match_kw(x):
                    if pd.isna(x): return False
                    kws = [k.strip().title() for k in str(x).split(';')]
                    return termo in kws
                return df_global[df_global[col_kw].apply(match_kw)]
            else:
                return filtrar_por_entidade(df_global, termo, tipo)

        # --- FUNÇÃO: Motor de Índices Cientométricos ---
        def calcular_metricas_perfil(docs_subset):
            if 'TOTAL CITATIONS' not in docs_subset.columns: return 0, 0, 0, 0, 0, 0.0

            cits_raw = pd.to_numeric(docs_subset['TOTAL CITATIONS'], errors='coerce').fillna(0).astype(int)
            cits_sorted = cits_raw.sort_values(ascending=False).tolist()
            
            total_docs = len(cits_sorted)
            total_citacoes = sum(cits_sorted)
            h_idx = sum(c >= i + 1 for i, c in enumerate(cits_sorted))
            
            g_idx, soma = 0, 0
            for i, c in enumerate(cits_sorted):
                soma += c
                if soma >= (i + 1)**2: g_idx = i + 1
                else: break
                    
            i10_idx = sum(1 for c in cits_sorted if c >= 10)
            
            m_idx = 0.0
            if 'YEAR CLEAN' in docs_subset.columns:
                anos_validos = pd.to_numeric(docs_subset['YEAR CLEAN'], errors='coerce').dropna().astype(int).tolist()
                if anos_validos:
                    m_idx = round(h_idx / ((2026 - min(anos_validos)) + 1), 3)
                        
            return total_docs, total_citacoes, h_idx, g_idx, i10_idx, m_idx

        # --- FUNÇÃO: Renderizar Botões de Palavras-chave ---
        def mostrar_botoes_keywords(docs_subset):
            if not col_kw: return
            kws = docs_subset[col_kw].dropna().astype(str).str.split(';').explode().str.strip().str.title()
            kws = kws[kws != '']
            top_kws = kws.value_counts().head(12) # Exibe as 12 principais para não poluir
            
            if not top_kws.empty:
                st.write("**🔑 Principais Palavras-chave (clique para ver perfil):**")
                for kw, count in top_kws.items():
                    # Evita exibir a própria palavra-chave se já estiver no perfil dela
                    if tipo_ativo == "Palavra-chave" and kw == termo_ativo: continue
                    st.button(f"🔑 {kw} ({count})", key=f"btn_kw_{hash(kw)}_{tipo_ativo}", on_click=navegar_busca, args=("Palavra-chave", kw))

        # --- FUNÇÃO: Tabela Universal de Associações (Ecossistema da Entidade) ---
        def gerar_tabela_associacoes(docs_subset, df_global, tipo_entidade_atual):
            st.markdown(f"**🕸️ Ecossistema Associado (Especialização e Ocorrências):**")
            st.caption("Esta tabela revela quais outras entidades orbitam o perfil atual e qual o grau de especialização (QL) desta relação.")
            
            Q = df_global['TITLE'].nunique() if 'TITLE' in df_global.columns else len(df_global)
            Qk = docs_subset['TITLE'].nunique() if 'TITLE' in docs_subset.columns else len(docs_subset)
            
            if Q == 0 or Qk == 0: return
            
            associacoes = []
            
            def processar_coluna(col_name, tipo_nome):
                if col_name not in df_global.columns: return
                if tipo_nome == tipo_entidade_atual: return # Evita cruzar a entidade com o próprio tipo
                
                # Base Global
                df_g = df_global[['TITLE', col_name]].copy()
                df_g[col_name] = df_g[col_name].astype(str).str.split(';')
                df_g = df_g.explode(col_name)
                df_g[col_name] = df_g[col_name].str.strip().str.title() if tipo_nome != 'Local de Publicação (Venue)' else df_g[col_name].str.strip().str.upper()
                df_g = df_g[(df_g[col_name] != '') & (df_g[col_name] != 'Nan') & (df_g[col_name].notna())]
                df_g = df_g.drop_duplicates(subset=['TITLE', col_name])
                Qi_s = df_g[col_name].value_counts()
                
                # Subset Ativo
                df_s = docs_subset[['TITLE', col_name]].copy()
                df_s[col_name] = df_s[col_name].astype(str).str.split(';')
                df_s = df_s.explode(col_name)
                df_s[col_name] = df_s[col_name].str.strip().str.title() if tipo_nome != 'Local de Publicação (Venue)' else df_s[col_name].str.strip().str.upper()
                df_s = df_s[(df_s[col_name] != '') & (df_s[col_name] != 'Nan') & (df_s[col_name].notna())]
                df_s = df_s.drop_duplicates(subset=['TITLE', col_name])
                Qik_s = df_s[col_name].value_counts()
                
                for item, q_ik in Qik_s.items():
                    q_i = Qi_s.get(item, 0)
                    if q_i > 0:
                        ql = (q_ik / Qk) / (q_i / Q)
                        associacoes.append({
                            "Nome da Entidade": item,
                            "Tipo": tipo_nome,
                            "Ocorrências": q_ik,
                            "QL (Especialização)": round(ql, 2)
                        })
            
            # Processa todas as colunas disponíveis para cruzar os dados
            if col_kw: processar_coluna(col_kw, "Palavra-chave")
            if col_venue: processar_coluna(col_venue, "Local de Publicação (Venue)")
            if 'TEMA_GEMINI' in df_global.columns: processar_coluna('TEMA_GEMINI', "Tema")
            if col_autores: processar_coluna(col_autores, "Autor")
            if col_paises: processar_coluna(col_paises, "País")
            
            if associacoes:
                df_assoc = pd.DataFrame(associacoes)
                # Ordena primeiro pelo QL mais alto, depois pelas ocorrências
                df_assoc = df_assoc.sort_values(by=["QL (Especialização)", "Ocorrências"], ascending=[False, False]).head(50)
                
                st.dataframe(
                    df_assoc.style.background_gradient(subset=["QL (Especialização)"], cmap="Purples"),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("Nenhuma associação forte o suficiente encontrada para gerar o ecossistema.")

        # 3. Construção do Perfil e Métricas SNA
        if termo_ativo:
            subset_df = obter_docs_subset(df, termo_ativo, tipo_ativo)
            col_info, col_sna = st.columns([2, 1])
            
            grafo_global = obter_grafo_global_busca(df, col_titulos, col_autores, col_paises, col_venue)
            
            def calcular_sna_instantaneo(G, termo):
                if termo not in G: return None
                grau_abs = G.degree(termo)
                total_nos = len(G)
                cent_grau = grau_abs / (total_nos - 1) if total_nos > 1 else 0
                ego_net = nx.ego_graph(G, termo, radius=1) 
                betw = nx.betweenness_centrality(ego_net).get(termo, 0)
                clos = nx.closeness_centrality(ego_net).get(termo, 0)
                return {"Grau Absoluto": grau_abs, "Centralidade Grau": cent_grau, "Betweenness": betw, "Closeness": clos}

            with col_sna:
                sna_local = calcular_sna_instantaneo(grafo_global, termo_ativo)
                if sna_local:
                    st.markdown("##### Métricas Locais SNA")
                    st.metric("Grau Absoluto", sna_local["Grau Absoluto"])
                    st.metric("Centralidade Grau", f"{sna_local['Centralidade Grau']:.4f}")
                    st.metric("Betweenness", f"{sna_local['Betweenness']:.4f}")
                    st.metric("Closeness", f"{sna_local['Closeness']:.4f}")

            # --- RENDERIZAÇÃO DO PERFIL (COLUNA ESQUERDA) ---
            with col_info:
                st.info(f"**{tipo_ativo}:** {termo_ativo}")
                
                if tipo_ativo == "Documento":
                    doc = subset_df.iloc[0]
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
                    
                    mostrar_botoes_keywords(subset_df)
                    
                    col_abstract = next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)
                    if col_abstract and pd.notna(doc[col_abstract]):
                        with st.expander("Ler Resumo (Abstract)"):
                            st.write(doc[col_abstract])
                            
                elif tipo_ativo in ["Autor", "País", "Local de Publicação (Venue)", "Tema", "Palavra-chave"]:
                    docs, cits, h, g, i10, m = calcular_metricas_perfil(subset_df)
                    st.write(f"**Impacto Total:** {cits} citações em {docs} documentos")
                    st.markdown(f"**Índices:** `h: {h}` | `g: {g}` | `i10: {i10}` | `m: {m}`")
                    
                    mostrar_botoes_keywords(subset_df)
                    
                    # Exibe a Nova Tabela de Associações Universais
                    gerar_tabela_associacoes(subset_df, df, tipo_ativo)

                    with st.expander(f"📚 Ver Documentos ({len(subset_df)})"):
                        for i, (_, r) in enumerate(subset_df.iterrows()):
                            titulo_doc = r[col_titulos]
                            st.button(f"📄 {titulo_doc} ({r.get(col_ano, 'N/A')})", key=f"btn_nav_doc_{tipo_ativo}_{hash(titulo_doc)}_{i}", on_click=navegar_busca, args=("Documento", titulo_doc))

            # =========================================================
            # ABAS DO DOSSIÊ (HISTÓRICO, NUVEM E SIMILARES)
            # =========================================================
            tab_hist, tab_nuvem, tab_similares = st.tabs(["📈 Evolução Histórica", "☁️ Lexicometria", "🔗 Itens Semelhantes"])

            # --- ABA 1: HISTÓRICO AVANÇADO ---
            with tab_hist:
                st.markdown(f"**Produção e Impacto ao Longo do Tempo**")
                nomes_comuns_tipo = ['TYPE', 'DT', 'DOCUMENT TYPE', 'TY', 'TIPO', 'TIPO DE DOCUMENTO', 'TYPE OF REFERENCE', 'REFERENCE TYPE']
                col_tipo_doc = next((c for c in df.columns if str(c).strip().upper() in nomes_comuns_tipo), None)
                
                opcoes_visao = ["Visão Geral"]
                if col_tipo_doc: opcoes_visao.append("Separado por Tipo de Documento")
                if 'TEMA_GEMINI' in df.columns: opcoes_visao.append("Separado por Temas") 

                visao_hist = st.radio("Análise Histórica:", opcoes_visao, horizontal=True, key=f"rad_hist_{hash(termo_ativo)}")
                
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
                            st.plotly_chart(fig_hist, width='stretch')
                        elif visao_hist == "Separado por Tipo de Documento":
                            df_ano[col_tipo_doc] = df_ano[col_tipo_doc].fillna("Desconhecido")
                            hist_data = df_ano.groupby([col_ano, col_tipo_doc]).size().reset_index(name='Volume')
                            hist_data['Volume'] = pd.to_numeric(hist_data['Volume'], errors='coerce').fillna(0).astype(int)
                            fig_hist = px.bar(hist_data, x=col_ano, y='Volume', color=col_tipo_doc, title="Volume de Documentos por Tipo e Ano", template="plotly_white")
                            fig_hist.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                            st.plotly_chart(fig_hist, width='stretch')
                        elif visao_hist == "Separado por Temas":
                            df_ano['TEMA_GEMINI'] = df_ano['TEMA_GEMINI'].fillna("Não Categorizado")
                            hist_data = df_ano.groupby([col_ano, 'TEMA_GEMINI']).size().reset_index(name='Volume')
                            hist_data['Volume'] = pd.to_numeric(hist_data['Volume'], errors='coerce').fillna(0).astype(int)
                            fig_hist = px.bar(hist_data, x=col_ano, y='Volume', color='TEMA_GEMINI', title=f"Evolução Temporal: {termo_ativo}", template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_hist.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                            st.plotly_chart(fig_hist, width='stretch')
                    else: st.info("Não há dados temporais válidos para gerar o histórico.")
                else: st.info("A coluna de Ano não está disponível para esta análise.")

            # --- ABA 2: NUVEM DE PALAVRAS CUSTOMIZÁVEL ---
            with tab_nuvem:
                st.markdown(f"**Assinatura Semântica do Perfil**")
                c1, c2, c3 = st.columns(3)
                with c1: fonte_txt = st.selectbox("Composição do Texto:", ["Tudo Combinado", "Apenas Títulos", "Apenas Palavras-chave", "Apenas Resumo"], key=f"src_wc_{hash(termo_ativo)}")
                with c2: estilo_txt = st.selectbox("Tipografia:", ["Arial", "Verdana", "Courier New", "Georgia", "Impact", "Trebuchet MS"], key=f"font_wc_{hash(termo_ativo)}")
                with c3: tema_cor = st.selectbox("Paleta:", ["Oceano", "Fogo", "Floresta", "Cyberpunk", "Acadêmico"], index=4, key=f"pal_wc_{hash(termo_ativo)}")
                    
                paletas_dict = {
                    "Oceano": ["#0077b6", "#00b4d8", "#90e0ef", "#03045e", "#023e8a"], "Fogo": ["#ff4d00", "#ff8c00", "#ff0000", "#fad02c", "#e85d04"],
                    "Floresta": ["#2d6a4f", "#40916c", "#1b4332", "#74c69d", "#95d5b2"], "Cyberpunk": ["#f72585", "#7209b7", "#3a0ca3", "#4361ee", "#4cc9f0"],
                    "Acadêmico": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
                }

                textos_para_juntar = []
                col_ab = next((c for c in ['ABSTRACT', 'AB'] if c in df.columns), None)
                
                if ("Tudo" in fonte_txt or "Título" in fonte_txt) and col_titulos in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_titulos].dropna().astype(str)))
                if ("Tudo" in fonte_txt or "Palavras-chave" in fonte_txt) and col_kw in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_kw].dropna().astype(str).str.replace(';', ' ')))
                if ("Tudo" in fonte_txt or "Resumo" in fonte_txt) and col_ab in subset_df.columns:
                    textos_para_juntar.append(" ".join(subset_df[col_ab].dropna().astype(str)))

                texto_final_vetorizado = " ".join(textos_para_juntar)
                subset_df_wc = pd.DataFrame({'TEXTO_COMBINADO': [texto_final_vetorizado]})

                with st.spinner("Gerando nuvem de palavras específica..."):
                    wc_opcoes = gerar_nuvem_echarts(subset_df_wc, coluna='TEXTO_COMBINADO', fonte=estilo_txt, paleta=paletas_dict[tema_cor])
                    if wc_opcoes: st_echarts(options=wc_opcoes, height="450px", key=f"wc_{hash(termo_ativo)}_{fonte_txt}_{tema_cor}")
                    else: st.warning("Texto insuficiente nos documentos desta entidade para gerar a nuvem semântica.")

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
                    st.dataframe(df_sim, hide_index=True, width='stretch', column_config={"Similaridade (%)": st.column_config.ProgressColumn("Similaridade (%)", min_value=0, max_value=100, format="%.1f%%")})
                    with st.expander(f"Navegar para os perfis ({titulo_coluna_item})"):
                        for idx, row in df_sim.iterrows():
                            item_nome = row[titulo_coluna_item]
                            st.button(f"Ir para: {item_nome}", key=f"btn_sim_{tipo_nav}_{hash(item_nome)}_{idx}", on_click=navegar_busca, args=(tipo_nav, item_nome))

                if not similares or all(len(v) == 0 for v in similares.values()):
                    st.warning("Este item possui conexões muito isoladas para que vizinhos próximos sejam calculados com precisão.")
                else:
                    if tipo_ativo == 'Documento':
                        st.markdown("##### 📄 Documentos Semelhantes")
                        render_tabela_similares(similares.get('Documentos', []), "Documento", "Documento")
                    elif tipo_ativo == 'Autor':
                        st.markdown("##### ✍️ Autores com Perfil Semelhante")
                        render_tabela_similares(similares.get('Autores', []), "Autor", "Autor")
                    elif tipo_ativo in ['País', 'Local de Publicação (Venue)', 'Palavra-chave', 'Tema']:
                        st.markdown(f"##### 🔗 Entidades Semelhantes ({tipo_ativo})")
                        render_tabela_similares(similares.get('Itens', []), tipo_ativo, tipo_ativo)
  
    # =========================================================
    # --- ABA 4: ASSISTENTE CIENTÍFICO (CHATBOT) ---
    # =========================================================
    with tab_chat:
        st.header("🤖 Assistente Científico (Simetrics AI)")
        st.caption("Converse com a base de dados. Peça recomendações de leitura, indicação de especialistas ou sugestões de periódicos (venues) para submeter seu artigo com base no seu tema de pesquisa atual. \n ⚠️ Esteja ciente de que a qualidade das respostas depende da qualidade dos dados carregados. Este chatbot pode errar, sempre verifique as informações.")

        # 1. Verificação segura da chave de API
        api_key = get_gemini_api_key()

        if not api_key:
            st.warning("⚠️ O Assistente Científico está desativado. Certifique-se de que a 'GEMINI_API_KEY' está configurada nas Variáveis de Ambiente do Railway ou nos Secrets do Streamlit.")
        else:
            # 2. Inicialização do Cliente e Gestão de Estado
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": "Olá! Sou a IA do Simetrics. Já injetei toda a sua base de dados, autores, temas e métricas na minha memória de contexto. Qual desafio acadêmico posso ajudar a resolver hoje?"}
                ]

            # Renderiza o histórico de mensagens
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # 3. Sincronização de Contexto e Criação da Sessão
            with st.spinner("Sincronizando sinapses com a base de dados..."):
                from utils import preparar_contexto_llm
                # O contexto é extraído da base de dados ativa (considerando filtros e deduplicação)
                dados_json = preparar_contexto_llm(st.session_state['df_geral'])
                
                instrucao_sistema = f"""
                Você é um conselheiro acadêmico sênior e especialista em cienciometria operando dentro da plataforma 'Simetrics'.
                Abaixo estão os dados ESTRITOS da revisão de literatura do usuário em formato JSON:
                
                {dados_json}
                
                Suas tarefas:
                - Responder perguntas baseadas exclusivamente nos dados fornecidos.
                - Recomendar artigos fundamentais baseando-se nos temas, resumos e impacto (citações).
                - Sugerir periódicos (SECONDARY TITLE) para submissão com base no perfil do manuscrito do usuário.
                - Identificar especialistas (autores) para parcerias ou referências.
                
                Regras Absolutas:
                - Você SÓ PODE fazer recomendações usando os itens que existem no JSON acima. 
                - Cite os títulos exatos dos documentos ou nomes dos autores como aparecem na base.
                - Responda em Português de forma acadêmica, analítica e direta.
                """

                # Converte o histórico para o formato de objetos estritos exigido pela google-genai
                gemini_history = []
                for m in st.session_state.chat_messages[1:]: 
                    role_map = "user" if m["role"] == "user" else "model"
                    gemini_history.append(
                        types.Content(
                            role=role_map, 
                            parts=[types.Part.from_text(text=m["content"])]
                        )
                    )
                
                # Inicializa a sessão de chat com a instrução de sistema injetada na configuração
                chat_session = client.chats.create(
                    model="gemini-2.5-flash-lite",
                    history=gemini_history,
                    config=types.GenerateContentConfig(
                        system_instruction=instrucao_sistema
                    )
                )

            # 4. Interface de Entrada e Geração de Resposta
            if prompt := st.chat_input("Ex: Quais são os 3 documentos fundamentais para ler sobre este tema?"):
                
                # Registra e exibe a pergunta do usuário
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Processa a resposta da IA
                with st.chat_message("assistant"):
                    with st.spinner("Analisando topologia do conhecimento..."):
                        try:
                            response = chat_session.send_message(prompt)
                            texto_resposta = response.text
                            
                            st.markdown(texto_resposta)
                            st.session_state.chat_messages.append({"role": "assistant", "content": texto_resposta})
                            
                        except Exception as e:
                            erro = f"Ocorreu um erro na geração da resposta: {e}"
                            st.error(erro)
                            st.session_state.chat_messages.append({"role": "assistant", "content": erro})

    with tab_form:
        # =========================================================
        # --- AVALIAÇÃO DE USABILIDADE (SUS) ---
        # =========================================================
        st.header("📝 Avaliação da Plataforma Simetrics")
        st.markdown("""
        Bem-vindo(a)! Este formulário tem como objetivo avaliar a sua experiência ao utilizar a plataforma **Simetrics**. 
        Não existem respostas certas ou erradas; estamos avaliando o sistema, e não você. Suas respostas são anônimas 
        e fundamentais para o aprimoramento desta ferramenta de gestão do conhecimento acadêmico.
        """)

        # 1. Tenta estabelecer a conexão com o Google Sheets
        try:
            # Ele tentará buscar a URL no secrets.toml (local) ou nas variáveis do Railway
            url_planilha = st.secrets["spreadsheet"]
            conn = st.connection("gsheets", type=GSheetsConnection)
            conexao_ok = True
        except Exception as e:
            st.error(f"Erro de configuração: {e}")
            conexao_ok = False

        # =========================================================
        # --- (CHAVE DINÂMICA) ---
        # =========================================================
        
        # 1. Controle de Memória
        if "form_key" not in st.session_state:
            st.session_state["form_key"] = 0  # Chave inicial do formulário
        if "avaliacao_sucesso" not in st.session_state:
            st.session_state["avaliacao_sucesso"] = False

        # Mostra a mensagem de sucesso e balões (se o envio deu certo na rodada anterior)
        if st.session_state["avaliacao_sucesso"]:
            st.success("✅ Avaliação enviada com sucesso! A equipe de pesquisa agradece imensamente a sua contribuição.")
            st.balloons()
            st.session_state["avaliacao_sucesso"] = False  # Desliga a bandeira

        # 2. Formulário com CHAVE DINÂMICA
        # Toda vez que form_key mudar, o formulário renasce do zero!
        chave_dinamica = f"form_avaliacao_{st.session_state['form_key']}"
        
        with st.form(key=chave_dinamica, clear_on_submit=False):
            
            # --- PARTE 1: PERFIL ---
            st.subheader("Parte 1: Perfil do Participante")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                titulacao = st.selectbox(
                    "1. Nível de titulação acadêmica (concluído ou em andamento):",
                    ["Graduação", "Especialização", "Mestrado", "Doutorado", "Pós-Doutorado"],
                    index=None, placeholder="Selecione uma opção..."
                )
                area = st.selectbox(
                    "2. Principal Grande Área de atuação:",
                    ["Engenharias / Ciências Exatas e da Terra", "Ciências Sociais Aplicadas / Humanas", "Ciências da Saúde / Biológicas", "Outra"],
                    index=None, placeholder="Selecione uma opção..."
                )
            with col_p2:
                experiencia = st.radio(
                    "3. Experiência prévia com softwares de análise bibliométrica:",
                    ["Nenhuma experiência", "Iniciante (Já utilizei algumas vezes)", "Intermediário (Utilizo com certa regularidade)", "Avançado (Tenho domínio sobre as ferramentas)"],
                    index=None
                )

            st.divider()
            
            # --- PARTE 2: SUS ---
            st.subheader("Parte 2: Questionário Usabilidade")
            st.caption("Selecione sua concordância (1 = Discordo Fortemente | 5 = Concordo Fortemente)")

            opcoes_likert = [1, 2, 3, 4, 5]
            sus_1 = st.radio("1. Eu acho que gostaria de usar a plataforma Simetrics frequentemente.", opcoes_likert, index=None, horizontal=True)
            sus_2 = st.radio("2. Eu achei a plataforma desnecessariamente complexa.", opcoes_likert, index=None, horizontal=True)
            sus_3 = st.radio("3. Eu achei a plataforma fácil de usar.", opcoes_likert, index=None, horizontal=True)
            sus_4 = st.radio("4. Eu acho que precisaria de suporte técnico para usar a plataforma.", opcoes_likert, index=None, horizontal=True)
            sus_5 = st.radio("5. Eu achei que as várias funções estavam muito bem integradas.", opcoes_likert, index=None, horizontal=True)
            sus_6 = st.radio("6. Eu achei que havia muita inconsistência na plataforma.", opcoes_likert, index=None, horizontal=True)
            sus_7 = st.radio("7. Eu imagino que a maioria aprenderia a usar essa plataforma rapidamente.", opcoes_likert, index=None, horizontal=True)
            sus_8 = st.radio("8. Eu achei a plataforma muito difícil ou engessada de usar.", opcoes_likert, index=None, horizontal=True)
            sus_9 = st.radio("9. Eu me senti muito confiante ao navegar na plataforma.", opcoes_likert, index=None, horizontal=True)
            sus_10 = st.radio("10. Eu precisei aprender muitas coisas novas antes de lidar com a plataforma.", opcoes_likert, index=None, horizontal=True)

            st.divider()
            
            # --- PARTE 3: UX ---
            st.subheader("Parte 3: Avaliação de Interface e Experiência")
            ux_nav = st.text_area("11. Navegação e Arquitetura da Informação (Durante a importação da base e navegação, você encontrou alguma barreira ou teve dúvidas sobre onde clicar?):")
            ux_vis = st.text_area("12. Visualização de Dados e Metáforas (As visualizações (Boxplot, Sankey) foram claras e ajudaram na compreensão?):")
            ux_ia = st.text_area("13. Inteligência Artificial e Classificação Temática (Como avalia o agrupamento e os nomes gerados pela IA na base de dados testada? E também na conversa com a IA, as respostas fizeram sentido?):")
            ux_fric = st.text_area("14. Pontos de Fricção e Melhorias (Se você fosse o engenheiro responsável, qual seria a primeira alteração que faria na interface?):")
            ux_add = st.text_area("15. Comentários Adicionais (Opcional) (Deixe qualquer observação, crítica ou sugestão que não tenha sido abordada):")

            submit_btn = st.form_submit_button("Enviar Avaliação", type="primary", use_container_width=True)

        # 3. Lógica de Envio
        if submit_btn:
            respostas_sus = [sus_1, sus_2, sus_3, sus_4, sus_5, sus_6, sus_7, sus_8, sus_9, sus_10]
            
            if None in respostas_sus or titulacao is None or area is None or experiencia is None:
                st.error("⚠️ Por favor, responda todas as questões de múltipla escolha antes de enviar.")
            else:
                try:
                    url_planilha = st.secrets["connections"]["gsheets"]["spreadsheet"]
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    
                    with st.spinner("⚙️ Por favor, aguarde. Registrando avaliação no banco de dados..."):
                        nome_aba_planilha = "Respostas" 
                        
                        df_existente = conn.read(spreadsheet=url_planilha, worksheet=nome_aba_planilha, ttl=0)
                        df_existente = df_existente.dropna(how="all")

                        nova_resposta = pd.DataFrame([{
                            "Titulação": titulacao, "Área": area, "Experiência_Prévia": experiencia,
                            "SUS_01": sus_1, "SUS_02": sus_2, "SUS_03": sus_3, "SUS_04": sus_4, "SUS_05": sus_5,
                            "SUS_06": sus_6, "SUS_07": sus_7, "SUS_08": sus_8, "SUS_09": sus_9, "SUS_10": sus_10,
                            "UX_Navegação": ux_nav, "UX_Visualização": ux_vis, "UX_IA": ux_ia,
                            "UX_Melhorias": ux_fric, "UX_Comentários": ux_add,
                            "Data_Hora": pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%Y-%m-%d %H:%M:%S')
                        }])

                        df_atualizado = pd.concat([df_existente, nova_resposta], ignore_index=True)
                        df_atualizado = df_atualizado.fillna("").astype(str)
                        conn.update(spreadsheet=url_planilha, worksheet=nome_aba_planilha, data=df_atualizado)
                        
                        # --- O PULO DO GATO ---
                        # 1. Avisamos o sistema que deu sucesso
                        st.session_state["avaliacao_sucesso"] = True
                        # 2. Mudamos o ID do formulário (forçando ele a nascer em branco)
                        st.session_state["form_key"] += 1
                        # 3. Recarregamos a tela
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Erro ao processar envio: {e}")