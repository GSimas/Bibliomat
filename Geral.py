import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_agraph import agraph, Config
from utils import process_multiple_ris, criar_grafo_e_metricas, deduplicar_por_doi, deduplicar_por_similaridade

st.set_page_config(page_title="Bibliometrix Python", page_icon="🧬", layout="wide")

st.title("🧬 Painel de Análise Bibliométrica e Cientométrica")

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
    st.header("1. Envio de Arquivos")
    uploaded_files = st.file_uploader("Selecione arquivos RIS", type=['ris', 'txt'], accept_multiple_files=True)
    
    db_mapping = {}
    if uploaded_files:
        st.header("2. Atribuição de Base")
        for f in uploaded_files:
            db_mapping[f.name] = st.selectbox(f"{f.name}", options=["Scopus", "Web of Science", "SciELO", "Outra"], key=f"db_{f.name}")
            
        if st.button("Processar e Integrar", type="primary"):
            with st.spinner("Estruturando conhecimento..."):
                df_raw = process_multiple_ris(uploaded_files, db_mapping)
                st.session_state['df_original'] = df_raw.copy() if df_raw is not None else None
                st.session_state['df_geral'] = df_raw
                # IMPORTANTE: Resetar como DataFrame vazio aqui
                st.session_state['df_duplicados'] = pd.DataFrame()

if st.session_state['df_geral'] is not None:
    df = st.session_state['df_geral']
    
    tab_main, tab_grafos = st.tabs(["📊 Informações Principais", "🕸️ Redes e Grafos de Conhecimento"])

    with tab_main:
        st.subheader("Resumo Estrutural da Amostra")
        total_docs = len(df)
        
        timespan = f"{int(df['YEAR CLEAN'].min())}:{int(df['YEAR CLEAN'].max())}" if 'YEAR CLEAN' in df.columns and pd.notna(df['YEAR CLEAN'].min()) else "N/A"
        avg_age = round(2026 - df['YEAR CLEAN'].mean(), 2) if 'YEAR CLEAN' in df.columns else "N/A"
        authors_count = "N/A"
        if 'AUTHORS' in df.columns:
            flat_auths = [a.strip() for sublist in df['AUTHORS'].dropna().astype(str).str.split(';') for a in sublist if a.strip()]
            authors_count = len(set(flat_auths))

        c1, c2, c3, c4 = st.columns(4)
        with c1: create_kpi_card("Período", timespan)
        with c2: create_kpi_card("Total de Documentos", total_docs)
        with c3: create_kpi_card("Autores Únicos", authors_count, "#2E86C1", "white")
        with c4: create_kpi_card("Média de Idade (Docs)", avg_age, "#2E86C1", "white")

        st.divider()

        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1:
            st.markdown("##### Dinâmica de Produção Científica")
            if 'YEAR CLEAN' in df.columns:
                pubs_per_year = df.dropna(subset=['YEAR CLEAN'])['YEAR CLEAN'].value_counts().reset_index()
                pubs_per_year.columns = ['Ano', 'Documentos']
                fig_year = px.line(pubs_per_year.sort_values('Ano'), x='Ano', y='Documentos', markers=True, color_discrete_sequence=['#1273B9'])
                st.plotly_chart(fig_year, use_container_width=True)

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
                author_data = []
                for _, row in df.iterrows():
                    auth_list = [a.strip() for a in str(row['AUTHORS']).split(';') if a.strip()]
                    cit = row['TOTAL CITATIONS'] if pd.notna(row['TOTAL CITATIONS']) else 0
                    for a in auth_list:
                        author_data.append({'Autor': a, 'Documentos': 1, 'Citações': cit})
                
                df_auth_expanded = pd.DataFrame(author_data)
                # Agrupamento robusto para calcular a média
                res_auth = df_auth_expanded.groupby('Autor').agg({
                    'Documentos': 'sum', 
                    'Citações': 'sum'
                }).reset_index()
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

        # --- LINHA 3 DE GRÁFICOS: TOP PAÍSES ---
        st.divider()
        st.markdown("##### 🌍 Top 20 Países Mais Produtivos/Citados")
        
        col_country_sel, col_country_graph = st.columns([1, 3])
        
        with col_country_sel:
            st.write("") # Espaçador
            metric_country = st.radio(
                "Métrica para o Ranking de Países:", 
                ["Quantidade de Documentos", "Total de Citações", "Média de Citações"], 
                key="sel_pais"
            )
            st.info("Nota: Em casos de coautoria internacional, o crédito do documento e das citações é atribuído integralmente a cada país declarado no endereço dos autores.")

        with col_country_graph:
            if 'COUNTRY' in df.columns:
                country_data = []
                df_with_country = df.dropna(subset=['COUNTRY'])
                
                for _, row in df_with_country.iterrows():
                    countries = [c.strip() for c in str(row['COUNTRY']).split(';') if c.strip()]
                    cit = row['TOTAL CITATIONS'] if pd.notna(row['TOTAL CITATIONS']) else 0
                    for c in countries:
                        country_data.append({'País': c, 'Documentos': 1, 'Citações': cit})
                
                if country_data:
                    df_country_expanded = pd.DataFrame(country_data)
                    # Agrupamento para calcular a média por país
                    res_country = df_country_expanded.groupby('País').agg({
                        'Documentos': 'sum', 
                        'Citações': 'sum'
                    }).reset_index()
                    res_country['Média'] = (res_country['Citações'] / res_country['Documentos']).round(2)
                    
                    if metric_country == "Quantidade de Documentos":
                        top_c = res_country.nlargest(20, 'Documentos')
                        fig_c = px.bar(top_c, x='Documentos', y='País', orientation='h', color='Documentos', color_continuous_scale='Viridis')
                    elif metric_country == "Total de Citações":
                        top_c = res_country.nlargest(20, 'Citações')
                        fig_c = px.bar(top_c, x='Citações', y='País', orientation='h', color='Citações', color_continuous_scale='Plasma')
                    else: # Média de Citações por País
                        top_c = res_country.nlargest(20, 'Média')
                        fig_c = px.bar(top_c, x='Média', y='País', orientation='h', color='Média', color_continuous_scale='YlGnBu')
                    
                    fig_c.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_c, use_container_width=True)
        st.divider()

        # =========================================================
        # MÓDULO DE DEDUPLICAÇÃO E TABELAS
        # =========================================================
        st.markdown("### 📋 Tabela Completa e Estatísticas de Citação")
        
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
                    st.success("✅ Nenhum documento duplicado encontrado nesta ação.")

        if st.session_state['df_duplicados'] is not None and not st.session_state['df_duplicados'].empty:
            st.markdown("---")
            st.markdown("### 🚨 Relatório Permanente de Documentos Excluídos")
            
            dupes = st.session_state['df_duplicados']
            # Definindo colunas para exibição
            cols_exibicao = [c for c in ['TITLE', 'TI', 'DOCUMENTO DE REFERÊNCIA (MANTIDO)', 'BASE DE DADOS', 'DOI'] if c in dupes.columns]
            
            st.dataframe(dupes[cols_exibicao], use_container_width=True, hide_index=True)
            
            # Botão para baixar apenas os excluídos (útil para o anexo da metodologia PRISMA)
            csv_dupes = dupes.to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Relatório de Excluídos (CSV)", data=csv_dupes, file_name='documentos_excluidos.csv')

        st.write("")
        
        # Estatísticas e Tabela Bruta (Da base já limpa, caso tenha rodado a limpeza)
        if 'TOTAL CITATIONS' in df.columns:
            stats_df = df['TOTAL CITATIONS'].dropna()
            if not stats_df.empty:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Média de Citações", f"{stats_df.mean():.2f}")
                e2.metric("Mediana", f"{stats_df.median():.2f}")
                e3.metric("Desvio Padrão", f"{stats_df.std():.2f}")
                e4.metric("Máximo", f"{stats_df.max():.0f}")
        
        st.markdown("##### 📚 Base de Dados Ativa")
        cols = [c for c in df.columns if c != 'YEAR CLEAN']
        if 'TOTAL CITATIONS' in cols: cols.insert(0, cols.pop(cols.index('TOTAL CITATIONS')))
        st.dataframe(df[cols], use_container_width=True)
        st.download_button("Baixar Base Unificada (CSV)", data=df[cols].to_csv(index=False).encode('utf-8'), file_name='base_integrada.csv')

    # === ABA 2: REDES E GRAFOS ===
    with tab_grafos:
        st.subheader("Mapeamento do Ecossistema de Conhecimento")
        col_opcoes, col_grafo = st.columns([1, 3])
        
        with col_opcoes:
            tipo_grafo = st.selectbox("Mapear:", ["Rede de Coautoria", "Coocorrência de Palavras-chave"])
            top_n_nodes = st.slider("Top N Nós:", 10, 150, 50, 5)
            metric_for_size = st.selectbox("Basear tamanho do nó em:", ["Tamanho Fixo", "Grau Absoluto", "Centralidade (Eigen)", "Betweenness", "Closeness"])
            
            coluna_alvo = "AUTHORS" if tipo_grafo == "Rede de Coautoria" else None
            if not coluna_alvo:
                for col in ['KEYWORDS', 'KW', 'DE']:
                    if col in df.columns: coluna_alvo = col; break

        with col_grafo:
            if coluna_alvo and coluna_alvo in df.columns:
                with st.spinner("Calculando topologia SNA..."):
                    nodes, edges, df_nodes, net_metrics = criar_grafo_e_metricas(df, coluna_alvo, top_n_nodes, metric_for_size)
                    if len(nodes) > 0:
                        # Substitua a configuração do agraph por esta no Geral.py:
                        config = Config(
                            width="100%", 
                            height=700, # Aumentamos a altura para garantir que os botões não fiquem escondidos na borda
                            directed=False, 
                            physics=True, 
                            hierarchical=False,
                            # Parâmetro essencial para renderizar o D-pad e botões de +/-
                            navigationButtons=True, 
                            interaction={
                                "hover": True, 
                                "zoomView": True, 
                                "dragView": True,
                                # Em algumas versões, repetir aqui garante a ativação
                                "navigationButtons": True 
                            },
                            nodeHighlightBehavior=True,
                            highlightColor="#F7A7A6",
                            # Adicionando um pouco de suavização física para a rede não "fugir" da tela
                            stabilization=True 
                        )

                        # Na chamada do agraph, os botões agora devem aparecer no canto inferior
                        agraph(nodes=nodes, edges=edges, 
                        config = config
                        )
                    else: st.warning("Sem conexões suficientes.")
            else: st.warning("Coluna não encontrada.")

        st.divider()
        if coluna_alvo and len(nodes) > 0:
            st.markdown("### 📋 Tabela de Nós e Métricas")
            st.dataframe(df_nodes, use_container_width=True, hide_index=True)