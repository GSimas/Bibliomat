![Simetrics Logo](simetrics%20-%20logo.png)

# Simetrics

**Plataforma de Inteligência Bibliométrica e Mapeamento Científico**
*Explorando a Ecologia do Conhecimento através de Dados, Redes e Inteligência Artificial.*

---

## 🧬 Sobre o Simetrics

O **Simetrics** é uma plataforma analítica desenvolvida para pesquisadores, acadêmicos e analistas de dados que buscam uma ferramenta para bibliometria/cientometria. O projeto mapeia a estrutura intelectual a partir de arquivos de bases de dados (como Scopus, Web of Science, Scielo)

Este sistema permite a visualização de fluxos de colaboração global, a identificação de lacunas de pesquisa e a categorização temática automatizada, transformando bases de dados brutas em visões estratégicas.

## 🚀 Funcionalidades Principais

### 🤖 Inteligência Artificial & Categorização
* **Clustering Híbrido:** Agrupamento automático de documentos via TF-IDF e K-Means.
* **Otimização Silhouette:** Cálculo do número ideal de temas para evitar viés humano.
* **Rotulação via Gemini API:** Integração com Google Gemini para síntese semântica e nomeação de clusters.

### 🧠 Mapeamento Conceitual (Epistemologia Visual)
* **PCA 2D & 3D:** Redução de dimensionalidade para visualização da topologia do conhecimento em eixos interativos.
* **Identificação de Clusters:** Visualização de "ilhas de conhecimento" e termos de fronteira.

### 🌍 Geopolítica do Conhecimento
* **Global Collaboration Map:** Mapa coroplético interativo com arestas proporcionais à força da colaboração internacional.
* **Circular Network:** Grafo chordal detalhando parcerias entre nações com informações de hover dinâmico.

### 📊 Lexicometria e Impacto
* **Top Keywords por Impacto:** Rankings dinâmicos por quantidade de documentos, total de citações ou média de impacto.
* **Historiograph:** Linha do tempo interativa de conexões entre documentos.
* **Análise de Lei de Lotka:** Verificação da produtividade dos autores da base.

### 📋 Tabelas Analíticas de Deep-Dive
* Abas dedicadas para **Autores**, **Países**, **Fontes (Venues)** e **Palavras-chave**.
* Estatísticas descritivas completas (Média, Mediana, Desvio Padrão) e linhas do tempo formatadas por entidade.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12+
* **Interface:** Streamlit
* **Processamento de Dados:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-learn (PCA, K-Means, Silhouette Score)
* **Visualização:** Plotly (2D/3D/Geo), NetworkX, PyEcharts, WordCloud
* **IA Generativa:** Google Generative AI (Gemini 2.5 Flash)

## 📦 Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone [https://github.com/GSimas/simetrics.git](https://github.com/GSimas/simetrics.git)
cd simetrics
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

O projeto é inspirado em outras ferramentas como Bibliometrix, VOSViewer e NoCodeFunctions. 

☕ Desenvolvido com ajuda de alguns cafés por [Gustavo Simas](https://github.com/GSimas/)