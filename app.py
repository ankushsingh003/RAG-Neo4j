"""
Streamlit Web Application: Advanced RAG System
Features:
1. Hybrid Search + Reranking Playground
2. GraphRAG & Interactive Knowledge Graph Explorer (PyVis & Cypher)
3. RAGAS Evaluation Hub (Radar Charts & Metric Comparisons)
4. Knowledge Ingestion & Connection Settings
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import json
import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pyvis.network import Network

# Import core modules
from config import settings
from src.hybrid_rag.hybrid_pipeline import HybridRAGPipeline
from src.graph_rag.graph_pipeline import GraphRAGPipeline
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.reporter import BenchmarkReporter

# Streamlit page configuration
st.set_page_config(
    page_title="Advanced RAG Studio: Hybrid, GraphRAG & RAGAS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern, premium appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0e7ff;
        color: #3730a3;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache pipelines in session state so models stay in memory
@st.cache_resource(show_spinner="Initializing Hybrid RAG Pipeline...")
def get_hybrid_pipeline():
    pipeline = HybridRAGPipeline()
    pipeline.ingest_corpus()
    return pipeline

@st.cache_resource(show_spinner="Initializing GraphRAG Pipeline & Knowledge Graph...")
def get_graph_pipeline():
    pipeline = GraphRAGPipeline()
    pipeline.ingest_corpus()
    return pipeline

@st.cache_resource(show_spinner="Initializing RAGAS Benchmark Engine...")
def get_benchmark_runner():
    return BenchmarkRunner()

# Initialize pipelines
try:
    hybrid_pipe = get_hybrid_pipeline()
    graph_pipe = get_graph_pipeline()
    bench_runner = get_benchmark_runner()
except Exception as e:
    st.error(f"Error initializing pipelines: {e}")
    st.stop()

# Sidebar: System Status & Configuration
st.sidebar.title("⚙️ System Control")
st.sidebar.markdown(f"**Corpus Chunks**: `{len(hybrid_pipe.chunks)}`")
if graph_pipe.client.is_live_neo4j:
    st.sidebar.success("🟢 Live Neo4j Connected")
else:
    st.sidebar.info("🟡 In-Memory Graph Fallback Active")

st.sidebar.markdown(f"**Embedding Model**: `{settings.EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**Reranker Model**: `{settings.RERANKER_MODEL_NAME}`")
st.sidebar.markdown(f"**LLM Provider**: `{hybrid_pipe.generator.active_provider}`")

st.sidebar.divider()
st.sidebar.markdown("### 📚 Quick Presets")
preset_queries = [
    "What is the smoothing constant k in Reciprocal Rank Fusion?",
    "Why is a Cross-Encoder better than a bi-encoder for passage reranking?",
    "What services does AuthGateway depend on and communicate with?",
    "How does PaymentEngine use Neo4j to detect fraud rings?",
    "How does Neo4j achieve O(1) traversal step complexity in graph queries?"
]

selected_preset = st.sidebar.selectbox("Load Sample Query:", ["-- Select or Type Custom --"] + preset_queries)

# Main Header
st.markdown('<div class="main-header">⚡ Advanced RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Grade Hybrid Search + Reranking | GraphRAG with Neo4j | RAGAS Evaluation</div>', unsafe_allow_html=True)

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Hybrid Search + Reranking",
    "🕸️ 2. GraphRAG with Neo4j",
    "📊 3. Evaluate with RAGAS",
    "📂 4. Ingestion & Settings"
])

# ==============================================================================
# TAB 1: HYBRID SEARCH + RERANKING
# ==============================================================================
with tab1:
    st.markdown("### 🔍 Hybrid Lexical (BM25) + Dense Vector Search + Cross-Encoder Reranking")
    st.caption("Combines term-frequency exact matching with deep vector semantic similarity, fused via RRF or Convex Linear weighting, then re-scored by token-to-token cross-attention.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Search Query:",
            value=selected_preset if selected_preset != "-- Select or Type Custom --" else "What is the smoothing constant k in Reciprocal Rank Fusion?",
            key="tab1_query"
        )
    with col2:
        fusion_strategy = st.selectbox("Candidate Fusion Strategy:", ["RRF (Reciprocal Rank Fusion)", "Convex Linear Score"], index=0)

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        top_k = st.slider("Retrieval Depth (Top-K):", min_value=3, max_value=10, value=5)
    with col_p2:
        alpha = st.slider("Dense Alpha (0.0 = Pure BM25, 1.0 = Pure Dense):", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=(fusion_strategy.startswith("RRF")))
    with col_p3:
        enable_reranker = st.toggle("Enable Cross-Encoder Reranker", value=True)
        rerank_top_n = st.slider("Rerank Output Count (Top-N):", min_value=1, max_value=5, value=3, disabled=not enable_reranker)

    if st.button("🚀 Execute Hybrid Search & Rerank", type="primary", key="btn_hybrid"):
        method_str = "rrf" if fusion_strategy.startswith("RRF") else "linear"
        with st.spinner("Executing sparse, dense, fusion, and reranking pipeline..."):
            res = hybrid_pipe.query(
                query=query_input,
                top_k=top_k,
                fusion_method=method_str,
                alpha=alpha,
                apply_rerank=enable_reranker,
                rerank_top_n=rerank_top_n
            )

        # Timings badge row
        t = res["timings"]
        st.markdown(
            f'<span class="badge-pill">⏱️ Total: {t["total_search_ms"] + t["generation_ms"]:.1f}ms</span>'
            f'<span class="badge-pill">BM25: {t["bm25_retrieval_ms"]:.1f}ms</span>'
            f'<span class="badge-pill">Dense: {t["dense_retrieval_ms"]:.1f}ms</span>'
            f'<span class="badge-pill">Fusion: {t["fusion_ms"]:.1f}ms</span>'
            f'<span class="badge-pill">Reranking: {t["reranking_ms"]:.1f}ms</span>'
            f'<span class="badge-pill">LLM: {t["generation_ms"]:.1f}ms</span>',
            unsafe_allow_html=True
        )

        st.markdown("#### 🤖 Grounded Generation Response")
        st.info(f"**Answer**: {res['answer']}")
        if res.get("citations"):
            st.caption(f"Cited Sources: {', '.join([c['chunk_id'] for c in res['citations']])} | Provider: {res['llm_provider']}")

        st.divider()
        st.markdown("#### 🔬 Retrieval Breakdown & Candidate Evolution")

        subcol1, subcol2, subcol3 = st.columns(3)
        with subcol1:
            st.markdown("##### 1. BM25 Lexical (Top-3)")
            for item in res["bm25_candidates"][:3]:
                st.markdown(f"**[{item['chunk_id']}]** (Score: `{item['raw_score']:.2f}`)")
                st.caption(f"Matched terms: {item.get('matched_terms', [])}")
                st.text(item["content"][:140] + "...")

        with subcol2:
            st.markdown("##### 2. Dense Vector (Top-3)")
            for item in res["dense_candidates"][:3]:
                st.markdown(f"**[{item['chunk_id']}]** (Cos Sim: `{item['raw_score']:.3f}`)")
                st.caption(f"Norm score: {item['normalized_score']:.3f}")
                st.text(item["content"][:140] + "...")

        with subcol3:
            st.markdown(f"##### 3. Fused ({fusion_strategy[:3]})")
            for item in res["fused_candidates"][:3]:
                st.markdown(f"**Rank {item['fused_rank']}**: [{item['chunk_id']}]")
                st.caption(f"Fused Score: `{item['fused_score']:.4f}` (BM25: #{item.get('bm25_rank')}, Dense: #{item.get('dense_rank')})")
                st.text(item["content"][:140] + "...")

        if enable_reranker:
            st.markdown("#### 🎯 Cross-Encoder Reranking Impact (Before vs. After)")
            rerank_table = []
            for item in res["final_passages"]:
                delta = item.get("rank_delta", 0)
                delta_str = f"🔺 +{delta}" if delta > 0 else (f"🔻 {delta}" if delta < 0 else "⏺️ 0")
                rerank_table.append({
                    "Final Rank": item["final_rank"],
                    "Pre-Rerank Rank": item.get("pre_rerank_rank"),
                    "Rank Shift": delta_str,
                    "Chunk ID": item["chunk_id"],
                    "Rerank Cross-Attention Score": round(item.get("rerank_score", 0.0), 3),
                    "Fused Score": round(item.get("fused_score", 0.0), 4),
                    "Passage Preview": item["content"][:120] + "..."
                })
            st.dataframe(pd.DataFrame(rerank_table), use_container_width=True)

# ==============================================================================
# TAB 2: GRAPHRAG WITH NEO4J
# ==============================================================================
with tab2:
    st.markdown("### 🕸️ GraphRAG: Neo4j Knowledge Graph & Multi-hop Reasoning")
    st.caption("Traverses structured entity relationships and combines multi-hop graph subgraphs with vector text chunks to eliminate multi-document reasoning blind spots.")

    g_col1, g_col2 = st.columns([3, 1])
    with g_col1:
        graph_query = st.text_input(
            "GraphRAG Query:",
            value=selected_preset if selected_preset != "-- Select or Type Custom --" else "What services does AuthGateway depend on and communicate with?",
            key="tab2_query"
        )
    with g_col2:
        max_hops = st.slider("Graph Traversal Hops:", min_value=1, max_value=3, value=2)

    if st.button("🕸️ Execute GraphRAG Query", type="primary", key="btn_graph"):
        with st.spinner("Extracting query entities, executing Cypher/graph traversal, and synthesizing..."):
            g_res = graph_pipe.query(query=graph_query, max_hops=max_hops, top_vector_chunks=2)

        st.markdown("#### 🤖 Graph-Grounded LLM Answer")
        st.success(f"**Answer**: {g_res['answer']}")
        st.caption(f"Live Neo4j Active: {g_res['is_live_neo4j']} | Citations: {', '.join([c['chunk_id'] for c in g_res['citations']])}")

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("##### 🏷️ Anchor Entities Linked in Query")
            st.write(g_res["anchor_entities"])

            st.markdown("##### 🔗 Discovered Knowledge Graph Triples")
            if g_res["graph_triples"]:
                triple_df = pd.DataFrame(g_res["graph_triples"])[["source", "relationship", "target"]]
                st.dataframe(triple_df, use_container_width=True)
            else:
                st.info("No explicit relations found for anchor entities. Used dense vector fallback.")

        with res_col2:
            st.markdown("##### 📚 Formatted Graph Path Context")
            for fact in g_res["formatted_graph_facts"]:
                st.code(fact, language="cypher")

    st.divider()
    st.markdown("#### 🌐 Interactive 2D Knowledge Graph Explorer")
    st.caption("Visualizes all nodes (Entities, Microservices, Databases, Algorithms) and typed edges.")

    all_graph = graph_pipe.client.get_all_graph_data()
    if all_graph["nodes"]:
        net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="#1e293b", directed=True)
        net.barnes_hut()

        color_map = {
            "ALGORITHM": "#06b6d4",
            "DATABASE": "#eab308",
            "MICROSERVICE": "#8b5cf6",
            "SECURITY_ENTITY": "#ef4444",
            "ARCHITECTURE_PATTERN": "#10b981",
            "Chunk": "#94a3b8",
            "Entity": "#3b82f6"
        }

        for n in all_graph["nodes"][:80]:
            label = n.get("label", "Entity")
            c = color_map.get(label, "#3b82f6")
            net.add_node(n["id"], label=n.get("name", n["id"]), color=c, title=f"Type: {label}\n{n.get('description', '')}")

        for e in all_graph["edges"][:120]:
            net.add_edge(e["source"], e["target"], label=e.get("relationship", ""), color="#cbd5e1")

        graph_html_path = settings.DATA_DIR / "temp_graph.html"
        net.save_graph(str(graph_html_path))
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=480)
    else:
        st.info("No graph data available to display.")

# ==============================================================================
# TAB 3: EVALUATE WITH RAGAS
# ==============================================================================
with tab3:
    st.markdown("### 📊 RAGAS Evaluation Hub & Comparative Benchmark")
    st.caption("Standardized evaluation measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall across Baseline Vector, Hybrid+Reranking, and GraphRAG.")

    if st.button("🚀 Run Complete RAGAS Benchmark", type="primary", key="btn_eval"):
        with st.spinner("Evaluating test dataset across all 3 RAG architectures..."):
            bench_results = bench_runner.run_benchmark()
            st.session_state["bench_results"] = bench_results

    if "bench_results" in st.session_state:
        b_res = st.session_state["bench_results"]

        # 1. Macro Averages Table
        summary_rows = []
        labels = {
            "baseline_vector": "1. Baseline Dense Vector",
            "hybrid_rerank": "2. Hybrid + Cross-Encoder Reranking",
            "graph_rag": "3. GraphRAG with Neo4j"
        }
        for k, name in labels.items():
            avg = b_res[k]["averages"]
            summary_rows.append({
                "Architecture": name,
                "Faithfulness": avg["faithfulness"],
                "Answer Relevancy": avg["answer_relevancy"],
                "Context Precision": avg["context_precision"],
                "Context Recall": avg["context_recall"],
                "Composite RAGAS Score": avg["ragas_score"],
                "Avg Latency (ms)": avg["latency_ms"]
            })

        df_summary = pd.DataFrame(summary_rows)
        st.markdown("#### 🏆 Benchmark Summary Across 4 RAGAS Dimensions")
        st.dataframe(df_summary.style.highlight_max(subset=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Composite RAGAS Score"], color="#dcfce7"), use_container_width=True)

        # 2. Plotly Radar Chart
        st.markdown("#### 🕸️ Radar Comparison Chart")
        categories = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[b_res["baseline_vector"]["averages"][c.lower().replace(" ", "_")] for c in categories],
            theta=categories,
            fill='toself',
            name='Baseline Dense Vector',
            line_color='#94a3b8'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[b_res["hybrid_rerank"]["averages"][c.lower().replace(" ", "_")] for c in categories],
            theta=categories,
            fill='toself',
            name='Hybrid + Reranker',
            line_color='#4f46e5'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[b_res["graph_rag"]["averages"][c.lower().replace(" ", "_")] for c in categories],
            theta=categories,
            fill='toself',
            name='GraphRAG (Neo4j)',
            line_color='#06b6d4'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=True,
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. Drilldown inspector
        st.markdown("#### 🔍 Test Sample Drilldown")
        sample_questions = [s["question"] for s in b_res["hybrid_rerank"]["samples"]]
        chosen_q = st.selectbox("Select Sample Question:", sample_questions)

        sample_idx = sample_questions.index(chosen_q)
        s_base = b_res["baseline_vector"]["samples"][sample_idx]
        s_hyb = b_res["hybrid_rerank"]["samples"][sample_idx]
        s_graph = b_res["graph_rag"]["samples"][sample_idx]

        drill_c1, drill_c2, drill_c3 = st.columns(3)
        with drill_c1:
            st.markdown("##### Baseline Vector")
            st.write(s_base["metrics"])
            st.caption(f"Answer: {s_base['answer'][:160]}...")
        with drill_c2:
            st.markdown("##### Hybrid + Reranker")
            st.write(s_hyb["metrics"])
            st.caption(f"Answer: {s_hyb['answer'][:160]}...")
        with drill_c3:
            st.markdown("##### GraphRAG (Neo4j)")
            st.write(s_graph["metrics"])
            st.caption(f"Answer: {s_graph['answer'][:160]}...")

    else:
        st.info("Click 'Run Complete RAGAS Benchmark' above to execute the standardized comparative evaluation.")

# ==============================================================================
# TAB 4: INGESTION & SETTINGS
# ==============================================================================
with tab4:
    st.markdown("### 📂 Document Ingestion & Neo4j Settings")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("#### 📄 Ingest New Documents")
        uploaded_file = st.file_uploader("Upload Text or Markdown file:", type=["txt", "md"])
        if uploaded_file is not None:
            save_path = settings.DATA_DIR / "sample_corpus" / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved {uploaded_file.name} to corpus!")
            if st.button("🔄 Re-index All Pipelines"):
                hybrid_pipe.ingest_corpus()
                graph_pipe.ingest_corpus()
                st.success("Re-indexing complete across BM25, VectorStore, and Knowledge Graph!")

    with col_s2:
        st.markdown("#### 🔌 Live Neo4j Connection Settings")
        neo_uri = st.text_input("Neo4j URI:", value=settings.NEO4J_URI)
        neo_user = st.text_input("Neo4j Username:", value=settings.NEO4J_USERNAME)
        neo_pwd = st.text_input("Neo4j Password:", type="password", value=settings.NEO4J_PASSWORD)
        if st.button("Test Neo4j Connection"):
            try:
                from neo4j import GraphDatabase
                d = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pwd), connection_timeout=3.0)
                d.verify_connectivity()
                d.close()
                st.success(f"Connected to Neo4j successfully at {neo_uri}!")
            except Exception as e:
                st.error(f"Could not connect to Neo4j: {e}")
