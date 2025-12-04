"""RAG Research Bot - Streamlit UI with Agent, Memory, Reranking & Knowledge Graph"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(
    page_title="RAG Research Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from database_manager import DatabaseManager
from vector_store import VectorStore
from memory import ConversationMemory, LongTermMemory
from reranker import LLMReranker
from agent import ResearchAgent
from email_utils import send_papers_email
from graph_module import create_graph_for_streamlit, get_graph_stats

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }

    .main {
        padding: 2rem;
        background-color: #0f1729;
        color: #e2e8f0;
    }

    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.4rem !important; }

    /* Paragraphs and text */
    p, div, span, label {
        color: #e2e8f0 !important;
        line-height: 1.6;
    }

    /* Hero section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3rem 1.5rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.5);
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #e0e7ff !important;
        max-width: 700px;
        line-height: 1.7;
    }

    /* Chat messages with better contrast */
    .chat-message {
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .user-message {
        background-color: #1e3a8a;
        color: #e0e7ff !important;
        border-left: 4px solid #3b82f6;
    }
    .user-message b {
        color: #93c5fd !important;
    }
    .bot-message {
        background-color: #1e293b;
        color: #e2e8f0 !important;
        border-left: 4px solid #64748b;
    }
    .bot-message b {
        color: #cbd5e1 !important;
    }

    /* Text inputs with clear visibility */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
        padding: 0.875rem !important;
        font-size: 15px !important;
        color: #f8fafc !important;
    }
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
        outline: none;
    }

    /* Chat input */
    .stChatInput>div>div>input {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        color: #f8fafc !important;
        font-size: 15px !important;
    }

    /* Select boxes */
    .stSelectbox>div>div>div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }

    /* Result cards */
    .result-card {
        background-color: #1e293b;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
        transition: all 0.2s ease;
    }
    .result-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
    }

    /* Sidebar with better contrast */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    /* Sidebar inputs */
    [data-testid="stSidebar"] .stTextInput>div>div>input,
    [data-testid="stSidebar"] .stSelectbox>div>div>select {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569 !important;
    }

    /* Buttons with clear contrast */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 1.75rem;
        font-weight: 600;
        font-size: 15px !important;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Success/Error/Info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        color: #f8fafc !important;
        border-radius: 8px !important;
    }

    /* Links */
    a {
        color: #60a5fa !important;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        color: #93c5fd !important;
        text-decoration: underline;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        color: #cbd5e1;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Number input */
    .stNumberInput>div>div>input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 2px solid #475569 !important;
    }

    /* Checkbox */
    .stCheckbox {
        color: #e2e8f0 !important;
    }

    /* Spinner */
    .stSpinner>div {
        border-top-color: #3b82f6 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_components():
    """Initialize all components once"""
    db = DatabaseManager()
    reranker = LLMReranker()
    vector_store = VectorStore(reranker=reranker)
    return db, vector_store, reranker

db, vector_store, reranker = init_components()

# Session-based components (not cached)
if 'conv_memory' not in st.session_state:
    st.session_state.conv_memory = ConversationMemory()

if 'lt_memory' not in st.session_state:
    st.session_state.lt_memory = LongTermMemory(db)

if 'agent' not in st.session_state:
    st.session_state.agent = ResearchAgent(
        vector_store=vector_store,
        db_manager=db,
        conversation_memory=st.session_state.conv_memory,
        long_term_memory=st.session_state.lt_memory,
        reranker=reranker
    )

if 'user_email' not in st.session_state:
    st.session_state.user_email = None

if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🤖 RAG Bot</h2>", unsafe_allow_html=True)
    st.markdown("### 👤 User Profile")

    user_email = st.text_input(
        "Your Email",
        value=st.session_state.user_email or "",
        placeholder="you@example.com"
    )
    
    if user_email and user_email != st.session_state.user_email:
        st.session_state.user_email = user_email
        st.session_state.agent.set_user(user_email)
        st.session_state.user_logged_in = True
        st.success(f"Welcome!")
    
    if st.session_state.user_logged_in:
        st.markdown(f"✅ Logged in as: {st.session_state.user_email}")
        saved = st.session_state.lt_memory.get_saved_papers(
            st.session_state.agent.current_user_id
        )
        st.markdown(f"📚 Saved papers: {len(saved)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["💬 Chat", "🔍 Search", "🕸️ Knowledge Graph", "📊 Dashboard", "⚙️ Pipeline", "📚 Browse Papers", "📚 My Papers"]
    )
    
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 📈 Stats")
    stats = db.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", stats['total_papers'])
        st.metric("Embedded", stats['papers_with_embeddings'])
    with col2:
        st.metric("Users", stats['total_users'])
        st.metric("Searches", stats['total_searches'])

    with st.expander("📧 Email Settings"):
        smtp_user = st.text_input("SMTP Email", placeholder="your@gmail.com")
        smtp_pass = st.text_input("App Password", type="password")
        
        if smtp_user and smtp_pass:
            st.session_state.agent.set_email_credentials(smtp_user, smtp_pass)
            st.success("Email configured!")

if page == "💬 Chat":
    st.markdown("""
        <div class='hero-container'>
            <h1 class='hero-title'>Research Paper Assistant</h1>
            <p class='hero-subtitle'>Chat with an AI agent to find, understand, and organize research papers. 
            Try: "Find papers on RAG" or "Summarize the first paper"</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💬 Conversation")

    for msg in st.session_state.conv_memory.messages:
        if msg['role'] == 'user':
            st.markdown(f"<div class='chat-message user-message'><b>You:</b> {msg['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'><b>Assistant:</b> {msg['content']}</div>",
                       unsafe_allow_html=True)

    user_input = st.chat_input("Ask about papers... (e.g., 'Find papers on hallucination in LLMs')")

    if user_input:
        st.markdown(f"<div class='chat-message user-message'><b>You:</b> {user_input}</div>",
                   unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(user_input)

        st.markdown(f"<div class='chat-message bot-message'><b>Assistant:</b> {response}</div>",
                   unsafe_allow_html=True)
        st.rerun()

    if st.session_state.conv_memory.current_search_results:
        st.markdown("---")
        st.markdown("### 📄 Current Search Results")

        for i, paper in enumerate(st.session_state.conv_memory.current_search_results, 1):
            with st.expander(f"#{i} - {paper.get('title', 'Untitled')[:80]}..."):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**ArXiv ID:** {paper.get('arxiv_id')}")
                    sim = paper.get('similarity', 0)
                    rerank = paper.get('rerank_score', sim)
                    st.markdown(f"**Embedding Similarity:** {sim:.2%}")
                    if rerank != sim:
                        st.markdown(f"**Rerank Score:** {rerank:.2%}")
                    st.markdown("**Abstract:**")
                    st.markdown(paper.get('abstract', 'No abstract')[:500] + "...")

                with col2:
                    st.markdown(f"[📄 ArXiv](https://arxiv.org/abs/{paper.get('arxiv_id')})")

                    if st.session_state.user_logged_in:
                        if st.button(f"💾 Save", key=f"save_{i}"):
                            st.session_state.lt_memory.save_paper(
                                st.session_state.agent.current_user_id,
                                paper.get('arxiv_id')
                            )
                            st.success("Saved!")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.session_state.conv_memory.current_search_results:
            if st.button("🕸️ View Knowledge Graph"):
                st.session_state.show_graph = True
                st.info("Go to **🕸️ Knowledge Graph** page in the sidebar to see the visualization!")
    with col2:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conv_memory.clear()
            st.rerun()

elif page == "🔍 Search":
    st.markdown("""
        <div class='hero-container'>
            <h1 class='hero-title'>Search Papers</h1>
            <p class='hero-subtitle'>Semantic search with RAG and LLM reranking</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "Search",
            placeholder="Enter your search query...",
            label_visibility="collapsed"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_results = st.number_input("Results", 1, 20, 5)
        with col_b:
            use_reranker = st.checkbox("Use LLM Reranking", value=True)
        with col_c:
            search_btn = st.button("🔍 Search", type="primary")
    
    if query and search_btn:
        with st.spinner("Searching and reranking..."):
            user_interests = []
            if st.session_state.user_logged_in:
                user_interests = st.session_state.lt_memory.get_user_interests(
                    st.session_state.agent.current_user_id
                )

            results = vector_store.semantic_search(
                query,
                n_results=n_results,
                use_reranker=use_reranker,
                user_interests=user_interests
            )

            st.session_state.conv_memory.set_search_results(results, query)

            if st.session_state.user_logged_in:
                st.session_state.lt_memory.add_search(
                    st.session_state.agent.current_user_id,
                    query,
                    len(results)
                )

        if results:
            st.success(f"Found {len(results)} papers" + (" (reranked)" if use_reranker else ""))

            for i, paper in enumerate(results, 1):
                with st.expander(f"#{i} - {paper['title']}", expanded=(i == 1)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                        st.markdown(f"**Embedding Similarity:** {paper['similarity']:.2%}")
                        if paper.get('rerank_score'):
                            st.markdown(f"**Rerank Score:** {paper['rerank_score']:.2%}")
                        
                        st.markdown("**Abstract:**")
                        st.markdown(paper['abstract'])
                        
                        if paper.get('relevant_chunk'):
                            st.markdown("**Most Relevant Section:**")
                            st.info(paper['relevant_chunk'])
                    
                    with col2:
                        st.markdown(f"[📄 ArXiv](https://arxiv.org/abs/{paper['arxiv_id']})")
                        
                        if st.session_state.user_logged_in:
                            if st.button(f"💾 Save", key=f"search_save_{i}"):
                                st.session_state.lt_memory.save_paper(
                                    st.session_state.agent.current_user_id,
                                    paper['arxiv_id']
                                )
                                st.success("Saved!")
        else:
            st.warning("No papers found.")

elif page == "🕸️ Knowledge Graph":
    st.markdown("<h1 style='text-align: center;'>🕸️ Knowledge Graph</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cbd5e1;'>Visualize relationships between papers and concepts</p>", unsafe_allow_html=True)

    if not st.session_state.conv_memory.current_search_results:
        st.info("👆 First, search for papers in the **Chat** or **Search** page to generate a knowledge graph.")

        st.markdown("### Quick Search")
        quick_query = st.text_input("Enter a topic to search:", placeholder="e.g., retrieval augmented generation")

        if st.button("🔍 Search & Generate Graph", type="primary"):
            if quick_query:
                with st.spinner("Searching papers..."):
                    results = vector_store.semantic_search(quick_query, n_results=5)
                    st.session_state.conv_memory.set_search_results(results, quick_query)
                    st.rerun()
    else:
        papers = st.session_state.conv_memory.current_search_results
        topic = st.session_state.conv_memory.current_topic

        st.success(f"📊 Showing graph for **{len(papers)} papers** on topic: **{topic}**")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Papers in graph:** {len(papers)}")
        with col2:
            show_shared = st.checkbox("Highlight shared concepts", value=True)
        with col3:
            if st.button("🔄 Refresh Graph"):
                st.rerun()

        stats = get_graph_stats(papers)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Papers", stats["total_papers"])
        with col2:
            st.metric("Concepts", stats["total_concepts"])
        with col3:
            st.metric("Shared", stats["shared_concepts"])
        with col4:
            st.metric("Unique", stats["unique_concepts"])

        st.markdown("### 📈 Interactive Knowledge Graph")
        st.markdown("*🔵 Blue = Papers | 🟠 Orange = Shared Concepts | 🟢 Green = Unique Concepts*")

        with st.spinner("Generating knowledge graph..."):
            graph_html = create_graph_for_streamlit(papers, graph_type="simple")

        components.html(graph_html, height=650, scrolling=True)

        if stats["shared_list"]:
            st.markdown("### 🔗 Shared Concepts Across Papers")

            shared_concepts = stats["shared_list"]
            concept_details = stats["concept_details"]

            for concept in shared_concepts:
                papers_with_concept = concept_details.get(concept, [])
                with st.expander(f"**{concept}** (in {len(papers_with_concept)} papers)"):
                    for p in papers_with_concept:
                        st.markdown(f"- {p[:80]}...")

        st.markdown("### 📄 Papers & Their Concepts")

        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f'Paper {i}')
            abstract = paper.get('abstract', '')
            from graph_module import extract_simple_concepts
            concepts = extract_simple_concepts(title + " " + abstract)

            with st.expander(f"#{i} - {title[:60]}..."):
                st.markdown(f"**ArXiv ID:** {paper.get('arxiv_id', 'N/A')}")
                st.markdown(f"**Concepts ({len(concepts)}):** {', '.join(concepts[:15])}")

                if st.session_state.user_logged_in:
                    if st.button(f"💾 Save Paper", key=f"graph_save_{i}"):
                        st.session_state.lt_memory.save_paper(
                            st.session_state.agent.current_user_id,
                            paper.get('arxiv_id')
                        )
                        st.success("Saved!")

        st.markdown("---")
        st.download_button(
            label="📥 Download Graph HTML",
            data=graph_html,
            file_name="knowledge_graph.html",
            mime="text/html"
        )

elif page == "📊 Dashboard":
    st.markdown("<h1 style='text-align: center;'>Pipeline Dashboard</h1>", unsafe_allow_html=True)

    stats = db.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Papers", stats['total_papers'])
    with col2:
        st.metric("Processed", stats['processed_papers'])
    with col3:
        st.metric("Embedded", stats['papers_with_embeddings'])
    with col4:
        st.metric("Total Chunks", stats['total_chunks'])

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pipeline_data = {
            'Stage': ['Fetched', 'Processed', 'Embedded'],
            'Count': [
                stats['total_papers'],
                stats['processed_papers'],
                stats['papers_with_embeddings']
            ]
        }
        fig = px.funnel(pipeline_data, y='Stage', x='Count', title="Pipeline Funnel")
        fig.update_layout(
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        user_data = {
            'Metric': ['Users', 'Searches', 'Saved Papers'],
            'Count': [stats['total_users'], stats['total_searches'], stats['total_saved_papers']]
        }
        fig = px.bar(user_data, x='Metric', y='Count', title="User Activity")
        fig.update_layout(
            paper_bgcolor='#1e293b',
            plot_bgcolor='#1e293b',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Pipeline":
    st.markdown("<h1 style='text-align: center;'>Pipeline Control</h1>", unsafe_allow_html=True)

    from orchestrator import PipelineOrchestrator

    @st.cache_resource
    def get_orchestrator():
        return PipelineOrchestrator()

    orchestrator = get_orchestrator()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Run Pipeline")
        days = st.number_input("Days back", 1, 365, 7)
        max_papers = st.number_input("Max papers", 1, 200, 50)
        
        if st.button("🚀 Run Complete Pipeline", type="primary"):
            with st.spinner("Running pipeline..."):
                results = orchestrator.run_complete_pipeline()
                
                if results['status'] == 'SUCCESS':
                    st.success("Pipeline completed successfully!")

                    fetch = results['steps'].get('fetch', {})
                    parse = results['steps'].get('parse', {})
                    embed = results['steps'].get('embeddings', {})
                    summaries = results['steps'].get('summaries', {})

                    # Build results display
                    results_text = f"""
                    **Results:**
                    - Papers fetched: {fetch.get('papers_stored', 0)}
                    - Papers parsed: {parse.get('success', 0)}
                    - Embeddings created: {embed.get('success', 0)}
                    - Embedding API cost: ${embed.get('estimated_cost', 0):.4f}
                    """

                    # Add summary results if not skipped
                    if not summaries.get('skipped', False):
                        results_text += f"""
                    - Summaries generated: {summaries.get('success', 0)}
                    - Summary API cost: ${summaries.get('estimated_cost', 0):.4f}
                    - **Total API cost: ${embed.get('estimated_cost', 0) + summaries.get('estimated_cost', 0):.4f}**
                    """
                    else:
                        results_text += "\n- Summaries: Skipped (disabled or unavailable)"

                    st.markdown(results_text)
                else:
                    st.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
    
    with col2:
        st.markdown("### Individual Steps")
        
        if st.button("📥 Fetch Papers"):
            with st.spinner("Fetching..."):
                r = orchestrator.arxiv_bot.fetch_recent_papers(days, max_papers)
                st.success(f"Fetched {r['papers_stored']} papers")
        
        if st.button("📄 Parse PDFs"):
            with st.spinner("Parsing..."):
                r = orchestrator.pdf_parser.parse_all_unprocessed()
                st.success(f"Parsed {r['success']} papers")
        
        if st.button("🔮 Create Embeddings"):
            with st.spinner("Creating embeddings..."):
                r = orchestrator.vector_store.process_all_papers()
                st.success(f"Embedded {r['success']} papers (${r['estimated_cost']:.4f})")

        if st.button("📝 Generate Summaries"):
            with st.spinner("Generating summaries..."):
                if orchestrator.summarizer_enabled and orchestrator.summarizer:
                    summary_results = orchestrator.summarizer.generate_summaries_batch(limit=max_papers)
                    st.success(f"Generated summaries for {summary_results['success']} papers")
                    st.info(f"API cost: ${summary_results['estimated_cost']:.4f}")
                    if summary_results['failed']:
                        st.warning(f"Failed: {len(summary_results['failed'])} papers")
                else:
                    st.error("Summarizer not available")


elif page == "📚 Browse Papers":
    st.markdown("<h1 style='text-align: center;'>Browse Papers</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cbd5e1;'>Explore your research paper collection</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Import orchestrator here to avoid circular imports
    from orchestrator import PipelineOrchestrator

    @st.cache_resource
    def get_orchestrator():
        return PipelineOrchestrator()

    orchestrator = get_orchestrator()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_processed = st.checkbox("Only processed papers", value=False)
    with col2:
        filter_embedded = st.checkbox("Only with embeddings", value=False)
    with col3:
        filter_summarized = st.checkbox("Only with summaries", value=False)
    with col4:
        sort_order = st.selectbox("Sort by", ["Recent", "Title"])

    papers = orchestrator.get_recent_papers(50)
    filtered_papers = papers
    if filter_processed:
        filtered_papers = [p for p in filtered_papers if p['processed']]
    if filter_embedded:
        filtered_papers = [p for p in filtered_papers if p['has_embeddings']]
    if filter_summarized:
        filtered_papers = [p for p in filtered_papers if p.get('has_summary', False)]

    if sort_order == "Title":
        filtered_papers.sort(key=lambda x: x['title'])

    st.markdown(f"**Showing {len(filtered_papers)} of {len(papers)} papers**")
    st.markdown("<br>", unsafe_allow_html=True)

    for paper in filtered_papers:
        with st.expander(f"{paper['title'][:100]}..."):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                st.markdown(f"**Published:** {paper.get('published_date', 'N/A')}")

                # Check if summary exists
                has_summary = paper.get('has_summary', False)

                if has_summary:
                    # Fetch and display summary
                    summary = orchestrator.db.get_paper_summary(paper['arxiv_id'])
                    if summary:
                        st.markdown("### Structured Summary")

                        # Create tabs for summary sections
                        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Methodology", "Results", "Related Work"])

                        with tab1:
                            abstract = summary.get('abstract_summary', '').strip() if summary.get('abstract_summary') else ''
                            if abstract:
                                st.markdown("**Abstract:**")
                                st.markdown(abstract)
                            else:
                                st.info("No abstract available")

                            if summary.get('date'):
                                st.markdown(f"**Date:** {summary.get('date')}")
                            if summary.get('authors'):
                                st.markdown(f"**Authors:** {summary.get('authors')}")

                        with tab2:
                            methodology = summary.get('methodology', '').strip() if summary.get('methodology') else ''
                            if methodology:
                                st.markdown(methodology)
                            else:
                                st.info("No methodology information available")

                        with tab3:
                            results = summary.get('results', '').strip() if summary.get('results') else ''
                            if results:
                                st.markdown(results)
                            else:
                                st.info("No results information available")

                        with tab4:
                            related_work = summary.get('related_work', '').strip() if summary.get('related_work') else ''
                            if related_work:
                                st.markdown(related_work)
                            else:
                                st.info("No related work information available")

                        if summary.get('structure_score'):
                            st.caption(f"Summary Quality: {summary['structure_score']:.0f}%")

                        # Fallback: if all structured fields are empty, show raw summary
                        all_empty = not any([
                            summary.get('abstract_summary', '').strip(),
                            summary.get('methodology', '').strip(),
                            summary.get('results', '').strip(),
                            summary.get('related_work', '').strip()
                        ])

                        if all_empty and summary.get('raw_summary'):
                            st.warning("Structured sections unavailable - showing raw summary:")
                            st.text_area("Raw Summary", summary['raw_summary'], height=300, disabled=True)
                else:
                    # Show abstract if no summary
                    if paper.get('abstract'):
                        st.markdown("**Abstract:**")
                        st.markdown(paper['abstract'])

            with col2:
                st.markdown("**Status:**")
                if paper['pdf_downloaded']:
                    st.markdown("✅ PDF Downloaded")
                if paper['processed']:
                    st.markdown("✅ Parsed")
                if paper['has_embeddings']:
                    st.markdown("✅ Embeddings")
                if has_summary:
                    st.markdown("✅ Summary")

                st.markdown(f"[📄 View on ArXiv](https://arxiv.org/abs/{paper['arxiv_id']})")

                st.markdown("<br>", unsafe_allow_html=True)

                # Add summary generation button if not yet summarized
                if not has_summary and paper['processed']:
                    if st.button(f"Generate Summary", key=f"gen_{paper['arxiv_id']}"):
                        with st.spinner("Generating summary..."):
                            if orchestrator.summarizer_enabled and orchestrator.summarizer:
                                success = orchestrator.summarizer.generate_summary(paper['arxiv_id'])
                                if success:
                                    st.success("Summary generated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to generate summary")
                            else:
                                st.error("Summarizer not available")
                elif has_summary:
                    if st.button(f"Regenerate", key=f"regen_{paper['arxiv_id']}"):
                        with st.spinner("Regenerating summary..."):
                            if orchestrator.summarizer_enabled and orchestrator.summarizer:
                                success = orchestrator.summarizer.regenerate_summary(paper['arxiv_id'], force=True)
                                if success:
                                    st.success("Summary regenerated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to regenerate summary")
                            else:
                                st.error("Summarizer not available")


elif page == "📚 My Papers":
    st.markdown("<h1 style='text-align: center;'>My Papers</h1>", unsafe_allow_html=True)
    
    if not st.session_state.user_logged_in:
        st.warning("Please enter your email in the sidebar to view your saved papers.")
    else:
        user_id = st.session_state.agent.current_user_id
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["💾 Saved Papers", "🔍 Search History", "📊 My Stats"])
        
        with tab1:
            saved_papers = st.session_state.lt_memory.get_saved_papers(user_id)
            
            if saved_papers:
                st.markdown(f"**{len(saved_papers)} saved papers**")
                
                for paper in saved_papers:
                    with st.expander(f"{paper['title'][:80]}..."):
                        st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                        st.markdown(f"**Saved:** {paper['saved_at']}")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
                        
                        if paper.get('notes'):
                            st.info(f"**Your notes:** {paper['notes']}")
                        
                        # Add note
                        note = st.text_area(
                            "Add/Update note", 
                            value=paper.get('notes', ''),
                            key=f"note_{paper['arxiv_id']}"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Note", key=f"savenote_{paper['arxiv_id']}"):
                                st.session_state.lt_memory.add_note(user_id, paper['arxiv_id'], note)
                                st.success("Note saved!")
                        with col2:
                            if st.button("🗑️ Remove", key=f"remove_{paper['arxiv_id']}"):
                                st.session_state.lt_memory.unsave_paper(user_id, paper['arxiv_id'])
                                st.rerun()
                        
                        st.markdown(f"[📄 View on ArXiv](https://arxiv.org/abs/{paper['arxiv_id']})")
            else:
                st.info("No saved papers yet. Use the chat or search to find and save papers!")
        
        with tab2:
            history = st.session_state.lt_memory.get_search_history(user_id, limit=20)
            
            if history:
                st.markdown("**Recent Searches**")
                
                for search in history:
                    st.markdown(f"- **{search['query']}** ({search['results_count']} results) - {search['timestamp']}")
            else:
                st.info("No search history yet.")
        
        with tab3:
            st.markdown("### Your Activity")
            
            saved_count = len(st.session_state.lt_memory.get_saved_papers(user_id))
            search_count = len(st.session_state.lt_memory.get_search_history(user_id, limit=100))
            frequent = st.session_state.lt_memory.get_frequent_topics(user_id, limit=5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers Saved", saved_count)
            with col2:
                st.metric("Searches Made", search_count)
            
            if frequent:
                st.markdown("**Your Top Topics:**")
                for topic in frequent:
                    st.markdown(f"- {topic}")


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; padding: 1rem; border-top: 1px solid #334155;'>
        <p style='color: #94a3b8;'>RAG Research Bot v2.0 • Agent + Memory + Reranking</p>
    </div>
""", unsafe_allow_html=True)
