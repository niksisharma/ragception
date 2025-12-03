"""
RAG Research Bot - Streamlit UI
Author: Amaan
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from orchestrator import PipelineOrchestrator

# Page configuration
st.set_page_config(
    page_title="RAG Research Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding-top: 2rem;}
    .stButton>button {width: 100%;}
    .paper-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize orchestrator (cached to avoid reloading)
@st.cache_resource
def init_orchestrator():
    return PipelineOrchestrator()

orchestrator = init_orchestrator()

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Research Bot")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Navigation",
        ["🔍 Search Papers", "📊 Dashboard", "⚙️ Pipeline Control", "📚 Browse Papers"]
    )
    
    st.markdown("---")
    
    # Quick Stats
    stats = orchestrator.get_status()
    st.metric("Total Papers", stats['total_papers'])
    st.metric("With Embeddings", stats['papers_with_embeddings'])
    
    # API Cost tracker
    if stats.get('estimated_cost_usd'):
        st.metric("Total API Cost", f"${stats['estimated_cost_usd']:.4f}")

# Main content based on selected page
if page == "🔍 Search Papers":
    st.title("🔍 Search Research Papers")
    st.markdown("Search through the latest RAG and LLM research papers using semantic similarity.")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., hallucination, attention mechanism, RAG..."
        )
    with col2:
        n_results = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    if query:
        with st.spinner("Searching..."):
            results = orchestrator.search_papers(query, n_results)
            
        if results['results']:
            st.success(f"Found {len(results['results'])} relevant papers")
            
            # Display results
            for i, paper in enumerate(results['results'], 1):
                with st.expander(f"{i}. {paper['title']}", expanded=(i <= 3)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                        st.markdown("**Abstract:**")
                        st.write(paper['abstract'])
                        
                        if paper.get('relevant_chunk'):
                            st.markdown("**Most Relevant Section:**")
                            st.info(paper['relevant_chunk'])
                    
                    with col2:
                        # Similarity score visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=paper['similarity'],
                            title={'text': "Similarity"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [None, 1]},
                                   'bar': {'color': "darkblue"},
                                   'steps': [
                                       {'range': [0, 0.5], 'color': "lightgray"},
                                       {'range': [0.5, 0.8], 'color': "lightblue"}],
                                   'threshold': {'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75, 'value': 0.9}}
                        ))
                        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Link to paper
                        st.markdown(f"[📄 View on ArXiv](https://arxiv.org/abs/{paper['arxiv_id']})")
        else:
            st.warning("No relevant papers found. Try different keywords.")

elif page == "📊 Dashboard":
    st.title("📊 Pipeline Dashboard")
    
    # Metrics row
    stats = orchestrator.get_status()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", stats['total_papers'])
    with col2:
        st.metric("Processed", stats['processed_papers'])
    with col3:
        st.metric("With Embeddings", stats['papers_with_embeddings'])
    with col4:
        st.metric("Total Chunks", stats['total_chunks'])
    
    st.markdown("---")
    
    # Pipeline status
    if stats.get('last_run'):
        st.subheader("Last Pipeline Run")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Start Time:** {stats['last_run']['start_time']}")
        with col2:
            st.info(f"**Status:** {stats['last_run']['status']}")
        with col3:
            st.info(f"**Papers Processed:** {stats['last_run']['papers_processed']}")
    
    # Visualizations
    st.markdown("---")
    st.subheader("Pipeline Analytics")
    
    # Create data for visualization
    pipeline_data = {
        'Stage': ['Fetched', 'Downloaded', 'Parsed', 'Embedded'],
        'Count': [
            stats['total_papers'],
            stats.get('processed_papers', 0),
            stats.get('processed_papers', 0),
            stats.get('papers_with_embeddings', 0)
        ]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Funnel chart
        fig = px.funnel(
            pipeline_data,
            y='Stage',
            x='Count',
            title="Pipeline Funnel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart of processing status
        processing_status = {
            'Status': ['Fully Processed', 'Partially Processed', 'Not Processed'],
            'Count': [
                stats.get('papers_with_embeddings', 0),
                stats.get('processed_papers', 0) - stats.get('papers_with_embeddings', 0),
                stats['total_papers'] - stats.get('processed_papers', 0)
            ]
        }
        fig = px.pie(
            processing_status,
            values='Count',
            names='Status',
            title="Processing Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚙️ Pipeline Control":
    st.title("⚙️ Pipeline Control Center")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Run Pipeline")
        
        # Pipeline options
        days_back = st.number_input("Days to look back", min_value=1, max_value=365, value=7)
        max_papers = st.number_input("Max papers to fetch", min_value=1, max_value=200, value=50)
        
        if st.button("🚀 Run Complete Pipeline", type="primary"):
            with st.spinner("Running pipeline... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run pipeline with progress updates
                status_text.text("Step 1/3: Fetching papers from ArXiv...")
                progress_bar.progress(0.33)
                
                results = orchestrator.run_complete_pipeline()
                
                if results['status'] == 'SUCCESS':
                    st.success("✅ Pipeline completed successfully!")
                    
                    # Show results
                    fetch = results['steps'].get('fetch', {})
                    parse = results['steps'].get('parse', {})
                    embed = results['steps'].get('embeddings', {})
                    
                    st.info(f"""
                    **Pipeline Results:**
                    - Papers fetched: {fetch.get('papers_stored', 0)}
                    - Papers parsed: {parse.get('success', 0)}
                    - Embeddings created: {embed.get('success', 0)}
                    - API cost: ${embed.get('estimated_cost', 0):.4f}
                    """)
                else:
                    st.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
                
                progress_bar.progress(1.0)
    
    with col2:
        st.subheader("Individual Steps")
        
        if st.button("📥 Fetch Papers Only"):
            with st.spinner("Fetching papers..."):
                fetch_results = orchestrator.arxiv_bot.fetch_recent_papers(days_back, max_papers)
                st.success(f"Fetched {fetch_results['papers_stored']} papers")
        
        if st.button("📄 Parse PDFs Only"):
            with st.spinner("Parsing PDFs..."):
                parse_results = orchestrator.pdf_parser.parse_all_unprocessed()
                st.success(f"Parsed {parse_results['success']} papers")
        
        if st.button("🔮 Create Embeddings Only"):
            with st.spinner("Creating embeddings..."):
                embed_results = orchestrator.vector_store.process_all_papers()
                st.success(f"Created embeddings for {embed_results['success']} papers")
                st.info(f"API cost: ${embed_results['estimated_cost']:.4f}")

elif page == "📚 Browse Papers":
    st.title("📚 Browse Papers")
    
    # Get recent papers
    papers = orchestrator.get_recent_papers(50)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_processed = st.checkbox("Only processed papers", value=False)
    with col2:
        filter_embedded = st.checkbox("Only with embeddings", value=False)
    with col3:
        sort_order = st.selectbox("Sort by", ["Recent", "Title"])
    
    # Filter papers
    filtered_papers = papers
    if filter_processed:
        filtered_papers = [p for p in filtered_papers if p['processed']]
    if filter_embedded:
        filtered_papers = [p for p in filtered_papers if p['has_embeddings']]
    
    # Sort papers
    if sort_order == "Title":
        filtered_papers.sort(key=lambda x: x['title'])
    
    st.markdown(f"Showing {len(filtered_papers)} papers")
    
    # Display papers in a nice format
    for paper in filtered_papers:
        with st.expander(f"{paper['title'][:100]}..."):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                st.markdown(f"**Published:** {paper.get('published_date', 'N/A')}")
                
                if paper.get('abstract'):
                    st.markdown("**Abstract:**")
                    st.write(paper['abstract'])
            
            with col2:
                # Status indicators
                st.markdown("**Status:**")
                if paper['pdf_downloaded']:
                    st.success("✅ PDF Downloaded")
                if paper['processed']:
                    st.success("✅ Parsed")
                if paper['has_embeddings']:
                    st.success("✅ Embeddings Created")
                
                st.markdown(f"[📄 View on ArXiv](https://arxiv.org/abs/{paper['arxiv_id']})")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>RAG Research Bot v1.0 | Built with Streamlit & OpenAI</div>",
    unsafe_allow_html=True
)