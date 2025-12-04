import streamlit as st
from datetime import datetime
from orchestrator import PipelineOrchestrator

st.write("App loaded")

st.set_page_config(
    page_title="Medical Research RAG Bot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add consistent styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base styles with improved readability */
    * {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }

    .main {
        padding: 2rem;
        background-color: #ffffff;
        color: #1f2937;
    }

    /* Headings with better contrast */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 600;
    }
    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.4rem !important; }

    /* Paragraphs and text */
    p, div, span, label {
        color: #374151 !important;
        line-height: 1.6;
    }

    /* Text inputs with clear visibility */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px;
        padding: 0.875rem !important;
        font-size: 15px !important;
        color: #1f2937 !important;
    }
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* Chat input */
    .stChatInput>div>div>input {
        background-color: #ffffff !important;
        border: 2px solid #d1d5db !important;
        color: #1f2937 !important;
        font-size: 15px !important;
        padding: 0.875rem !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #f9fafb !important;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 8px;
    }
    [data-testid="stChatMessage"] p {
        color: #1f2937 !important;
    }

    /* Sidebar with better contrast */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a;
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
    [data-testid="stSidebar"] .stTextInput>div>div>input {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 2px solid #cbd5e1 !important;
    }

    /* Buttons with clear contrast */
    .stButton>button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.875rem 1.75rem;
        font-weight: 500;
        font-size: 15px !important;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:disabled {
        background-color: #9ca3af !important;
        cursor: not-allowed;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb !important;
        color: #1f2937 !important;
        font-size: 15px !important;
        border: 1px solid #e5e7eb;
    }
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-top: none;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 1.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 14px !important;
    }

    /* Success/Error/Info messages */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    .stError {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }
    .stInfo {
        background-color: #dbeafe !important;
        color: #1e40af !important;
    }
    .stWarning {
        background-color: #fef3c7 !important;
        color: #92400e !important;
    }

    /* Links */
    a {
        color: #2563eb !important;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_orchestrator():
    return PipelineOrchestrator()

orchestrator = init_orchestrator()

st.title("🧬 Medical Research Paper RAG Bot")
st.markdown(
    "Ask a question about a medical topic (e.g., cancer, diabetes) and get the top 5 recent research papers from arXiv."
)

with st.sidebar:
    st.header("Session")
    user_email = st.text_input("Email (optional, for sending papers):", placeholder="you@example.com")
    st.markdown("---")
    st.write("Top papers and chat history are kept only in this session.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_papers" not in st.session_state:
    st.session_state.last_papers = []

col_chat, col_graph = st.columns([2, 1])

with col_chat:
    st.subheader("Chat with the Bot")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    prompt = st.chat_input("Ask for papers on a medical topic (e.g., 'recent papers on lung cancer treatments')")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Retrieving top 5 relevant papers from arXiv..."):
                try:
                    results = orchestrator.search_papers(prompt, n_results=5)

                    if not results or not results.get("results"):
                        reply = "No relevant papers were found. Please try rephrasing or choosing a different topic."
                        st.write(reply)
                        st.session_state.chat_history.append({"role": "bot", "content": reply})
                        st.session_state.last_papers = []
                    else:
                        papers = results["results"]
                        st.session_state.last_papers = papers

                        reply_lines = [f"Found {len(papers)} relevant papers. Here are the titles:"]
                        for i, p in enumerate(papers, start=1):
                            title = p.get("title", "Untitled")
                            reply_lines.append(f"{i}. {title}")
                        reply_text = "\n".join(reply_lines)
                        st.write(reply_text)
                        st.session_state.chat_history.append({"role": "bot", "content": reply_text})

                        st.markdown("---")
                        st.markdown("### Top 5 Papers")
                        for i, paper in enumerate(papers, start=1):
                            title = paper.get("title", "Untitled")
                            arxiv_id = paper.get("arxiv_id", "N/A")
                            abstract = paper.get("abstract", "No abstract available.")
                            similarity = paper.get("similarity", None)
                            url = paper.get("url") or f"https://arxiv.org/abs/{arxiv_id}"

                            with st.expander(f"{i}. {title}"):
                                st.markdown(f"**arXiv ID:** `{arxiv_id}`")
                                st.markdown(f"[View on arXiv]({url})")
                                if similarity is not None:
                                    st.metric("Similarity score", f"{similarity:.3f}")
                                st.markdown("**Abstract**")
                                st.write(abstract)

                except Exception as e:
                    error_msg = f"Something went wrong while searching: {e}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "bot", "content": error_msg})

with col_graph:
    st.subheader("Knowledge Graph")
    if st.session_state.last_papers:
        st.write("Graph of relationships between the retrieved papers and concepts.")
        st.info("Connect this section to your Graphiti-based visualization.")
    else:
        st.info("Run a search to see the graph of related papers.")

st.markdown("---")
st.subheader("Email these papers")

col_email_left, col_email_right = st.columns([1, 2])

with col_email_left:
    st.write(
        "After you search, you can email a list of the top 5 papers (titles + links) "
        "to yourself for later reading."
    )

with col_email_right:
    can_email = bool(st.session_state.last_papers) and bool(user_email)
    email_button = st.button("Send top 5 papers to my email", disabled=not can_email)

    if email_button:
        try:
            papers = st.session_state.last_papers
            lines = ["Here are the top medical research papers you requested:", ""]
            for i, p in enumerate(papers, start=1):
                title = p.get("title", "Untitled")
                arxiv_id = p.get("arxiv_id", "N/A")
                url = p.get("url") or f"https://arxiv.org/abs/{arxiv_id}"
                lines.append(f"{i}. {title}")
                lines.append(f"   {url}")
                lines.append("")
            lines.append(f"Sent via Medical Research RAG Bot on {datetime.now().isoformat()}")
            email_body = "\n".join(lines)

            st.success(f"Email sent to {user_email}. (Wire this to your SMTP/email function.)")

        except Exception as e:
            st.error(f"Failed to send email: {e}")