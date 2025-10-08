import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import CognitronRetriever
from app.llm_handler import generate_answer

# Page config
st.set_page_config(
    page_title="Cognitron-RAG-Engine",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Cognitron-RAG-Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered document Q&A system</p>', unsafe_allow_html=True)

# Load documents
@st.cache_resource
def load_documents():
    data_dir = "data/company_docs"
    docs = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    docs.append(content)
    
    return docs

# Initialize system
docs = load_documents()

if not docs:
    st.warning("⚠️ No documents found in `data/company_docs/`. Please add some .txt files to get started!")
    st.info("**Getting Started:**\n1. Create .txt files with your company documents\n2. Place them in the `data/company_docs/` folder\n3. Restart the application")
else:
    # Initialize Cognitron
    if 'retriever' not in st.session_state:
        with st.spinner("Initializing Cognitron-RAG-Engine..."):
            st.session_state.retriever = CognitronRetriever(docs)
    
    retriever = st.session_state.retriever
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Ask your question:",
            placeholder="e.g., What is the company's leave policy?"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_button = st.button("🔍 Ask Cognitron", type="primary")
    
    # Process question
    if (question and ask_button) or (question and st.session_state.get('auto_submit', False)):
        with st.spinner("🧠 Cognitron is thinking..."):
            # Retrieve relevant context
            relevant_docs, distances = retriever.get_relevant_context(question, top_k=3)
            context = "\n\n".join(relevant_docs)
            
            # Generate answer
            answer = generate_answer(context, question)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Answer")
        st.write(answer)
        
        # Show source context
        with st.expander("📚 Source Context"):
            for i, (doc, distance) in enumerate(zip(relevant_docs, distances)):
                st.markdown(f"**Source {i+1}** (Relevance: {1/(1+distance):.3f})")
                st.text_area(f"Context {i+1}", doc, height=100, key=f"context_{i}")
    
    # Statistics
    st.sidebar.markdown("### 📊 System Stats")
    st.sidebar.metric("Documents Loaded", len(docs))
    st.sidebar.metric("Embedding Model", "all-MiniLM-L6-v2")
    st.sidebar.metric("LLM", "GPT-4")
    
    # Example questions
    st.sidebar.markdown("### 💡 Example Questions")
    example_questions = [
        "What is the leave policy?",
        "How do I request remote work?",
        "Who should I contact for IT support?",
        "What are the office hours?"
    ]
    
    for eq in example_questions:
        if st.sidebar.button(eq):
            st.session_state.auto_submit = True
            st.rerun()
