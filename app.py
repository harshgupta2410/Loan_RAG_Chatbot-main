import streamlit as st
from rag_chatbot import load_vector_store, retrieve_context, load_generator, generate_answer
import pandas as pd

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="💬 Loan Approval RAG Chatbot",
    layout="wide",
    page_icon="🤖"
)

# --------------------- Custom CSS ---------------------
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .big-font {
            font-size:24px !important;
            font-weight: 600;
            color: #0E1117;
        }
        .box {
            border: 2px solid #F39C12;
            padding: 15px;
            border-radius: 10px;
            background-color: #FFF8DC;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------- Title ---------------------
st.title("🤖💬 Loan Approval RAG Chatbot")
st.markdown("Ask intelligent questions about the loan approval dataset and get smart, document-aware answers!")

# --------------------- Sidebar ---------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/281/281769.png", width=100)
st.sidebar.title("📊 Dataset Explorer")

with st.sidebar.expander("🔍 Preview Datasets"):
    train_df = pd.read_csv("data/Training Dataset.csv")
    test_df = pd.read_csv("data/Testing Dataset.csv")
    submission_df = pd.read_csv("data/SampleSubmission.csv")

    st.subheader("🧪 Training Data")
    st.dataframe(train_df.head())

    st.subheader("🔬 Testing Data")
    st.dataframe(test_df.head())

    st.subheader("📤 Sample Submission")
    st.dataframe(submission_df.head())

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using FAISS + Hugging Face")

# --------------------- Session State ---------------------
if 'generator' not in st.session_state:
    with st.spinner("🔄 Loading generator..."):
        st.session_state.generator = load_generator()

if 'index' not in st.session_state:
    with st.spinner("📦 Loading vector index..."):
        index, texts, embedder = load_vector_store()
        st.session_state.index = index
        st.session_state.texts = texts
        st.session_state.embedder = embedder

if 'history' not in st.session_state:
    st.session_state.history = []

# --------------------- Main Chat Interface ---------------------
st.markdown("### 💡 Ask a question about the dataset")
query = st.text_input("What do you want to know?", placeholder="e.g. What is the distribution of Loan_Status in training data?", key="input")

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("🚀 Ask")
with col2:
    clear = st.button("🧹 Clear History")

if clear:
    st.session_state.history = []
    st.success("Chat history cleared.")

if ask and query:
    with st.spinner("🤔 Thinking..."):
        context = retrieve_context(query, st.session_state.index, st.session_state.texts, st.session_state.embedder)
        answer = generate_answer(query, context, st.session_state.generator)

        st.session_state.history.append({
            "query": query,
            "context": context,
            "answer": answer
        })

# --------------------- Display Chat History ---------------------
if st.session_state.history:
    st.markdown("## 📝 Answer History")
    for idx, qa in enumerate(reversed(st.session_state.history)):
        st.markdown(f"#### ❓ Question {len(st.session_state.history)-idx}: {qa['query']}")
        st.markdown(f"<div class='box'><b>🤖 Answer:</b><br>{qa['answer']}</div>", unsafe_allow_html=True)

        with st.expander("📄 Show Retrieved Context"):
            st.code(qa['context'], language='text')
        st.markdown("---")
