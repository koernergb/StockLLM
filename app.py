# app.py
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone

# Setup page config
st.set_page_config(page_title="Stock Search", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    # Get API key from environment variable
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("Pinecone API key not found in environment variables!")
        return None, None
    
    # Initialize embeddings model
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name="stocks",
        embedding=hf_embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    return hf_embeddings, vectorstore

# Initialize components
embeddings, vectorstore = init_components()

# App header
st.title("Stock Search")
st.write("Search for stocks using natural language queries")

# Search interface
query = st.text_input("Enter your search query:", placeholder="e.g., tech companies focused on AI")

if query and vectorstore:
    with st.spinner("Searching..."):
        # Perform similarity search
        results = vectorstore.similarity_search(
            query,
            k=5  # Number of results to return
        )
        
        # Display results
        for doc in results:
            with st.expander(f"🏢 {doc.metadata['Name']} ({doc.metadata['Ticker']})"):
                st.write(f"**Industry:** {doc.metadata['Industry']}")
                st.write(f"**Sector:** {doc.metadata['Sector']}")
                st.write(f"**Location:** {doc.metadata['City']}, {doc.metadata['State']}, {doc.metadata['Country']}")
                st.write("**Business Summary:**")
                st.write(doc.page_content)
