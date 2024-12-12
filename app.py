import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings  # Changed import
import os
from pinecone import Pinecone
import time

# Setup page config
st.set_page_config(page_title="Stock Search", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    try:
        # Get API key from environment variable
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            st.error("Pinecone API key not found!")
            return None, None
        
        # Initialize embeddings model with specific model name
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
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Initialize components
embeddings, vectorstore = init_components()

# App header
st.title("Stock Search")
st.write("Search for stocks using natural language queries")

# Search interface
query = st.text_input("Enter your search query:", placeholder="e.g., tech companies focused on AI")

if query and vectorstore:
    try:
        with st.spinner("Searching..."):
            # Perform similarity search
            results = vectorstore.similarity_search(
                query,
                k=5  # Number of results to return
            )
            
            if not results:
                st.warning("No matching stocks found.")
            else:
                st.success(f"Found {len(results)} matching stocks:")
                
                # Display results
                for i, doc in enumerate(results, 1):
                    with st.expander(f"🏢 {doc.metadata.get('Name', 'N/A')} ({doc.metadata.get('Ticker', 'N/A')})"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write("**Company Details:**")
                            st.write(f"- Industry: {doc.metadata.get('Industry', 'N/A')}")
                            st.write(f"- Sector: {doc.metadata.get('Sector', 'N/A')}")
                        
                        with col2:
                            st.write("**Location:**")
                            st.write(f"- City: {doc.metadata.get('City', 'N/A')}")
                            st.write(f"- State: {doc.metadata.get('State', 'N/A')}")
                            st.write(f"- Country: {doc.metadata.get('Country', 'N/A')}")
                        
                        st.write("**Business Summary:**")
                        st.write(doc.page_content)
                        
                        st.divider()
                        
    except Exception as e:
        st.error(f"Error performing search: {str(e)}")
else:
    if not vectorstore:
        st.error("Unable to connect to the vector store. Please check your configuration.")
