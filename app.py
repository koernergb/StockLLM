import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone

# Setup page config
st.set_page_config(page_title="Stock Search", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            st.error("Pinecone API key not found!")
            return None, None
        
        # Initialize embeddings
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize Pinecone directly
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("stocks")
        
        return hf_embeddings, index
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None

# Initialize components
embeddings, pinecone_index = init_components()

# App header
st.title("Stock Search")
st.write("Search for stocks using natural language queries")

# Search interface
query = st.text_input("Enter your search query:", placeholder="e.g., tech companies focused on AI")

if query and pinecone_index:
    try:
        with st.spinner("Searching..."):
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Query Pinecone directly
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace="stock-descriptions"
            )
            
            if not results.matches:
                st.warning("No matching stocks found.")
            else:
                st.success(f"Found {len(results.matches)} matching stocks:")
                
                # Display results
                for match in results.matches:
                    metadata = match.metadata
                    with st.expander(f"🏢 {metadata.get('Name', 'N/A')} ({metadata.get('Ticker', 'N/A')})"):
                        st.write(f"**Industry:** {metadata.get('Industry', 'N/A')}")
                        st.write(f"**Sector:** {metadata.get('Sector', 'N/A')}")
                        st.write(f"**Location:** {metadata.get('City', 'N/A')}, {metadata.get('State', 'N/A')}")
                        st.write("**Business Summary:**")
                        st.write(metadata.get('Business Summary', 'N/A'))
                        st.write(f"**Score:** {match.score:.3f}")
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
