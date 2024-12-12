import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone
import time

# Setup page config
st.set_page_config(page_title="Stock Search", layout="wide")

# Initialize components
@st.cache_resource
def init_components():
    try:
        # Debug: Check API key
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        st.write("DEBUG: Checking Pinecone API key...")
        if not pinecone_api_key:
            st.error("Pinecone API key not found!")
            return None, None
        st.write("DEBUG: Found Pinecone API key")
        
        # Debug: Initialize Pinecone client
        st.write("DEBUG: Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("stocks")
        stats = index.describe_index_stats()
        st.write(f"DEBUG: Pinecone index stats: {stats}")
        
        # Initialize embeddings model
        st.write("DEBUG: Initializing HuggingFace embeddings...")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        st.write("DEBUG: HuggingFace embeddings initialized")
        
        # Initialize vector store
        st.write("DEBUG: Initializing PineconeVectorStore...")
        vectorstore = PineconeVectorStore(
            index_name="stocks",
            embedding=hf_embeddings,
            pinecone_api_key=pinecone_api_key
        )
        st.write("DEBUG: PineconeVectorStore initialized")
        
        return hf_embeddings, vectorstore
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.write(f"DEBUG: Full error details: {type(e).__name__}: {str(e)}")
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
            st.write(f"DEBUG: Processing query: {query}")
            
            # Get embedding of query for debugging
            query_embedding = embeddings.embed_query(query)
            st.write(f"DEBUG: Generated query embedding of length: {len(query_embedding)}")
            
            # Perform similarity search
            st.write("DEBUG: Performing similarity search...")
            results = vectorstore.similarity_search(
                query,
                k=5  # Number of results to return
            )
            st.write(results)
            st.write(f"DEBUG: Got {len(results)} results back from search")
            
            if not results:
                st.warning("No matching stocks found.")
                st.write("DEBUG: Search returned empty results list")
            else:
                st.success(f"Found {len(results)} matching stocks:")
                
                # Display results
                for i, doc in enumerate(results, 1):
                    st.write(f"DEBUG: Processing result {i}:")
                    st.write(f"DEBUG: Metadata keys: {doc.metadata.keys()}")
                    
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
        st.write(f"DEBUG: Search error details: {type(e).__name__}: {str(e)}")
else:
    if not vectorstore:
        st.error("Unable to connect to the vector store. Please check your configuration.")
        st.write("DEBUG: vectorstore object is None")
