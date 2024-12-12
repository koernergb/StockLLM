import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup page config
st.set_page_config(page_title="Stock Search", layout="wide")

def check_environment():
    """Check all required environment variables and configurations"""
    logger.debug("Checking environment setup...")
    
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY not found in environment")
        return False
        
    logger.debug("Environment check complete")
    return True

# Initialize components
@st.cache_resource
def init_components():
    try:
        logger.debug("Starting component initialization")
        
        if not check_environment():
            logger.error("Environment check failed")
            return None, None
            
        # Get API key
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        logger.debug("Got Pinecone API key")
        
        # Initialize Pinecone
        try:
            logger.debug("Initializing Pinecone client...")
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index("stocks")
            stats = index.describe_index_stats()
            logger.debug(f"Pinecone index stats: {stats}")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            raise
            
        # Initialize embeddings
        try:
            logger.debug("Initializing HuggingFace embeddings...")
            hf_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            logger.debug("HuggingFace embeddings initialized")
        except Exception as e:
            logger.error(f"HuggingFace initialization error: {str(e)}")
            raise
            
        # Initialize vector store
        try:
            logger.debug("Initializing PineconeVectorStore...")
            vectorstore = PineconeVectorStore(
                index_name="stocks",
                embedding=hf_embeddings,
                pinecone_api_key=pinecone_api_key
            )
            logger.debug("PineconeVectorStore initialized")
        except Exception as e:
            logger.error(f"VectorStore initialization error: {str(e)}")
            raise
            
        return hf_embeddings, vectorstore
        
    except Exception as e:
        logger.error(f"Error in init_components: {type(e).__name__}: {str(e)}")
        return None, None

try:
    logger.debug("Starting app initialization")
    embeddings, vectorstore = init_components()
    
    if not embeddings or not vectorstore:
        st.error("Failed to initialize components. Check the logs for details.")
        sys.exit(1)
        
    # App header
    st.title("Stock Search")
    st.write("Search for stocks using natural language queries")
    
    # Search interface
    query = st.text_input("Enter your search query:", placeholder="e.g., tech companies focused on AI")
    
    if query:
        try:
            logger.debug(f"Processing query: {query}")
            with st.spinner("Searching..."):
                results = vectorstore.similarity_search(query, k=5)
                
                if not results:
                    logger.warning("No results found for query")
                    st.warning("No matching stocks found.")
                else:
                    st.success(f"Found {len(results)} matching stocks")
                    for doc in results:
                        with st.expander(f"🏢 {doc.metadata.get('Name', 'N/A')} ({doc.metadata.get('Ticker', 'N/A')})"):
                            st.write(f"**Industry:** {doc.metadata.get('Industry', 'N/A')}")
                            st.write(f"**Sector:** {doc.metadata.get('Sector', 'N/A')}")
                            st.write(f"**Location:** {doc.metadata.get('City', 'N/A')}, {doc.metadata.get('State', 'N/A')}")
                            st.write("**Business Summary:**")
                            st.write(doc.page_content)
                            
        except Exception as e:
            logger.error(f"Search error: {type(e).__name__}: {str(e)}")
            st.error(f"Error during search: {str(e)}")
            
except Exception as e:
    logger.error(f"Application error: {type(e).__name__}: {str(e)}")
    st.error("An error occurred while starting the application")
