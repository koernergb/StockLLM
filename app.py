import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone
import time
import sys


# Get port from environment variable with default fallback
port = int(os.environ.get("PORT", 10000))

# Configure Streamlit port binding
server_address = "0.0.0.0"
server_port = port


# Immediately print startup message
print("Starting application initialization...")

# Setup page config with try-catch
try:
    st.set_page_config(page_title="Stock Search", layout="wide")
    print("Page config set successfully")
except Exception as e:
    print(f"Error setting page config: {str(e)}")
    sys.exit(1)

# Initialize components with explicit steps
def init_components():
    print("Starting component initialization...")
    
    try:
        # 1. Check Pinecone API key
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            print("ERROR: Pinecone API key not found in environment variables")
            return None, None
        print("Found Pinecone API key")
        
        # 2. Initialize Pinecone client
        print("Initializing Pinecone client...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # 3. Try to access the index
        print("Accessing Pinecone index...")
        index = pc.Index("stocks")
        stats = index.describe_index_stats()
        print(f"Pinecone index stats: {stats}")
        
        # 4. Initialize embeddings
        print("Initializing HuggingFace embeddings...")
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        print("HuggingFace embeddings initialized")
        
        # 5. Initialize vector store
        print("Initializing PineconeVectorStore...")
        vectorstore = PineconeVectorStore(
            index_name="stocks",
            embedding=hf_embeddings,
            pinecone_api_key=pinecone_api_key
        )
        print("PineconeVectorStore initialized successfully")
        
        return hf_embeddings, vectorstore
        
    except Exception as e:
        print(f"ERROR in init_components: {type(e).__name__}: {str(e)}")
        return None, None

# Main app with try-catch blocks
try:
    print("Starting main application...")
    
    # Initialize components
    embeddings, vectorstore = init_components()
    if not embeddings or not vectorstore:
        st.error("Failed to initialize components. Check the application logs.")
        print("ERROR: Component initialization failed")
        st.stop()
    
    # App header
    st.title("Stock Search")
    st.write("Search for stocks using natural language queries")
    print("UI elements initialized")
    
    # Search interface
    query = st.text_input("Enter your search query:", placeholder="e.g., tech companies focused on AI")
    
    if query:
        print(f"Processing query: {query}")
        try:
            with st.spinner("Searching..."):
                results = vectorstore.similarity_search(query, k=5)
                print(f"Search completed, found {len(results)} results")
                
                if not results:
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
            print(f"ERROR during search: {type(e).__name__}: {str(e)}")
            st.error(f"Error during search: {str(e)}")

except Exception as e:
    print(f"FATAL ERROR: {type(e).__name__}: {str(e)}")
    st.error("An error occurred while starting the application")
