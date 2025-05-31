# StockLLM - Natural Language Stock Search

StockLLM is a web application that allows users to search for stocks using natural language queries. It leverages the power of language models and vector similarity search to find relevant stocks based on user descriptions.

## Features

- Natural language search interface
- Real-time stock information retrieval
- Detailed company information including:
  - Industry and sector classification
  - Business summaries
  - Geographic location
  - Relevance scoring

## Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: HuggingFace (sentence-transformers/all-mpnet-base-v2)
- **Vector Database**: Pinecone
- **Deployment**: Render

## Prerequisites

- Python 3.8+
- Pinecone API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StockLLM.git
cd StockLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Pinecone API key:
   ```PINECONE_API_KEY=your_api_key_here```

## Running Locally

Start the application with:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Deployment

This project is configured for deployment on Render. The `render.yaml` file includes the necessary configuration for automatic deployment.

To deploy:
1. Fork this repository
2. Create a new Web Service on Render
3. Connect your repository
4. Add your `PINECONE_API_KEY` in the environment variables
5. Deploy!

## Usage

1. Enter a natural language query describing the type of stocks you're looking for
2. The system will return the top 5 most relevant matches
3. Click on each result to view detailed company information

Example queries:
- "tech companies focused on AI"
- "renewable energy companies in California"
- "healthcare companies with strong R&D"
