# Advanced RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot implementation using LangChain, OpenAI embeddings, and FAISS vector storage. This project implements state-of-the-art techniques for conversational document retrieval.

##Features

- Advanced Document Processing: Smart chunking strategies with optimizable parameters
- Sophisticated Retrieval: Multiple retrieval methods including hybrid search and query expansion
- Post-Retrieval Processing: Re-ranking and context optimization
- Conversation Memory: Long-context understanding through summarization
- Evaluation Framework: Comprehensive metrics for retrieval and generation quality
- Interactive UI: Streamlit-based interface for all operations
- Visualization Tools: Analyze chunk distribution, retrieval metrics, etc.

##Core Components

1.Document Processor: Handles various document formats and implements multiple chunking strategies
2.Embedding Manager: Manages vector embeddings with OpenAI's state-of-the-art models
3.Vector Store: FAISS-based vector database for efficient similarity search
4.Advanced Retriever: Implements various retrieval methods with post-processing
5.Conversation Memory: Manages conversational context with summarization
6.RAG Pipeline: Orchestrates the complete document-to-answer flow
7.Evaluation Framework: Comprehensive metrics for system performance

##Installation

##Prerequisites

- Python 3.9 or higher
- OpenAI API key

##Setup

1.Clone the repository:
"""bash
git clone https://github.com/yourusername/advanced-rag-chatbot.git
cd advanced-rag-chatbot
"""

2.Create a virtual environment:
"""bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
"""

3.Install dependencies:
"""bash
pip install -r requirements.txt
"""

4.Set up your OpenAI API key:
"""bash
export OPENAI_API_KEY=your-api-key
"""
Or create a `.env` file with:
"""
OPENAI_API_KEY=your-api-key
"""

##Usage

##Running the Application

Start the Streamlit application:

"""bash
streamlit run app.py
"""

The application will be available at http://localhost:8501

##Application Workflow

1.Settings: Configure the system parameters and initialize the pipeline
2.Documents: Upload and index your documents
3.Chat: Interact with your documents through the conversational interface
4.Evaluation: Assess the system's performance

##Example: Indexing Documents

"""python
from src.rag_pipeline import RAGPipeline

#Initialize the pipeline
pipeline = RAGPipeline()

#Add documents
pipeline.add_documents([
    "data/document1.pdf",
    "data/document2.docx",
    "data/document3.txt"
])

#Save the pipeline state
pipeline.save_state("data/state")
"""

##Example: Querying

"""python
#Ask a question
response = pipeline.query(
    "What are the key findings in the research?",
    use_memory=True,
    use_hybrid_search=True
)

print(response)
"""

##Chunking Strategies

The system implements multiple chunking strategies, each optimized for different document types:

1.Fixed Chunking: Divides documents into chunks of fixed token size
2.Semantic Chunking: Creates chunks based on semantic boundaries (paragraphs, sections)
3.Paragraph Chunking: Splits documents at paragraph boundaries

You can analyze and optimize chunking parameters through the visualization tools.

##Retrieval Methods

The system supports various retrieval methods:

1.Vector Similarity: Standard similarity search using embeddings
2.Hybrid Search: Combines vector similarity with BM25 keyword matching
3.Query Expansion: Generates multiple search queries to improve recall

Post-retrieval processing includes:

- Document re-ranking
- Content highlighting
- Source diversity optimization
- Token-based filtering

##Evaluation Metrics

The evaluation framework provides comprehensive metrics:

##Retrieval Metrics
- Precision
- Recall
- F1 Score
- Mean Reciprocal Rank (MRR)

##Generation Metrics
- ROUGE scores
- Semantic similarity
- Answer relevancy
- Faithfulness
- Citation accuracy

##Project Structure

"""
advanced_rag_chatbot/
├── README.md                 # Project documentation
├── app.py                    # Streamlit application
├── config.py                 # Configuration settings
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
├── .gitignore                # Git ignore file
├── data/                     # Storage for documents
├── eval/                     # Evaluation scripts and results
│   ├── __init__.py
│   ├── retrieval_eval.py     # Retrieval evaluation metrics
│   └── generation_eval.py    # Generation evaluation metrics
├── src/                      # Source code
│   ├── __init__.py
│   ├── document_processor.py # Document processing and chunking
│   ├── embedding.py          # Embedding models and functions
│   ├── vectordb.py           # FAISS vector store implementation
│   ├── retriever.py          # Retrieval logic including post-processing
│   ├── rag_pipeline.py       # Complete RAG pipeline
│   ├── memory.py             # Conversation memory management
│   └── visualization.py      # Chunk visualization utilities
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_generation.py
"""

##Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##License

This project is licensed under the MIT License - see the LICENSE file for details.

##Acknowledgments

- LangChain for the powerful framework
- OpenAI for the embedding and completion models
- FAISS for efficient vector search

##Citation

If you use this code in your research, please cite:

"""bibtex
@software{analysis_chatbot},
  author = {Nithin Bolishetti},
  title = {Analysis Chatbot},
  year = {2025},
  url = {https://github.com/NithinAIML/Analysis_Chatbot.git}
}
"""