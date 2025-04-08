import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional
import config
from src.document_processor import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectordb import FAISSManager
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline
from src.memory import ConversationMemory
from src.visualization import ChunkVisualizer
from eval.retrieval_eval import RetrievalEvaluator
from eval.generation_eval import GenerationEvaluator
from langchain.schema import Document

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem !important;
#         color: #7E56C2 !important;
#     }
#     .subheader {
#         font-size: 1.5rem !important;
#         color: #4A4A4A !important;
#     }
#     .info-box {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#     }
#     .stButton button {
#         background-color: #7E56C2;
#         color: white;
#     }
#     .upload-box {
#         border: 2px dashed #7E56C2;
#         border-radius: 0.5rem;
#         padding: 1rem;
#         text-align: center;
#     }
#     .chat-message {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 0.5rem;
#     }
#     .user-message {
#         background-color: #E8EAF6;
#         border-left: 5px solid #7E56C2;
#     }
#     .bot-message {
#         background-color: #F5F5F5;
#         border-left: 5px solid #64B5F6;
#     }
#     .citation {
#         background-color: #FFF9C4;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.3rem;
#         font-size: 0.8rem;
#     }
# </style>
# """, unsafe_allow_html=True)
# Modify the custom CSS section at the beginning of app.py

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #7E56C2 !important;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #4A4A4A !important;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton button {
        background-color: #7E56C2;
        color: white;
    }
    .upload-box {
        border: 2px dashed #7E56C2;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        color: #333333;  /* Darker text color for better contrast */
    }
    .user-message {
        background-color: #E8EAF6;
        border-left: 5px solid #7E56C2;
        color: #333333;  /* Dark text color for user messages */
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #64B5F6;
        color: #333333;  /* Dark text color for bot messages */
    }
    .citation {
        background-color: #FFF9C4;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.processing_complete = False
    st.session_state.document_stats = {}
    st.session_state.pipeline = None
    st.session_state.active_tab = "Chat"
    st.session_state.chunk_size = config.DEFAULT_CHUNK_SIZE
    st.session_state.chunk_overlap = config.DEFAULT_CHUNK_OVERLAP
    st.session_state.top_k = config.DEFAULT_TOP_K
    st.session_state.chunk_df = pd.DataFrame()
    st.session_state.last_query = ""
    st.session_state.last_response = {}

def initialize_rag_pipeline():
    """
    Initialize the RAG pipeline components.
    """
    st.session_state.document_processor = DocumentProcessor(
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap
    )
    st.session_state.embedding_manager = EmbeddingManager()
    st.session_state.faiss_manager = FAISSManager(config.FAISS_INDEX_PATH)
    st.session_state.retriever = Retriever(st.session_state.embedding_manager)
    st.session_state.memory = ConversationMemory()
    
    st.session_state.pipeline = RAGPipeline(
        document_processor=st.session_state.document_processor,
        embedding_manager=st.session_state.embedding_manager,
        faiss_manager=st.session_state.faiss_manager,
        retriever=st.session_state.retriever,
        memory=st.session_state.memory
    )
    
    st.session_state.chunk_visualizer = ChunkVisualizer()
    st.session_state.retrieval_evaluator = RetrievalEvaluator()
    st.session_state.generation_evaluator = GenerationEvaluator()
    
    st.session_state.initialized = True

# def display_chat_history():
#     """
#     Display the chat history.
#     """
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
#         else:
#             # Extract any citations from the message content
#             content = message["content"]
            
#             # Display the message
#             st.markdown(f'<div class="chat-message bot-message">{content}</div>', unsafe_allow_html=True)
def display_chat_history():
    """
    Display the chat history.
    """
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Extract any citations from the message content
            content = message["content"]
            
            # Display the message
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)

def process_documents():
    """
    Process the uploaded documents.
    """
    with st.spinner("Processing documents..."):
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # Process documents
        st.session_state.document_stats = st.session_state.pipeline.process_documents(config.DATA_DIR)
        
        # Get chunk distribution data
        chunked_docs = st.session_state.document_processor.process_directory(config.DATA_DIR)
        st.session_state.chunk_df = st.session_state.document_processor.analyze_chunk_distribution(chunked_docs)
        
        st.session_state.processing_complete = True
        st.success(f"Successfully processed {st.session_state.document_stats['num_documents']} documents!")

def handle_user_input():
    """
    Handle user input from the chat interface.
    """
    user_input = st.session_state.user_input
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Clear input box
        st.session_state.user_input = ""
        
        with st.spinner("Thinking..."):
            # Generate response
            response_data = st.session_state.pipeline.generate_response(
                user_input,
                top_k=st.session_state.top_k
            )
            
            # Store query and response for evaluation
            st.session_state.last_query = user_input
            st.session_state.last_response = response_data
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response_data["response"]})

def display_sidebar():
    """
    Display the sidebar with controls.
    """
    st.sidebar.markdown('<p class="subheader">Settings</p>', unsafe_allow_html=True)
    
    # Document Processing Settings
    st.sidebar.markdown("### Document Processing")
    
    # Chunk Size slider
    new_chunk_size = st.sidebar.slider(
        "Chunk Size (characters)",
        min_value=config.MIN_CHUNK_SIZE,
        max_value=config.MAX_CHUNK_SIZE,
        value=st.session_state.chunk_size,
        step=config.CHUNK_SIZE_STEP
    )
    
    # Chunk Overlap slider
    new_chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (characters)",
        min_value=config.MIN_CHUNK_OVERLAP,
        max_value=config.MAX_CHUNK_OVERLAP,
        value=st.session_state.chunk_overlap,
        step=config.CHUNK_OVERLAP_STEP
    )
    
    # TopK slider
    new_top_k = st.sidebar.slider(
        "Number of Retrieved Documents (K)",
        min_value=1,
        max_value=config.MAX_TOP_K,
        value=st.session_state.top_k,
        step=1
    )
    
    # Update values if changed
    if new_chunk_size != st.session_state.chunk_size or new_chunk_overlap != st.session_state.chunk_overlap:
        st.session_state.chunk_size = new_chunk_size
        st.session_state.chunk_overlap = new_chunk_overlap
        
        # Reinitialize document processor with new parameters
        st.session_state.document_processor = DocumentProcessor(
            chunk_size=new_chunk_size,
            chunk_overlap=new_chunk_overlap
        )
        
        if st.session_state.pipeline:
            st.session_state.pipeline.document_processor = st.session_state.document_processor
    
    if new_top_k != st.session_state.top_k:
        st.session_state.top_k = new_top_k
    
    # Document Upload and Processing
    st.sidebar.markdown("### Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        # Save uploaded files to data directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(config.DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved: {uploaded_file.name}")
    
    # Process Documents button
    if st.sidebar.button("Process Documents"):
        process_documents()
    
    # Reset Chat button
    if st.sidebar.button("Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.sidebar.success("Chat history has been reset!")
    
    # Display processing info
    if st.session_state.processing_complete:
        st.sidebar.markdown("### Processing Info")
        st.sidebar.info(f"""
        - Documents Processed: {st.session_state.document_stats.get('num_documents', 0)}
        - Total Chunks: {st.session_state.document_stats.get('total_chunks', 0)}
        - Avg. Chunk Size: {st.session_state.document_stats.get('avg_chunk_size', 0):.1f} chars
        - Avg. Tokens per Chunk: {st.session_state.document_stats.get('avg_tokens', 0):.1f}
        - Processing Time: {st.session_state.document_stats.get('processing_time', 0):.2f} seconds
        """)

def display_chat_tab():
    """
    Display the chat interface tab.
    """
    st.markdown('<p class="subheader">Chat with your documents</p>', unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat_history()
        
        # Chat input
        st.text_input(
            "Ask a question about your documents:",
            key="user_input",
            on_change=handle_user_input
        )
    
    # Display retrieved documents if available
    if st.session_state.last_response and "retrieved_documents" in st.session_state.last_response:
        with st.expander("View Retrieved Documents", expanded=False):
            docs = st.session_state.last_response["retrieved_documents"]
            
            for i, (content, metadata, score) in enumerate(docs):
                st.markdown(f"### Document {i+1} (Score: {score:.3f})")
                st.markdown(f"**Source:** {metadata.get('source', 'Unknown')}, **Page:** {metadata.get('page', 'Unknown')}")
                st.markdown(f"```\n{content[:500]}{'...' if len(content) > 500 else ''}\n```")
                st.markdown("---")

def display_visualization_tab():
    """
    Display the visualization tab.
    """
    st.markdown('<p class="subheader">Document Chunking Visualization</p>', unsafe_allow_html=True)
    
    if not st.session_state.processing_complete:
        st.info("Please process documents first to see visualizations.")
        return
    
    # Chunk Distribution
    st.markdown("### Chunk Size Distribution")
    fig1 = st.session_state.chunk_visualizer.visualize_chunk_distribution(st.session_state.chunk_df)
    st.pyplot(fig1)
    
    # Token Distribution
    st.markdown("### Token Count Distribution")
    fig2 = st.session_state.chunk_visualizer.visualize_token_distribution(st.session_state.chunk_df)
    st.pyplot(fig2)
    
    # Chunk Overlap Visualization
    st.markdown("### Chunk Overlap Visualization")
    fig3 = st.session_state.chunk_visualizer.visualize_chunk_overlap(
        chunk_size=st.session_state.chunk_size,
        overlap_values=[0, int(st.session_state.chunk_size * 0.1), 
                      st.session_state.chunk_overlap, 
                      int(st.session_state.chunk_size * 0.4)]
    )
    st.pyplot(fig3)
    
    # If there's a recent retrieval, show it
    if st.session_state.last_response and "retrieved_documents" in st.session_state.last_response:
        st.markdown("### Recent Retrieval Results")
        
        # Extract data for visualization
        query = st.session_state.last_query
        
        # Convert the retrieved_documents to the format expected by visualize_retrieval_results
        # The format needs to be a list of (Document, score) tuples
        retrieved_docs = []
        for doc_content, doc_metadata, score in st.session_state.last_response["retrieved_documents"]:
            # Create a Document object from the content and metadata
            doc = Document(page_content=doc_content, metadata=doc_metadata)
            # Add the (Document, score) tuple to the list
            retrieved_docs.append((doc, score))
        
        # Now we can safely visualize the results
        if retrieved_docs:  # Make sure we have some results
            fig4 = st.session_state.chunk_visualizer.visualize_retrieval_results(query, retrieved_docs)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No retrieval results to visualize.")
# def display_visualization_tab():
#     """
#     Display the visualization tab.
#     """
#     st.markdown('<p class="subheader">Document Chunking Visualization</p>', unsafe_allow_html=True)
    
#     if not st.session_state.processing_complete:
#         st.info("Please process documents first to see visualizations.")
#         return
    
#     # Chunk Distribution
#     st.markdown("### Chunk Size Distribution")
#     fig1 = st.session_state.chunk_visualizer.visualize_chunk_distribution(st.session_state.chunk_df)
#     st.pyplot(fig1)

#     # Token Distribution
#     st.markdown("### Token Count Distribution")
#     fig2 = st.session_state.chunk_visualizer.visualize_token_distribution(st.session_state.chunk_df)
#     st.pyplot(fig2)
    
#     # Chunk Overlap Visualization
#     st.markdown("### Chunk Overlap Visualization")
#     fig3 = st.session_state.chunk_visualizer.visualize_chunk_overlap(
#         chunk_size=st.session_state.chunk_size,
#         overlap_values=[0, int(st.session_state.chunk_size * 0.1), 
#                       st.session_state.chunk_overlap, 
#                       int(st.session_state.chunk_size * 0.4)]
#     )
#     st.pyplot(fig3)
    
#     # If there's a recent retrieval, show it
#     if st.session_state.last_response and "retrieved_documents" in st.session_state.last_response:
#         st.markdown("### Recent Retrieval Results")
        
#         # Extract data for visualization
#         query = st.session_state.last_query
#         retrieved_docs = [(doc, score) for doc, _, score in st.session_state.last_response["retrieved_documents"]]
        
#         # Visualize retrieval results
#         fig4 = st.session_state.chunk_visualizer.visualize_retrieval_results(query, retrieved_docs)
#         st.plotly_chart(fig4, use_container_width=True)

def display_evaluation_tab():
    """
    Display the evaluation tab.
    """
    st.markdown('<p class="subheader">Retrieval and Generation Evaluation</p>', unsafe_allow_html=True)
    
    # Check if documents have been processed
    if not st.session_state.processing_complete:
        st.info("Please process documents first to enable evaluation.")
        return
    
    # Retrieval Evaluation Section
    st.markdown("### Retrieval Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sample Query and Expected Documents")
        
        eval_query = st.text_input(
            "Enter a sample query for evaluation:",
            value="What is HealthChoice Illinois?"
        )
        
        st.markdown("Select relevant document IDs for the query:")
        
        # Show a sample of the available documents
        if not st.session_state.chunk_df.empty:
            doc_df = st.session_state.chunk_df.copy()
            doc_df['document_id'] = doc_df.apply(
                lambda row: f"{row['source']}-{row['chunk_id']}", axis=1
            )
            
            # Display a selection of documents for relevance marking
            doc_sample = doc_df.sample(min(10, len(doc_df)))
            
            selected_docs = []
            # Use a unique key for each checkbox by including the index i
            for i, (_, row) in enumerate(doc_sample.iterrows()):
                unique_key = f"doc_{i}_{row['document_id']}"  # Make sure the key is unique
                if st.checkbox(f"{row['document_id']} (Page {row['page']})", key=unique_key):
                    selected_docs.append(row['document_id'])
        else:
            st.warning("No documents available for evaluation.")
            selected_docs = []
    
    with col2:
        st.markdown("#### Evaluation Results")
        
        if st.button("Run Retrieval Evaluation"):
            with st.spinner("Evaluating retrieval..."):
                # Run retrieval evaluation for the sample query
                retrieved_docs = st.session_state.retriever.retrieve_documents(
                    eval_query, st.session_state.faiss_manager, top_k=st.session_state.top_k
                )
                
                # Evaluate retrieval
                eval_metrics = st.session_state.retrieval_evaluator.evaluate_retrieval(
                    eval_query, retrieved_docs, selected_docs
                )
                
                # Display metrics
                st.markdown("Retrieval Metrics:")
                st.json(eval_metrics)
                
                # Display retrieved documents
                st.markdown("Retrieved Documents:")
                for i, (doc, score) in enumerate(retrieved_docs):
                    doc_id = f"{doc.metadata.get('source', 'Unknown')}-{doc.metadata.get('chunk_id', i)}"
                    st.markdown(f"**Document {i+1} ({doc_id})**: Score = {score:.3f}")
                    st.markdown(f"```\n{doc.page_content[:200]}...\n```")
    
    # Generation Evaluation Section
    st.markdown("### Generation Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sample Generation")
        
        gen_query = st.text_input(
            "Enter a query for generation evaluation:",
            value="How do I get my medical ID card?"
        )
        
        reference_response = st.text_area(
            "Enter a reference (expected) response:",
            value="You'll get a Member ID card within five days of enrolling with Aetna Better Health of Illinois. You should always carry your card with you. It has important phone numbers. You'll need to show it when you get any services, too."
        )
    
    with col2:
        st.markdown("#### Evaluation Results")
        
        if st.button("Run Generation Evaluation"):
            with st.spinner("Generating and evaluating response..."):
                # Generate response
                response_data = st.session_state.pipeline.generate_response(
                    gen_query, top_k=st.session_state.top_k
                )
                
                generated_response = response_data["response"]
                
                # Display generated response
                st.markdown("Generated Response:")
                st.markdown(f"```\n{generated_response}\n```")
                
                # Evaluate generation
                eval_metrics = st.session_state.generation_evaluator.evaluate_response(
                    generated_response, reference_response
                )
                
                # Display metrics
                st.markdown("Generation Metrics:")
                st.json(eval_metrics)


# def display_evaluation_tab():
#     """
#     Display the evaluation tab.
#     """
#     st.markdown('<p class="subheader">Retrieval and Generation Evaluation</p>', unsafe_allow_html=True)
    
#     # Check if documents have been processed
#     if not st.session_state.processing_complete:
#         st.info("Please process documents first to enable evaluation.")
#         return
    
#     # Retrieval Evaluation Section
#     st.markdown("### Retrieval Evaluation")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("#### Sample Query and Expected Documents")
        
#         eval_query = st.text_input(
#             "Enter a sample query for evaluation:",
#             value="What is HealthChoice Illinois?"
#         )
        
#         st.markdown("Select relevant document IDs for the query:")
        
#         # Show a sample of the available documents
#         if not st.session_state.chunk_df.empty:
#             doc_df = st.session_state.chunk_df.copy()
#             doc_df['document_id'] = doc_df.apply(
#                 lambda row: f"{row['source']}-{row['chunk_id']}", axis=1
#             )
            
#             # Display a selection of documents for relevance marking
#             doc_sample = doc_df.sample(min(10, len(doc_df)))
            
#             selected_docs = []
#             for _, row in doc_sample.iterrows():
#                 if st.checkbox(f"{row['document_id']} (Page {row['page']})", key=f"doc_{row['document_id']}_{i}"):
#                     selected_docs.append(row['document_id'])
#         else:
#             st.warning("No documents available for evaluation.")
#             selected_docs = []
    
#     with col2:
#         st.markdown("#### Evaluation Results")
        
#         if st.button("Run Retrieval Evaluation"):
#             with st.spinner("Evaluating retrieval..."):
#                 # Run retrieval evaluation for the sample query
#                 retrieved_docs = st.session_state.retriever.retrieve_documents(
#                     eval_query, st.session_state.faiss_manager, top_k=st.session_state.top_k
#                 )
                
#                 # Evaluate retrieval
#                 eval_metrics = st.session_state.retrieval_evaluator.evaluate_retrieval(
#                     eval_query, retrieved_docs, selected_docs
#                 )
                
#                 # Display metrics
#                 st.markdown("Retrieval Metrics:")
#                 st.json(eval_metrics)
                
#                 # Display retrieved documents
#                 st.markdown("Retrieved Documents:")
#                 for i, (doc, score) in enumerate(retrieved_docs):
#                     doc_id = f"{doc.metadata.get('source', 'Unknown')}-{doc.metadata.get('chunk_id', i)}"
#                     st.markdown(f"**Document {i+1} ({doc_id})**: Score = {score:.3f}")
#                     st.markdown(f"```\n{doc.page_content[:200]}...\n```")
    
#     # Generation Evaluation Section
#     st.markdown("### Generation Evaluation")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("#### Sample Generation")
        
#         gen_query = st.text_input(
#             "Enter a query for generation evaluation:",
#             value="How do I get my medical ID card?"
#         )
        
#         reference_response = st.text_area(
#             "Enter a reference (expected) response:",
#             value="You'll get a Member ID card within five days of enrolling with Aetna Better Health of Illinois. You should always carry your card with you. It has important phone numbers. You'll need to show it when you get any services, too."
#         )
    
#     with col2:
#         st.markdown("#### Evaluation Results")
        
#         if st.button("Run Generation Evaluation"):
#             with st.spinner("Generating and evaluating response..."):
#                 # Generate response
#                 response_data = st.session_state.pipeline.generate_response(
#                     gen_query, top_k=st.session_state.top_k
#                 )
                
#                 generated_response = response_data["response"]
                
#                 # Display generated response
#                 st.markdown("Generated Response:")
#                 st.markdown(f"```\n{generated_response}\n```")
                
#                 # Evaluate generation
#                 eval_metrics = st.session_state.generation_evaluator.evaluate_response(
#                     generated_response, reference_response
#                 )
                
#                 # Display metrics
#                 st.markdown("Generation Metrics:")
#                 st.json(eval_metrics)

def display_about_tab():
    """
    Display the about tab with information about the RAG chatbot.
    """
    st.markdown('<p class="subheader">About this Analysis Chatbot</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    
    This advanced RAG (Retrieval-Augmented Generation) chatbot is designed to provide accurate and contextually relevant responses based on your documents. It combines the power of retrieval-based and generation-based approaches to deliver high-quality answers.
    
    ### Key Features
    
    - **Optimized Document Processing**: Implements advanced chunking strategies with visualizations for chunk size and overlap optimization
    - **Conversation Memory**: Maintains chat history to provide contextually relevant responses
    - **FAISS Vector Database**: Fast and efficient similarity search for document retrieval
    - **Advanced Retrieval**: Includes post-retrieval processing for improved relevance
    - **OpenAI Integration**: Uses state-of-the-art embedding and text generation models
    - **Comprehensive Evaluation**: Metrics for both retrieval and generation performance
    
    ### How It Works
    
    1. **Document Processing**: Documents are loaded, chunked into manageable pieces, and stored in the system
    2. **Embedding**: Document chunks are embedded using OpenAI's text-embedding-3-large model
    3. **Retrieval**: When you ask a question, the system retrieves the most relevant chunks from your documents
    4. **Generation**: The retrieved information and your question are used to generate a comprehensive response
    5. **Continuous Learning**: The system maintains conversation context to provide coherent responses
    
    ### Technologies Used
    
    - **LangChain**: Framework for building LLM applications
    - **OpenAI**: State-of-the-art embedding and text generation models
    - **FAISS**: Efficient similarity search library
    - **Streamlit**: Interactive web interface
    - **Matplotlib, Plotly**: Visualization libraries
    
    ### Usage Tips
    
    - Upload and process documents to build your knowledge base
    - Ask questions about the content of your documents
    - Adjust parameters in the sidebar to optimize performance
    - Use the visualization and evaluation tabs to understand and improve the system
    """)

def main():
    """
    Main function for the Streamlit app.
    """
    # Display header
    st.markdown('<p class="main-header">RAG Q&A Chatbot</p>', unsafe_allow_html=True)
    
    # Initialize pipeline if not already initialized
    if not st.session_state.initialized:
        initialize_rag_pipeline()
    
    # Display sidebar
    display_sidebar()
    
    # Display main content with tabs
    tabs = st.tabs(["Chat", "Visualizations", "Evaluation", "About"])
    
    with tabs[0]:
        display_chat_tab()
    
    with tabs[1]:
        display_visualization_tab()
    
    with tabs[2]:
        display_evaluation_tab()
    
    with tabs[3]:
        display_about_tab()

if __name__ == "__main__":
    main()