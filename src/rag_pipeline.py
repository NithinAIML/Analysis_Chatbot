import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import config
from src.document_processor import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectordb import FAISSManager
from src.retriever import Retriever
from src.memory import ConversationMemory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG pipeline class that coordinates the entire process.
    """
    
    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        faiss_manager: Optional[FAISSManager] = None,
        retriever: Optional[Retriever] = None,
        memory: Optional[ConversationMemory] = None,
        model_name: str = config.GENERATION_MODEL
    ):
        """
        Initialize the RAG pipeline with optional components.
        
        Args:
            document_processor: Document processor for chunking
            embedding_manager: Manager for embeddings
            faiss_manager: FAISS vector database manager
            retriever: Document retriever
            memory: Conversation memory
            model_name: OpenAI model name for generation
        """
        # Initialize components if not provided
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.faiss_manager = faiss_manager or FAISSManager(config.FAISS_INDEX_PATH)
        self.retriever = retriever or Retriever(self.embedding_manager)
        self.memory = memory or ConversationMemory()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.1
        )
    
    def process_documents(self, directory_path: str) -> Dict[str, Any]:
        """
        Process documents in a directory for the RAG pipeline.
        
        Args:
            directory_path: Path to directory containing documents
        
        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()
        logger.info(f"Processing documents in {directory_path}")
        
        # Load and chunk documents
        chunked_documents = self.document_processor.process_directory(directory_path)
        
        # Get chunk statistics
        stats = self.document_processor.get_chunk_statistics(chunked_documents)
        
        # Embed documents
        embedded_documents = self.embedding_manager.embed_documents(chunked_documents)
        
        # Add to FAISS index
        self.faiss_manager.add_documents(embedded_documents)
        
        # Save index
        self.faiss_manager.save_index(config.FAISS_INDEX_PATH)
        
        # Add processing time to stats
        stats["processing_time"] = time.time() - start_time
        stats["num_documents"] = len(chunked_documents)
        
        logger.info(f"Document processing completed in {stats['processing_time']:.2f} seconds")
        return stats
    
    def generate_response(
        self, 
        query: str, 
        top_k: int = config.DEFAULT_TOP_K, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a query using the RAG pipeline.
        
        Args:
            query: The user query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter for retrieval
        
        Returns:
            Dictionary containing the response, retrieved documents, and other info
        """
        start_time = time.time()
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Add user message to memory
        self.memory.add_message("user", query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_documents(
            query, 
            self.faiss_manager, 
            top_k=top_k, 
            filter_metadata=filter_metadata
        )
        
        # Format documents for prompt
        context_str = self._format_documents_for_prompt(retrieved_docs)
        
        # Get conversation history
        conversation_context = self.memory.get_context_string()
        
        # Generate response using LLM
        response = self._generate_llm_response(query, context_str, conversation_context)
        
        # Add assistant message to memory
        self.memory.add_message("assistant", response)
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "retrieved_documents": [(doc.page_content, doc.metadata, score) for doc, score in retrieved_docs],
            "num_docs_retrieved": len(retrieved_docs),
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Response generated in {result['processing_time']:.2f} seconds")
        return result
    
    def _format_documents_for_prompt(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Format retrieved documents for inclusion in the prompt.
        
        Args:
            retrieved_docs: List of (Document, score) tuples
        
        Returns:
            Formatted document context string
        """
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, (doc, score) in enumerate(retrieved_docs):
            # Format each document with metadata
            source_info = f"Source: {doc.metadata.get('source', 'Unknown')}"
            page_info = f"Page: {doc.metadata.get('page', 'Unknown')}"
            relevance_info = f"Relevance: {score:.2f}"
            
            # Add document to context with its reference number for citation
            context_parts.append(
                f"[Document {i+1}] {source_info}, {page_info}, {relevance_info}\n{doc.page_content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _generate_llm_response(self, query: str, context_str: str, conversation_context: str) -> str:
        """
        Generate a response using the LLM with context and conversation history.
        
        Args:
            query: User query
            context_str: Retrieved document context
            conversation_context: Conversation history
        
        Returns:
            Generated response
        """
        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, accurate assistant. Your job is to provide informative, comprehensive responses based on the retrieved documents and the conversation history.

Instructions:
1. Base your answers primarily on the information in the retrieved documents.
2. If the answer isn't in the documents, say so clearly rather than making things up.
3. Use the conversation history for context, but prioritize the most recent query.
4. Use citations to reference the source documents, e.g., [Document 1], [Document 2], etc.
5. Present information in a clear, well-organized manner.
6. Be concise but complete in your answers.
7. If documents contradict each other, note this and explain the different perspectives.

Retrieved Documents:
{context}

Conversation History:
{conversation_history}"""),
            ("human", "{query}")
        ])
        
        # Format the prompt with the inputs
        prompt = prompt_template.format_messages(
            context=context_str,
            conversation_history=conversation_context,
            query=query
        )
        
        # Generate the response
        try:
            raw_response = self.llm.invoke(prompt)
            return raw_response.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."