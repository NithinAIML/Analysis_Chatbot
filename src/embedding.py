import logging
import numpy as np
from typing import List, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Class for managing document and query embeddings.
    """
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        Initialize the EmbeddingManager with a specified model.
        
        Args:
            model_name: The name of the OpenAI embedding model to use
        """
        logger.info(f"Initializing EmbeddingManager with model: {model_name}")
        
        self.model_name = model_name
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=config.OPENAI_API_KEY,
            dimensions=1536  # Explicitly set dimensions for text-embedding-3-large
        )
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of Document objects to embed
        
        Returns:
            List of dictionaries containing the document, its content, metadata, and embedding
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return []
        
        logger.info(f"Embedding {len(documents)} documents with model {self.model_name}")
        
        try:
            # Get text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Combine documents with their embeddings
            embedded_documents = []
            for i, doc in enumerate(documents):
                embedded_documents.append({
                    "document": doc,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embeddings[i]
                })
            
            logger.info(f"Successfully embedded {len(embedded_documents)} documents")
            return embedded_documents
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: The query string to embed
        
        Returns:
            The query embedding as a list of floats
        """
        logger.info(f"Embedding query: {query[:50]}...")
        
        try:
            query_embedding = self.embedding_model.embed_query(query)
            return query_embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def compute_similarity(self, query_embedding: List[float], document_embedding: List[float]) -> float:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: The query embedding
            document_embedding: The document embedding
        
        Returns:
            Cosine similarity score (higher is more similar)
        """
        # Convert to numpy arrays for efficient computation
        query_array = np.array(query_embedding)
        doc_array = np.array(document_embedding)
        
        # Compute dot product
        dot_product = np.dot(query_array, doc_array)
        
        # Compute magnitudes
        query_magnitude = np.linalg.norm(query_array)
        doc_magnitude = np.linalg.norm(doc_array)
        
        # Compute cosine similarity
        if query_magnitude > 0 and doc_magnitude > 0:
            similarity = dot_product / (query_magnitude * doc_magnitude)
            return float(similarity)
        else:
            return 0.0