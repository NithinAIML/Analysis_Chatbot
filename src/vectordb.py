import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from langchain.schema import Document
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSManager:
    """
    Class for managing FAISS vector database operations.
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the FAISS vector database manager.
        
        Args:
            index_path: Path to load an existing index from
        """
        self.index = None
        self.documents = []
        self.document_embeddings = []
        self.dimension = 1536  # The dimension of text-embedding-3-large
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            self.init_index()
    
    def init_index(self):
        """
        Initialize a new FAISS index.
        """
        logger.info(f"Initializing new FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.document_embeddings = []
    
    def add_documents(self, embedded_documents: List[Dict[str, Any]]):
        """
        Add documents to the FAISS index.
        
        Args:
            embedded_documents: List of dictionaries containing documents and their embeddings
        """
        if not embedded_documents:
            logger.warning("No documents provided to add to index")
            return
        
        logger.info(f"Adding {len(embedded_documents)} documents to FAISS index")
        
        # Extract embeddings and convert to numpy array
        embeddings = [doc["embedding"] for doc in embedded_documents]
        embeddings_array = np.array(embeddings).astype("float32")
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store documents and embeddings
        for doc in embedded_documents:
            self.documents.append(doc["document"])
            self.document_embeddings.append(doc["embedding"])
        
        logger.info(f"FAISS index now contains {len(self.documents)} documents")
    
    def search(self, query_embedding: List[float], top_k: int = config.DEFAULT_TOP_K) -> List[Tuple[Document, float]]:
        """
        Search the FAISS index for similar documents.
        
        Args:
            query_embedding: The query embedding
            top_k: Number of results to return
        
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("Cannot search: FAISS index is empty")
            return []
        
        logger.info(f"Searching FAISS index for top {top_k} results")
        
        # Convert query embedding to numpy array
        query_array = np.array([query_embedding]).astype("float32")
        
        # Search index
        distances, indices = self.index.search(query_array, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            # Convert distance to similarity score (FAISS returns L2 distance)
            # Smaller L2 distance = more similar, so we invert
            distance = distances[0][i]
            max_distance = 16  # Empirically determined for text-embedding-3-large
            similarity = max(0, 1 - (distance / max_distance))
            
            # Add to results
            results.append((self.documents[idx], similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def save_index(self, index_path: str):
        """
        Save the FAISS index and related data to disk.
        
        Args:
            index_path: Path to save the index to
        """
        if not self.index:
            logger.warning("Cannot save: FAISS index not initialized")
            return
        
        logger.info(f"Saving FAISS index to {index_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # Save documents and embeddings
        with open(f"{index_path}.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "document_embeddings": self.document_embeddings
            }, f)
        
        logger.info(f"Successfully saved FAISS index with {len(self.documents)} documents")
    
    def load_index(self, index_path: str):
        """
        Load a FAISS index and related data from disk.
        
        Args:
            index_path: Path to load the index from
        """
        logger.info(f"Loading FAISS index from {index_path}")
        
        # Check if files exist
        if not os.path.exists(f"{index_path}.faiss") or not os.path.exists(f"{index_path}.pkl"):
            logger.error(f"Cannot load: FAISS index files not found at {index_path}")
            self.init_index()
            return
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # Load documents and embeddings
            with open(f"{index_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.document_embeddings = data["document_embeddings"]
            
            logger.info(f"Successfully loaded FAISS index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            self.init_index()