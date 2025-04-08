import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
from src.embedding import EmbeddingManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    """
    Class for retrieving and post-processing documents from the vector database.
    """
    
    def __init__(
        self, 
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        reranking_enabled: bool = config.RERANKING_ENABLED
    ):
        """
        Initialize the Retriever.
        
        Args:
            embedding_manager: The embedding manager for query embedding
            similarity_threshold: Minimum similarity score for retrieved documents
            reranking_enabled: Whether to enable reranking of results
        """
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.reranking_enabled = reranking_enabled
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=1, stop_words='english', 
            ngram_range=(1, 2)
        )
    
    def retrieve_documents(
        self, 
        query: str, 
        db_manager, 
        top_k: int = config.DEFAULT_TOP_K, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents from the vector database based on a query.
        
        Args:
            query: The search query
            db_manager: The vector database manager
            top_k: Number of results to retrieve
            filter_metadata: Optional metadata filter for the search
        
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Embed the query
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Search the vector database
        search_results = db_manager.search(query_embedding, top_k=top_k * 2)  # Get more for post-processing
        
        # Filter by similarity threshold
        filtered_results = [
            (doc, score) for doc, score in search_results 
            if score >= self.similarity_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents above similarity threshold {self.similarity_threshold}")
        
        # Apply metadata filtering if provided
        if filter_metadata:
            filtered_results = self._filter_by_metadata(filtered_results, filter_metadata)
            logger.info(f"After metadata filtering: {len(filtered_results)} documents")
        
        # Apply post-processing
        processed_results = self._post_process_results(query, filtered_results, top_k)
        
        return processed_results
    
    def _filter_by_metadata(
        self, 
        results: List[Tuple[Document, float]], 
        filter_metadata: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """
        Filter results based on metadata criteria.
        
        Args:
            results: List of (Document, score) tuples
            filter_metadata: Dictionary with metadata filters
        
        Returns:
            Filtered list of results
        """
        filtered_results = []
        
        for doc, score in results:
            metadata = doc.metadata
            match = True
            
            for key, value in filter_metadata.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_results.append((doc, score))
        
        return filtered_results
    
    def _post_process_results(
        self, 
        query: str, 
        results: List[Tuple[Document, float]], 
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """
        Apply post-processing to improve retrieval results.
        
        Args:
            query: The original query
            results: List of (Document, score) tuples
            top_k: Number of results to return
        
        Returns:
            Processed list of results
        """
        if not results:
            return []
        if self.reranking_enabled and len(results) > 1:
            results = self._rerank_results(query, results)
        
        # Remove duplicates
        results = self._remove_duplicates(results)
        
        # Limit to top_k
        return results[:top_k]
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        Rerank results based on TF-IDF similarity to enhance semantic matching.
        
        Args:
            query: The original query
            results: List of (Document, score) tuples
        
        Returns:
            Reranked list of results
        """
        if len(results) <= 1:
            return results
            
        # Extract document texts
        texts = [doc.page_content for doc, _ in results]
        
        # Fit TF-IDF vectorizer
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts + [query])
            
            # Get query vector (last row of the matrix)
            query_vector = tfidf_matrix[-1:]
            
            # Get document vectors (all rows except the last one)
            doc_vectors = tfidf_matrix[:-1]
            
            # Calculate TF-IDF similarities
            tfidf_similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Combine vector similarity with TF-IDF similarity
            # with a 70% weight to vector similarity and 30% to TF-IDF
            combined_scores = []
            for i, (doc, vector_sim) in enumerate(results):
                tfidf_sim = tfidf_similarities[i]
                combined_score = 0.7 * vector_sim + 0.3 * tfidf_sim
                combined_scores.append((doc, combined_score))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            return combined_scores
            
        except Exception as e:
            logger.warning(f"Error in TF-IDF reranking: {str(e)}. Falling back to original ranking.")
            return results
    
    def _remove_duplicates(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Remove duplicate or near-duplicate documents from results.
        
        Args:
            results: List of (Document, score) tuples
        
        Returns:
            Deduplicated list of results
        """
        if len(results) <= 1:
            return results
            
        deduplicated = []
        seen_contents = set()
        
        for doc, score in results:
            # Create a simplified representation of the content for deduplication
            # This helps catch near-duplicates by ignoring whitespace and case
            simple_content = ' '.join(doc.page_content.lower().split())
            
            # Skip if we've seen this content before
            if simple_content in seen_contents:
                continue
                
            # Add to deduplicated results and mark as seen
            deduplicated.append((doc, score))
            seen_contents.add(simple_content)
        
        return deduplicated
        