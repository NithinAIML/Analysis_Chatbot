import unittest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from src.retriever import Retriever
from src.embedding import EmbeddingManager
import numpy as np

class TestRetrieval(unittest.TestCase):
    """
    Test cases for document retrieval functionality.
    """
    
    def setUp(self):
        """Set up common test resources."""
        # Create mock embedding manager
        self.mock_embedding_manager = MagicMock(spec=EmbeddingManager)
        self.mock_embedding_manager.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Create retriever with mock embedding manager
        self.retriever = Retriever(self.mock_embedding_manager)
        
        # Create test documents
        self.test_docs = [
            Document(
                page_content="This is a document about artificial intelligence.",
                metadata={"source": "ai.txt", "chunk_id": 1}
            ),
            Document(
                page_content="Machine learning is a subset of AI that involves statistical techniques.",
                metadata={"source": "ml.txt", "chunk_id": 2}
            ),
            Document(
                page_content="Natural language processing helps computers understand human language.",
                metadata={"source": "nlp.txt", "chunk_id": 3}
            ),
        ]
        
        # Create mock similarity scores
        self.similarity_scores = [0.85, 0.6, 0.75]
    
    def test_retrieve_documents(self):
        """Test basic document retrieval functionality."""
        # Create mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.search.return_value = [
            (self.test_docs[0], self.similarity_scores[0]),
            (self.test_docs[1], self.similarity_scores[1]),
            (self.test_docs[2], self.similarity_scores[2]),
        ]
        
        # Retrieve documents
        results = self.retriever.retrieve_documents(
            "What is AI?", mock_db_manager, top_k=3
        )
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0].page_content, self.test_docs[0].page_content)
        self.assertEqual(results[0][1], self.similarity_scores[0])
        
        # Verify that documents are sorted by score (highest first)
        self.assertTrue(results[0][1] >= results[1][1])
        self.assertTrue(results[1][1] >= results[2][1])
    
    def test_filter_by_similarity_threshold(self):
        """Test filtering by similarity threshold."""
        # Create mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.search.return_value = [
            (self.test_docs[0], 0.85),
            (self.test_docs[1], 0.6),
            (self.test_docs[2], 0.3),  # Below default threshold (0.7)
        ]
        
        # Set high similarity threshold
        self.retriever.similarity_threshold = 0.7
        
        # Retrieve documents
        results = self.retriever.retrieve_documents(
            "What is AI?", mock_db_manager, top_k=3
        )
        
        # Verify that only documents above threshold are returned
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0][1], 0.7)
        self.assertGreaterEqual(results[1][1], 0.7)
    
    def test_filter_by_metadata(self):
        """Test filtering by metadata criteria."""
        # Create mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.search.return_value = [
            (self.test_docs[0], 0.85),
            (self.test_docs[1], 0.75),
            (self.test_docs[2], 0.65),
        ]
        
        # Set filter criteria
        filter_metadata = {"source": "ml.txt"}
        
        # Retrieve documents with filter
        results = self.retriever.retrieve_documents(
            "What is machine learning?", 
            mock_db_manager, 
            top_k=3,
            filter_metadata=filter_metadata
        )
        
        # Verify that only matching documents are returned
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].metadata["source"], "ml.txt")
    
    def test_deduplicate_results(self):
        """Test deduplication of similar results."""
        # Create documents with duplicate content
        duplicate_docs = [
            Document(
                page_content="This is a document about AI.",
                metadata={"source": "ai1.txt", "chunk_id": 1}
            ),
            Document(
                page_content="This is a document about AI.",  # Duplicate
                metadata={"source": "ai2.txt", "chunk_id": 2}
            ),
            Document(
                page_content="Machine learning is different.",
                metadata={"source": "ml.txt", "chunk_id": 3}
            ),
        ]
        
        # Create mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.search.return_value = [
            (duplicate_docs[0], 0.85),
            (duplicate_docs[1], 0.8),  # Duplicate
            (duplicate_docs[2], 0.7),
        ]
        
        # Retrieve documents
        results = self.retriever.retrieve_documents(
            "Tell me about AI", mock_db_manager, top_k=3
        )
        
        # Verify that duplicates are removed
        self.assertEqual(len(results), 2)
    
    @patch('src.retriever.TfidfVectorizer')
    def test_reranking(self, mock_tfidf):
        """Test reranking of results."""
        # Configure mock TF-IDF
        mock_tfidf_instance = MagicMock()
        mock_tfidf.return_value = mock_tfidf_instance
        
        # Mock TF-IDF matrix and cosine similarity
        mock_tfidf_instance.fit_transform.return_value = MagicMock()
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # Mock cosine similarity to favor the third document
            mock_cosine.return_value = np.array([[0.5, 0.4, 0.9]])
            
            # Create mock database manager
            mock_db_manager = MagicMock()
            mock_db_manager.search.return_value = [
                (self.test_docs[0], 0.7),  # Original rank: 1
                (self.test_docs[1], 0.6),  # Original rank: 2
                (self.test_docs[2], 0.5),  # Original rank: 3
            ]
            
            # Enable reranking
            self.retriever.reranking_enabled = True
            
            # Retrieve documents
            results = self.retriever.retrieve_documents(
                "How does NLP work?", mock_db_manager, top_k=3
            )
            
            # With reranking, the third document should now be first (or at least higher)
            # due to the mocked cosine similarity
            self.assertEqual(results[0][0].metadata["source"], "nlp.txt")

if __name__ == "__main__":
    unittest.main()