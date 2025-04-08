import unittest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.embedding import EmbeddingManager
from src.vectordb import FAISSManager
from src.retriever import Retriever
from src.memory import ConversationMemory

class TestGeneration(unittest.TestCase):
    """
    Test cases for response generation in the RAG pipeline.
    """
    
    def setUp(self):
        """Set up common test resources."""
        # Create mock components
        self.mock_document_processor = MagicMock(spec=DocumentProcessor)
        self.mock_embedding_manager = MagicMock(spec=EmbeddingManager)
        self.mock_faiss_manager = MagicMock(spec=FAISSManager)
        self.mock_retriever = MagicMock(spec=Retriever)
        self.mock_memory = MagicMock(spec=ConversationMemory)
        
        # Create test documents
        self.test_docs = [
            Document(
                page_content="HealthChoice Illinois is a Medicaid managed care program. The program provides health care to most Illinois Medicaid participants.",
                metadata={"source": "handbook.pdf", "page": 5, "chunk_id": 1}
            ),
            Document(
                page_content="To keep getting care through HealthChoice Illinois, you need to renew your Medicaid coverage every year.",
                metadata={"source": "handbook.pdf", "page": 5, "chunk_id": 2}
            ),
        ]
        
        # Create pipeline with mock components
        self.pipeline = RAGPipeline(
            document_processor=self.mock_document_processor,
            embedding_manager=self.mock_embedding_manager,
            faiss_manager=self.mock_faiss_manager,
            retriever=self.mock_retriever,
            memory=self.mock_memory
        )
        
        # Mock the LLM
        self.pipeline.llm = MagicMock()
    
    def test_response_generation(self):
        """Test response generation with retrieved documents."""
        # Configure mocks
        self.mock_retriever.retrieve_documents.return_value = [
            (self.test_docs[0], 0.9),
            (self.test_docs[1], 0.7),
        ]
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "HealthChoice Illinois is a Medicaid managed care program that provides health care to most Illinois Medicaid participants. You need to renew your Medicaid coverage every year to keep getting care through this program."
        self.pipeline.llm.invoke.return_value = mock_response
        
        # Generate response
        result = self.pipeline.generate_response("What is HealthChoice Illinois?")
        
        # Verify response
        self.assertEqual(result["response"], mock_response.content)
        self.assertEqual(len(result["retrieved_documents"]), 2)
        
        # Verify that retriever was called
        self.mock_retriever.retrieve_documents.assert_called_once()
        
        # Verify that memory was updated
        self.mock_memory.add_message.assert_called()
    
    def test_conversation_context(self):
        """Test that conversation context is included in generation."""
        # Configure mocks
        self.mock_retriever.retrieve_documents.return_value = [
            (self.test_docs[0], 0.9),
        ]
        
        # Mock conversation context
        self.mock_memory.get_context_string.return_value = "User: What is HealthChoice Illinois?\nAssistant: HealthChoice Illinois is a Medicaid managed care program."
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "You need to renew your Medicaid coverage every year to keep getting care through HealthChoice Illinois."
        self.pipeline.llm.invoke.return_value = mock_response
        
        # Generate response with follow-up question
        result = self.pipeline.generate_response("Do I need to renew my coverage?")
        
        # Verify that LLM invoke was called with conversation context
        args, kwargs = self.pipeline.llm.invoke.call_args
        prompt = args[0]
        
        # Check that the prompt contains conversation history
        prompt_str = str(prompt)
        self.assertIn("What is HealthChoice Illinois?", prompt_str)
        
        # Verify that memory was updated
        self.mock_memory.add_message.assert_called()
    
    def test_no_relevant_documents(self):
        """Test response generation when no relevant documents are found."""
        # Configure mocks
        self.mock_retriever.retrieve_documents.return_value = []
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "I don't have specific information about that in my knowledge base."
        self.pipeline.llm.invoke.return_value = mock_response
        
        # Generate response
        result = self.pipeline.generate_response("What is quantum computing?")
        
        # Verify that LLM still generated a response
        self.assertEqual(result["response"], mock_response.content)
        
        # Verify that context included information about no relevant documents
        args, kwargs = self.pipeline.llm.invoke.call_args
        prompt = args[0]
        prompt_str = str(prompt)
        self.assertIn("No relevant information found", prompt_str)
    
    def test_document_formatting(self):
        """Test proper formatting of documents for the prompt."""
        # Configure mocks
        self.mock_retriever.retrieve_documents.return_value = [
            (self.test_docs[0], 0.9),
            (self.test_docs[1], 0.7),
        ]
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        self.pipeline.llm.invoke.return_value = mock_response
        
        # Generate response
        self.pipeline.generate_response("Test query")
        
        # Check document formatting in prompt
        args, kwargs = self.pipeline.llm.invoke.call_args
        prompt = args[0]
        prompt_str = str(prompt)
        
        # Verify document citation format
        self.assertIn("[Document 1]", prompt_str)
        self.assertIn("Source: handbook.pdf", prompt_str)
        self.assertIn("Page: 5", prompt_str)
        self.assertIn("Relevance: 0.9", prompt_str)

if __name__ == "__main__":
    unittest.main()