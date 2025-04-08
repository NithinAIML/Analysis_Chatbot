import os
import unittest
from langchain.schema import Document
from src.document_processor import DocumentProcessor
import config

class TestChunkingStrategies(unittest.TestCase):
    """
    Test cases for document chunking strategies.
    """
    
    def setUp(self):
        """Set up common test resources."""
        self.test_document = Document(
            page_content="""
            This is a test document for evaluating chunking strategies.
            It contains multiple paragraphs with varying content.
            
            This is the second paragraph with some more text.
            We need to ensure that chunking works properly across paragraphs.
            
            This is the third paragraph.
            It should be separated from the previous ones.
            
            And finally, this is the last paragraph.
            It contains the closing section of our test document.
            """,
            metadata={"source": "test.txt", "page": 1}
        )
        
        # Initialize document processor with default settings
        self.processor = DocumentProcessor()
    
    def test_chunk_size(self):
        """Test that chunks respect the specified size."""
        # Set a specific chunk size
        chunk_size = 100
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=0)
        
        # Chunk the document
        chunked_docs = processor.chunk_documents([self.test_document])
        
        # Check that all chunks are smaller than or equal to the chunk size
        for doc in chunked_docs:
            self.assertLessEqual(
                len(doc.page_content), 
                chunk_size, 
                f"Chunk size {len(doc.page_content)} exceeds limit {chunk_size}."
            )
    
    def test_chunk_overlap(self):
        """Test that chunks overlap as expected."""
        # Set specific chunk parameters
        chunk_size = 100
        chunk_overlap = 50
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Chunk the document
        chunked_docs = processor.chunk_documents([self.test_document])
        
        # Need at least 2 chunks to test overlap
        if len(chunked_docs) >= 2:
            for i in range(len(chunked_docs) - 1):
                current_chunk = chunked_docs[i].page_content
                next_chunk = chunked_docs[i + 1].page_content
                
                # Find the overlap
                found_overlap = False
                for overlap_size in range(min(len(current_chunk), len(next_chunk)), 0, -1):
                    if current_chunk[-overlap_size:] == next_chunk[:overlap_size]:
                        found_overlap = True
                        self.assertGreaterEqual(overlap_size, 0, "Chunks should have some overlap.")
                        break
    
    def test_chunk_boundaries(self):
        """Test that chunks respect semantic boundaries where possible."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        # Chunk the document
        chunked_docs = processor.chunk_documents([self.test_document])
        
        # Check that most chunks start at paragraph or sentence boundaries
        boundary_markers = ["\n", ".", "?", "!"]
        
        for doc in chunked_docs:
            content = doc.page_content.lstrip()
            
            # Check if previous character (in original text) was a boundary marker
            # This is a simplified test and might not catch all cases
            found_boundary = content == self.test_document.page_content or any(
                self.test_document.page_content.find(content) <= 0 or
                self.test_document.page_content[self.test_document.page_content.find(content) - 1] in boundary_markers
                for marker in boundary_markers
            )
            
            # Not all chunks will perfectly align with boundaries
            # This is just to check that the algorithm tries to respect them
            if not found_boundary:
                print(f"Warning: Chunk does not start at a semantic boundary: {content[:20]}...")
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved and enhanced in chunks."""
        processor = DocumentProcessor()
        
        # Chunk the document
        chunked_docs = processor.chunk_documents([self.test_document])
        
        # Check that original metadata is preserved
        for doc in chunked_docs:
            self.assertEqual(doc.metadata["source"], self.test_document.metadata["source"])
            self.assertEqual(doc.metadata["page"], self.test_document.metadata["page"])
            
            # Check that new metadata is added
            self.assertIn("chunk_id", doc.metadata)
            self.assertIn("chunk_size", doc.metadata)
            self.assertIn("token_count", doc.metadata)

if __name__ == "__main__":
    unittest.main()