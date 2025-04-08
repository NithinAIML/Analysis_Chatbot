import os
import re
import fitz  # PyMuPDF
import tiktoken
import logging
from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import numpy as np
from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Class for processing documents, including loading and chunking.
    """
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initialize the DocumentProcessor with configurable chunking parameters.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document and convert it to text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content and metadata
        """
        logger.info(f"Loading PDF from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        documents = []
        try:
            pdf_document = fitz.open(file_path)
            
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    metadata = {
                        "source": os.path.basename(file_path),
                        "page": page_num + 1,
                        "total_pages": len(pdf_document),
                        "file_path": file_path,
                    }
                    documents.append(Document(page_content=text, metadata=metadata))
                    
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for more effective processing.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents with chunk size {self.chunk_size} and overlap {self.chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False,
        )
        
        chunked_documents = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                # Create a new metadata dict for each chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["chunk_size"] = len(chunk)
                chunk_metadata["token_count"] = len(self.tokenizer.encode(chunk))
                
                # Add the chunked document
                chunked_documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of chunked Document objects from all PDFs
        """
        logger.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        all_documents = []
        
        # Find all PDF files in the directory
        pdf_files = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.lower().endswith(".pdf")
        ]
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                documents = self.load_pdf(pdf_file)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        # Chunk all documents
        return self.chunk_documents(all_documents)
    
    def get_chunk_statistics(self, chunked_documents: List[Document]) -> Dict[str, Any]:
        """
        Calculate statistics about the document chunks.
        
        Args:
            chunked_documents: List of Document chunks
            
        Returns:
            Dictionary of statistics about the chunks
        """
        if not chunked_documents:
            return {}
            
        # Extract chunk sizes and token counts
        chunk_sizes = [doc.metadata.get("chunk_size", 0) for doc in chunked_documents]
        token_counts = [doc.metadata.get("token_count", 0) for doc in chunked_documents]
        
        # Calculate statistics
        stats = {
            "total_chunks": len(chunked_documents),
            "avg_chunk_size": np.mean(chunk_sizes),
            "min_chunk_size": np.min(chunk_sizes),
            "max_chunk_size": np.max(chunk_sizes),
            "std_chunk_size": np.std(chunk_sizes),
            "avg_tokens": np.mean(token_counts),
            "min_tokens": np.min(token_counts),
            "max_tokens": np.max(token_counts),
            "total_tokens": np.sum(token_counts),
            "total_characters": np.sum(chunk_sizes),
        }
        
        return stats
    
    def optimize_chunk_parameters(
        self, 
        document: Document, 
        min_size: int = MIN_CHUNK_SIZE, 
        max_size: int = MAX_CHUNK_SIZE,
        step: int = 100
    ) -> Tuple[int, int]:
        """
        Find optimal chunk size and overlap for a document.
        This is a simplified approach - for a production system, you would want
        to evaluate retrieval performance for different chunk parameters.
        
        Args:
            document: Document to optimize for
            min_size: Minimum chunk size to try
            max_size: Maximum chunk size to try
            step: Step size for chunk size testing
            
        Returns:
            Tuple of (optimal_chunk_size, optimal_chunk_overlap)
        """
        logger.info(f"Optimizing chunk parameters for document of size {len(document.page_content)}")
        
        # Simple heuristic: aim for chunks of approximately 150-250 tokens
        # This is based on research showing this range works well for many RAG tasks
        
        # Get total tokens in document
        total_tokens = len(self.tokenizer.encode(document.page_content))
        
        # Estimate optimal chunk size based on token count
        if total_tokens < 1000:
            # For small documents, use smaller chunks
            optimal_chunk_size = 800
            optimal_chunk_overlap = 150
        elif total_tokens < 10000:
            # For medium documents
            optimal_chunk_size = 1000
            optimal_chunk_overlap = 200
        else:
            # For large documents
            optimal_chunk_size = 1500
            optimal_chunk_overlap = 300
            
        logger.info(f"Determined optimal chunk size: {optimal_chunk_size}, overlap: {optimal_chunk_overlap}")
        return optimal_chunk_size, optimal_chunk_overlap

    def analyze_chunk_distribution(self, chunked_documents: List[Document]) -> pd.DataFrame:
        """
        Create a DataFrame with information about chunks for visualization.
        
        Args:
            chunked_documents: List of Document chunks
            
        Returns:
            DataFrame with chunk statistics
        """
        if not chunked_documents:
            return pd.DataFrame()
            
        data = []
        for doc in chunked_documents:
            data.append({
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
                "chunk_size": doc.metadata.get("chunk_size", 0),
                "token_count": doc.metadata.get("token_count", 0),
            })
            
        return pd.DataFrame(data)