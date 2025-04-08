import logging
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkVisualizer:
    """
    Class for visualizing document chunks and chunk parameters.
    """
    
    def __init__(self):
        """
        Initialize the chunk visualizer.
        """
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
    
    def visualize_chunk_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create a visualization of chunk size distribution.
        
        Args:
            df: DataFrame containing chunk data
        
        Returns:
            Matplotlib figure
        """
        if df.empty:
            logger.warning("Cannot visualize: DataFrame is empty")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of chunk sizes
        sns.histplot(df['chunk_size'], bins=20, kde=True, ax=ax)
        
        # Add vertical line for mean
        mean_size = df['chunk_size'].mean()
        ax.axvline(mean_size, color='r', linestyle='--', 
                   label=f'Mean Size: {mean_size:.1f} chars')
        
        # Annotate
        ax.set_title('Distribution of Chunk Sizes', fontsize=16)
        ax.set_xlabel('Chunk Size (characters)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def visualize_token_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """
        Create a visualization of token count distribution.
        
        Args:
            df: DataFrame containing chunk data
        
        Returns:
            Matplotlib figure
        """
        if df.empty or 'token_count' not in df.columns:
            logger.warning("Cannot visualize: DataFrame is empty or missing token_count")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No token data available", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of token counts
        sns.histplot(df['token_count'], bins=20, kde=True, ax=ax)
        
        # Add vertical line for mean
        mean_tokens = df['token_count'].mean()
        ax.axvline(mean_tokens, color='r', linestyle='--', 
                   label=f'Mean Tokens: {mean_tokens:.1f}')
        
        # Annotate
        ax.set_title('Distribution of Token Counts per Chunk', fontsize=16)
        ax.set_xlabel('Token Count', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def visualize_chunk_overlap(
        self, 
        chunk_size: int = config.DEFAULT_CHUNK_SIZE, 
        overlap_values: Optional[List[int]] = None
    ) -> plt.Figure:
        """
        Create a visualization of different chunk overlap strategies.
        
        Args:
            chunk_size: The chunk size to visualize
            overlap_values: List of overlap values to visualize
        
        Returns:
            Matplotlib figure
        """
        if overlap_values is None:
            overlap_values = [0, 100, 200, 300, 400]
        
        # Create text representation of chunks with different overlaps
        fig, ax = plt.subplots(figsize=(12, 8))
        y_positions = []
        labels = []
        
        # For each overlap value, create a visual representation
        for i, overlap in enumerate(overlap_values):
            # Calculate effective chunks
            effective_size = chunk_size - overlap
            y_pos = i * 2  # Vertical position
            
            # Create colored rectangles representing chunks
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            
            # Draw chunks with overlaps
            for j in range(5):  # Draw 5 chunks
                start_pos = j * effective_size
                rect = plt.Rectangle((start_pos, y_pos-0.4), chunk_size, 0.8, 
                                     fc=colors[j % len(colors)], alpha=0.7)
                ax.add_patch(rect)
                
                # Add text in the middle of the chunk
                text_pos = start_pos + chunk_size / 2
                ax.text(text_pos, y_pos, f'Chunk {j+1}', 
                        ha='center', va='center', fontweight='bold')
            
            # Add label for this overlap setting
            y_positions.append(y_pos)
            labels.append(f'Overlap: {overlap} chars ({overlap/chunk_size:.0%})')
        
        # Set axis limits
        ax.set_xlim(-chunk_size*0.1, chunk_size*5)
        ax.set_ylim(-1, len(overlap_values)*2)
        
        # Add y-axis labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        
        # Add annotations and titles
        ax.set_title('Visualization of Chunk Overlaps', fontsize=16)
        ax.set_xlabel('Text Position (characters)', fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Remove y-axis line
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    def visualize_chunk_size_impact(
        self,
        sizes: Optional[List[int]] = None,
        metrics: Optional[Dict[int, float]] = None
    ) -> plt.Figure:
        """
        Visualize the impact of different chunk sizes on retrieval quality.
        
        Args:
            sizes: List of chunk sizes
            metrics: Dictionary mapping chunk sizes to performance metrics
        
        Returns:
            Matplotlib figure
        """
        if sizes is None or metrics is None:
            # Create sample data for illustration
            sizes = list(range(500, 5001, 500))
            
            # Simulate a performance curve with a peak
            base_metrics = [0.6, 0.7, 0.78, 0.82, 0.85, 0.83, 0.8, 0.76, 0.72, 0.68]
            metrics = {size: metric for size, metric in zip(sizes, base_metrics)}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the metrics
        sizes_list = list(metrics.keys())
        metrics_list = list(metrics.values())
        
        ax.plot(sizes_list, metrics_list, 'o-', linewidth=2, markersize=8)
        
        # Find optimal chunk size
        optimal_size = sizes_list[np.argmax(metrics_list)]
        max_metric = max(metrics_list)
        
        # Highlight optimal point
        ax.plot(optimal_size, max_metric, 'ro', markersize=12, 
                label=f'Optimal Size: {optimal_size} chars')
        
        # Annotate
        ax.set_title('Impact of Chunk Size on Retrieval Quality', fontsize=16)
        ax.set_xlabel('Chunk Size (characters)', fontsize=12)
        ax.set_ylabel('Quality Metric (e.g., F1 Score)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def visualize_retrieval_results(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        max_display: int = 5
    ) -> go.Figure:
        """
        Create an interactive visualization of retrieval results.
        
        Args:
            query: The query string
            retrieved_docs: List of (Document, score) tuples
            max_display: Maximum number of documents to display
        
        Returns:
            Plotly figure
        """
        if not retrieved_docs:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No documents retrieved",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Prepare data
        docs = [doc for doc, _ in retrieved_docs[:max_display]]
        scores = [score for _, score in retrieved_docs[:max_display]]
        labels = [f"Doc {i+1}: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}" 
                 for i, doc in enumerate(docs)]
        
        # Create bar chart
        fig = px.bar(
            x=scores,
            y=labels,
            orientation='h',
            labels={'x': 'Relevance Score', 'y': 'Document'},
            title=f'Retrieved Documents for Query: "{query[:50]}..."'
        )
        
        # Add document text as hover info
        hover_texts = [doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content 
                      for doc in docs]
        
        for i, hover_text in enumerate(hover_texts):
            fig.data[0].hovertext = hover_texts
        
        # Customize layout
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_range=[0, 1],
            yaxis=dict(autorange="reversed")  # Top document first
        )
        
        return fig