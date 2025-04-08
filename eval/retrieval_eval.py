import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain.schema import Document
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
from src.retriever import Retriever

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Class for evaluating document retrieval performance.
    """
    
    def __init__(self):
        """
        Initialize the retrieval evaluator.
        """
        pass
    
    def evaluate_retrieval(
        self, 
        query: str, 
        retrieved_docs: List[Tuple[Document, float]], 
        relevant_doc_ids: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance for a single query.
        
        Args:
            query: The query string
            retrieved_docs: List of (Document, score) tuples
            relevant_doc_ids: List of IDs of known relevant documents
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating retrieval for query: {query[:50]}...")
        
        # Extract IDs of retrieved documents
        retrieved_ids = []
        for doc, _ in retrieved_docs:
            # Use chunk_id as the document ID if available
            if "chunk_id" in doc.metadata and "source" in doc.metadata:
                doc_id = f"{doc.metadata['source']}-{doc.metadata['chunk_id']}"
            else:
                # Fallback to using the content hash
                doc_id = str(hash(doc.page_content))
            retrieved_ids.append(doc_id)
        
        # Calculate metrics
        metrics = {}
        
        # Precision
        tp = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0
        metrics["precision"] = precision
        
        # Recall
        recall = tp / len(relevant_doc_ids) if relevant_doc_ids else 0
        metrics["recall"] = recall
        
        # F1 Score
        if precision + recall > 0:
            metrics["f1"] = 2 * precision * recall / (precision + recall)
        else:
            metrics["f1"] = 0
        
        # Mean Reciprocal Rank (MRR)
        rank = None
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_doc_ids:
                rank = i + 1
                break
        
        metrics["mrr"] = 1 / rank if rank is not None else 0
        
        logger.info(f"Retrieval evaluation metrics: {metrics}")
        return metrics
    
    def evaluate_retrieval_batch(
        self,
        queries_and_relevance: List[Dict[str, Any]],
        retriever: Retriever,
        db_manager,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance across multiple queries.
        
        Args:
            queries_and_relevance: List of dicts with query and relevant doc IDs
            retriever: The retriever to evaluate
            db_manager: The vector database manager
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary of aggregated evaluation metrics and per-query results
        """
        logger.info(f"Evaluating retrieval batch with {len(queries_and_relevance)} queries")
        
        results = []
        
        for item in queries_and_relevance:
            query = item["query"]
            relevant_doc_ids = item["relevant_doc_ids"]
            
            # Retrieve documents
            retrieved_docs = retriever.retrieve_documents(
                query, db_manager, top_k=top_k
            )
            
            # Evaluate
            metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)
            
            # Store results
            results.append({
                "query": query,
                "metrics": metrics,
                "retrieved_docs": retrieved_docs,
                "relevant_doc_ids": relevant_doc_ids
            })
        
        # Calculate aggregate metrics
        aggregated = {
            "precision": np.mean([r["metrics"]["precision"] for r in results]),
            "recall": np.mean([r["metrics"]["recall"] for r in results]),
            "f1": np.mean([r["metrics"]["f1"] for r in results]),
            "mrr": np.mean([r["metrics"]["mrr"] for r in results]),
            "num_queries": len(results),
            "per_query_results": results
        }
        
        logger.info(f"Aggregate evaluation metrics: precision={aggregated['precision']:.3f}, recall={aggregated['recall']:.3f}, f1={aggregated['f1']:.3f}, mrr={aggregated['mrr']:.3f}")
        return aggregated
    
    def plot_precision_recall_curve(
        self,
        queries_and_relevance: List[Dict[str, Any]],
        retriever: Retriever,
        db_manager
    ) -> plt.Figure:
        """
        Plot precision-recall curve for retrieval evaluation.
        
        Args:
            queries_and_relevance: List of dicts with query and relevant doc IDs
            retriever: The retriever to evaluate
            db_manager: The vector database manager
        
        Returns:
            Matplotlib figure with precision-recall curve
        """
        logger.info("Plotting precision-recall curve")
        
        # Collect all relevance scores and true relevance
        all_scores = []
        all_relevance = []
        
        for item in queries_and_relevance:
            query = item["query"]
            relevant_doc_ids = item["relevant_doc_ids"]
            
            # Retrieve a larger number of documents to plot the curve
            retrieved_docs = retriever.retrieve_documents(
                query, db_manager, top_k=50
            )
            
            # Extract IDs and scores
            for doc, score in retrieved_docs:
                if "chunk_id" in doc.metadata and "source" in doc.metadata:
                    doc_id = f"{doc.metadata['source']}-{doc.metadata['chunk_id']}"
                else:
                    doc_id = str(hash(doc.page_content))
                
                all_scores.append(score)
                all_relevance.append(1 if doc_id in relevant_doc_ids else 0)
        
        # If no data, return empty figure
        if not all_scores:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Convert to numpy arrays
        scores_array = np.array(all_scores)
        relevance_array = np.array(all_relevance)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(relevance_array, scores_array)
        
        # Calculate AUC
        pr_auc = auc(recall, precision)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(recall, precision, 'b-', linewidth=2,
                label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
        
        # Add baseline
        baseline = np.sum(relevance_array) / len(relevance_array)
        ax.plot([0, 1], [baseline, baseline], 'r--', 
                label=f'Baseline (Random) = {baseline:.3f}')
        
        # Annotate
        ax.set_title('Precision-Recall Curve for Document Retrieval', fontsize=16)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        fig.tight_layout()
        return fig
    
    def visualize_evaluation_results(self, results: Dict[str, Any]) -> plt.Figure:
        """
        Visualize evaluation results across multiple metrics.
        
        Args:
            results: Aggregated evaluation results
        
        Returns:
            Matplotlib figure with evaluation visualization
        """
        # Extract per-query metrics
        if "per_query_results" not in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No detailed results available", ha='center', va='center')
            return fig
            
        query_results = results["per_query_results"]
        
        # Create a DataFrame for easy plotting
        data = []
        for res in query_results:
            data.append({
                "query": res["query"][:30] + "..." if len(res["query"]) > 30 else res["query"],
                "precision": res["metrics"]["precision"],
                "recall": res["metrics"]["recall"],
                "f1": res["metrics"]["f1"],
                "mrr": res["metrics"]["mrr"]
            })
            
        df = pd.DataFrame(data)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision
        axs[0, 0].bar(df["query"], df["precision"], color='skyblue')
        axs[0, 0].set_title('Precision by Query', fontsize=14)
        axs[0, 0].set_ylim([0, 1])
        axs[0, 0].set_ylabel('Precision')
        axs[0, 0].tick_params(axis='x', rotation=45)
        axs[0, 0].axhline(results["precision"], color='r', linestyle='--', 
                         label=f'Avg: {results["precision"]:.3f}')
        axs[0, 0].legend()
        
        # Recall
        axs[0, 1].bar(df["query"], df["recall"], color='lightgreen')
        axs[0, 1].set_title('Recall by Query', fontsize=14)
        axs[0, 1].set_ylim([0, 1])
        axs[0, 1].set_ylabel('Recall')
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].axhline(results["recall"], color='r', linestyle='--', 
                        label=f'Avg: {results["recall"]:.3f}')
        axs[0, 1].legend()
        
        # F1 Score
        axs[1, 0].bar(df["query"], df["f1"], color='salmon')
        axs[1, 0].set_title('F1 Score by Query', fontsize=14)
        axs[1, 0].set_ylim([0, 1])
        axs[1, 0].set_ylabel('F1 Score')
        axs[1, 0].tick_params(axis='x', rotation=45)
        axs[1, 0].axhline(results["f1"], color='r', linestyle='--', 
                        label=f'Avg: {results["f1"]:.3f}')
        axs[1, 0].legend()
        
        # MRR
        axs[1, 1].bar(df["query"], df["mrr"], color='mediumpurple')
        axs[1, 1].set_title('Mean Reciprocal Rank by Query', fontsize=14)
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].set_ylabel('MRR')
        axs[1, 1].tick_params(axis='x', rotation=45)
        axs[1, 1].axhline(results["mrr"], color='r', linestyle='--', 
                        label=f'Avg: {results["mrr"]:.3f}')
        axs[1, 1].legend()
        
        # Overall title
        fig.suptitle('Retrieval Evaluation Results', fontsize=16)
        
        fig.tight_layout()
        return fig