import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Evaluation module for the Personal Knowledge Summary System.
Computes NDCG, MAP, ROUGE, and other metrics.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer

from .logger import get_logger

logger = get_logger()


class EvaluationMetrics:
    """Computes evaluation metrics for retrieval and summarization."""
    
    @staticmethod
    def compute_ndcg(
        relevance_scores: List[float],
        k: int = 5
    ) -> float:
        """
        Compute NDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            relevance_scores: List of relevance scores (0-5)
            k: Cutoff position
            
        Returns:
            NDCG@k score
        """
        # Compute DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            dcg += score / np.log2(i + 2)  # i+2 because ranking starts from 1
        
        # Compute IDCG (ideal DCG with perfect ranking)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores[:k]):
            idcg += score / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return float(ndcg)
    
    @staticmethod
    def compute_map(
        relevance_scores: List[float],
        k: int = 5,
        threshold: float = 3.0
    ) -> float:
        """
        Compute MAP@k (Mean Average Precision).
        
        Args:
            relevance_scores: List of relevance scores
            k: Cutoff position
            threshold: Score threshold for relevance
            
        Returns:
            MAP@k score
        """
        relevant_count = 0
        sum_precisions = 0.0
        
        for i, score in enumerate(relevance_scores[:k]):
            if score >= threshold:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                sum_precisions += precision
        
        total_relevant = sum(1 for s in relevance_scores if s >= threshold)
        map_score = sum_precisions / total_relevant if total_relevant > 0 else 0.0
        
        return float(map_score)
    
    @staticmethod
    def compute_rouge(
        reference: str,
        hypothesis: str,
        rouge_types: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            rouge_types: List of ROUGE types to compute
            
        Returns:
            Dictionary of ROUGE scores
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]
        
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        
        result = {}
        for rouge_type in rouge_types:
            if rouge_type in scores:
                result[rouge_type] = scores[rouge_type].fmeasure
        
        return result
    
    @staticmethod
    def compute_mrr(
        relevance_scores: List[float],
        threshold: float = 3.0
    ) -> float:
        """
        Compute MRR (Mean Reciprocal Rank).
        
        Args:
            relevance_scores: List of relevance scores
            threshold: Score threshold for relevance
            
        Returns:
            MRR score
        """
        for i, score in enumerate(relevance_scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        
        return 0.0


class Evaluator:
    """Evaluates system performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        logger.info("Evaluator initialized")
    
    def evaluate_retrieval(
        self,
        retrieval_results: List[Dict[str, Any]],
        relevance_labels: List[float],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieval_results: List of retrieved results
            relevance_labels: Relevance labels for results
            k: Cutoff position
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(relevance_labels) != len(retrieval_results):
            logger.warning("Number of labels doesn't match number of results")
            relevance_labels = relevance_labels[:len(retrieval_results)]
        
        metrics = {
            "ndcg@5": EvaluationMetrics.compute_ndcg(relevance_labels, k=5),
            "ndcg@10": EvaluationMetrics.compute_ndcg(relevance_labels, k=10),
            "map@5": EvaluationMetrics.compute_map(relevance_labels, k=5),
            "map@10": EvaluationMetrics.compute_map(relevance_labels, k=10),
            "mrr": EvaluationMetrics.compute_mrr(relevance_labels),
            "num_relevant": sum(1 for label in relevance_labels if label >= 3.0)
        }
        
        return metrics
    
    def evaluate_summarization(
        self,
        reference_summary: str,
        generated_summary: str,
        rouge_types: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate summarization quality.
        
        Args:
            reference_summary: Reference summary
            generated_summary: Generated summary
            rouge_types: List of ROUGE types
            
        Returns:
            Dictionary of ROUGE scores
        """
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]
        
        scores = EvaluationMetrics.compute_rouge(
            reference_summary,
            generated_summary,
            rouge_types
        )
        
        return scores
    
    def evaluate_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        retrieval_metrics: Dict[str, float],
        summarization_metrics: Dict[str, float],
        rerank_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Evaluate a complete experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration parameters
            retrieval_metrics: Retrieval evaluation metrics
            summarization_metrics: Summarization evaluation metrics
            rerank_mode: Reranking mode used
            
        Returns:
            Complete experiment evaluation
        """
        evaluation = {
            "experiment_name": experiment_name,
            "rerank_mode": rerank_mode,
            "config": config,
            "retrieval_metrics": retrieval_metrics,
            "summarization_metrics": summarization_metrics,
            "overall_score": (
                sum(retrieval_metrics.values()) / len(retrieval_metrics) +
                sum(summarization_metrics.values()) / len(summarization_metrics)
            ) / 2 if retrieval_metrics and summarization_metrics else 0.0
        }
        
        return evaluation
    
    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str = "./output/experiments/experiments_results.csv"
    ) -> None:
        """
        Save evaluation results to CSV.
        
        Args:
            results: List of evaluation results
            output_file: Output file path
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten results for CSV
        flattened = []
        for result in results:
            row = {
                "experiment_name": result.get("experiment_name", ""),
                "rerank_mode": result.get("rerank_mode", ""),
                "overall_score": result.get("overall_score", 0.0)
            }
            
            # Add retrieval metrics
            for metric_name, metric_value in result.get("retrieval_metrics", {}).items():
                row[f"retrieval_{metric_name}"] = metric_value
            
            # Add summarization metrics
            for metric_name, metric_value in result.get("summarization_metrics", {}).items():
                row[f"summarization_{metric_name}"] = metric_value
            
            flattened.append(row)
        
        # Write to CSV
        if flattened:
            fieldnames = list(flattened[0].keys())
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened)
            
            logger.info(f"Results saved to {output_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate system performance"
    )
    parser.add_argument(
        "--retrieval-results",
        type=str,
        help="Path to retrieval results JSON"
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to relevance labels CSV"
    )
    parser.add_argument(
        "--reference-summary",
        type=str,
        help="Path to reference summary"
    )
    parser.add_argument(
        "--generated-summary",
        type=str,
        help="Path to generated summary"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/experiments/eval_results.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    evaluator = Evaluator()
    
    # Evaluate retrieval if labels provided
    if args.retrieval_results and args.labels:
        with open(args.retrieval_results, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        with open(args.labels, 'r', encoding='utf-8') as f:
            labels = [float(row['label']) for row in csv.DictReader(f)]
        
        retrieval_metrics = evaluator.evaluate_retrieval(
            retrieval_data.get("candidates", []),
            labels
        )
        
        print("\nRetrieval Metrics:")
        for metric_name, metric_value in retrieval_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Evaluate summarization if summaries provided
    if args.reference_summary and args.generated_summary:
        with open(args.reference_summary, 'r', encoding='utf-8') as f:
            reference = f.read()
        
        with open(args.generated_summary, 'r', encoding='utf-8') as f:
            generated = f.read()
        
        summarization_metrics = evaluator.evaluate_summarization(reference, generated)
        
        print("\nSummarization Metrics (ROUGE):")
        for metric_name, metric_value in summarization_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
