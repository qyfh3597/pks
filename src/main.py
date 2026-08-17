import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Main pipeline for the Personal Knowledge Summary System.
Orchestrates the entire workflow from retrieval to summarization.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config
from .logger import get_logger
from .embed import EmbeddingManager
from .retrieval import Retriever
from .rerank import HybridReranker
from .summarize1 import SummarizationEngine
from .explain import ExplainabilityEngine

logger = get_logger()


class PersonalSummaryPipeline:
    """Main pipeline for personal knowledge summarization."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.config.ensure_directories()
        
        # Initialize components
        logger.info("Initializing Personal Summary Pipeline...")
        
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.get("embedding.model_name"),
            persist_dir=self.config.get("chroma.persist_dir"),
            collection_name=self.config.get("chroma.collection_name"),
            device=self.config.get("embedding.device", "cpu")
        )
        
        self.retriever = Retriever(
            embedding_manager=self.embedding_manager,
            top_k=self.config.get("retrieval.top_candidates_after_retrieval", 50)
        )
        
        self.reranker = HybridReranker(
            embedding_manager=self.embedding_manager,
            weights=self.config.get("reranking.manual_weights")
        )
        
        self.summarizer = SummarizationEngine(
            model_name=self.config.get("summarization.zhipu_model"),
            use_zhipu=self.config.get("summarization.use_zhipu_ai", True)
        )
        
        self.explainer = ExplainabilityEngine()
        
        logger.info("Pipeline initialization complete")
    
    def process_query(
        self,
        query: str,
        top_k: int = 5,
        rerank_mode: str = "hybrid",
        save_outputs: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single query through the entire pipeline.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            rerank_mode: Reranking mode ('hybrid', 'learned', 'manual')
            save_outputs: Whether to save outputs to files
            
        Returns:
            Complete pipeline output
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Retrieval
        logger.info("Step 1: Initial Retrieval")
        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.config.get("retrieval.top_candidates_after_retrieval", 50)
        )
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # Step 2: Reranking
        logger.info("Step 2: Hybrid Reranking")
        reranked_results = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
            mode=rerank_mode
        )
        logger.info(f"Reranked to top-{len(reranked_results)} results")
        
        # Step 3: Summarization
        logger.info("Step 3: Summarization")
        summary = self.summarizer.summarize(
            query=query,
            top_k_results=reranked_results,
            max_summary_length=self.config.get("summarization.max_summary_length", 250)
        )
        logger.info("Summary generated")
        
        # Step 4: Explainability
        logger.info("Step 4: Generating Explanations")
        feature_explanation = self.explainer.generate_feature_explanation(reranked_results)
        evidence_explanation = self.explainer.generate_evidence_explanation(summary, reranked_results)
        logger.info("Explanations generated")
        
        # Compile results
        output = {
            "query": query,
            "retrieval": {
                "num_candidates": len(candidates),
                "top_5_candidates": candidates[:5]
            },
            "reranking": {
                "mode": rerank_mode,
                "num_results": len(reranked_results),
                "results": reranked_results
            },
            "summary": summary,
            "explanations": {
                "feature_importance": feature_explanation,
                "evidence": evidence_explanation
            }
        }
        
        # Step 5: Save outputs
        if save_outputs:
            logger.info("Step 5: Saving Outputs")
            self._save_outputs(query, output)
        
        logger.info("Query processing complete")
        return output
    
    def _save_outputs(self, query: str, output: Dict[str, Any]) -> None:
        """
        Save pipeline outputs to files.
        
        Args:
            query: Query text
            output: Pipeline output
        """
        query_id = query[:50].replace(" ", "_").replace("/", "_")
        
        # Save summary
        summary_file = Path(self.config.get("output.summaries_dir")) / f"{query_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "summary": output["summary"]
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_file}")
        
        # Save reranked results
        results_file = Path(self.config.get("output.experiments_dir")) / f"{query_id}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "reranked_results": output["reranking"]["results"]
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")
        
        # Generate and save HTML report
        html_content = self.explainer.generate_html_report(
            query,
            output["summary"],
            output["reranking"]["results"],
            output["explanations"]["feature_importance"],
            output["explanations"]["evidence"]
        )
        
        report_file = Path(self.config.get("output.reports_dir")) / f"{query_id}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report saved to {report_file}")
    
    def batch_process(
        self,
        queries: List[str],
        top_k: int = 5,
        rerank_mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of top results per query
            rerank_mode: Reranking mode
            
        Returns:
            List of outputs for each query
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
            
            try:
                output = self.process_query(query, top_k, rerank_mode)
                results.append(output)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Summary System - Main Pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="File with multiple queries (one per line)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top results to return"
    )
    parser.add_argument(
        "--rerank-mode",
        type=str,
        default="hybrid",
        choices=['hybrid', 'learned', 'manual'],
        help="Reranking mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save outputs to files"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PersonalSummaryPipeline(args.config)
    
    # Process queries
    if args.query:
        # Single query
        output = pipeline.process_query(
            query=args.query,
            top_k=args.topk,
            rerank_mode=args.rerank_mode,
            save_outputs=not args.no_save
        )
        
        print("\n" + "="*80)
        print("QUERY RESULT")
        print("="*80)
        print(f"Query: {args.query}")
        print(f"\nSummary:\n{output['summary'].get('summary', 'N/A')}")
        print(f"\nKeywords: {', '.join(output['summary'].get('keywords', []))}")
        print(f"\nTop {args.topk} Results:")
        for i, result in enumerate(output['reranking']['results'], 1):
            print(f"  [{i}] {result['metadata']['title']} (score: {result['rerank_score']:.4f})")
        
    elif args.queries_file:
        # Multiple queries from file
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        outputs = pipeline.batch_process(
            queries=queries,
            top_k=args.topk,
            rerank_mode=args.rerank_mode
        )
        
        print(f"\nProcessed {len(outputs)} queries")
        print(f"Results saved to {pipeline.config.get('output.root_dir')}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
