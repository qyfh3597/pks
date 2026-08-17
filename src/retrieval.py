import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Retrieval module for the Personal Knowledge Summary System.
Handles initial retrieval from the vector database.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

from .embed import EmbeddingManager
from .logger import get_logger

logger = get_logger()


class Retriever:
    """Handles document retrieval from the vector database."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        top_k: int = 50
    ):
        """
        Initialize retriever.
        
        Args:
            embedding_manager: EmbeddingManager instance
            top_k: Number of top candidates to retrieve
        """
        self.embedding_manager = embedding_manager
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return (uses default if None)
            where: Optional metadata filter
            
        Returns:
            List of retrieved documents with scores
        """
        k = top_k or self.top_k
        
        logger.info(f"Retrieving top-{k} documents for query: {query[:100]}...")
        
        # Query the collection
        results = self.embedding_manager.query(
            query_text=query,
            top_k=k,
            where=where
        )
        
        # Process results
        candidates = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i, (doc_id, document, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                # Convert distance to similarity (cosine distance to similarity)
                # For cosine distance in Chroma, similarity = 1 - distance
                similarity = 1 - distance
                
                candidate = {
                    "rank": i + 1,
                    "doc_id": doc_id,
                    "text": document,
                    "similarity_score": float(similarity),
                    "distance": float(distance),
                    "metadata": {
                        "doc_id": metadata.get("doc_id", ""),
                        "title": metadata.get("title", ""),
                        "paragraph_index": int(metadata.get("paragraph_index", 0)),
                        "keywords": json.loads(metadata.get("keywords", "[]")),
                        "language": metadata.get("language", ""),
                        "length": int(metadata.get("length", 0))
                    }
                }
                
                candidates.append(candidate)
        
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        return candidates
    
    def retrieve_and_save(
        self,
        query: str,
        output_file: str,
        top_k: Optional[int] = None,
        format: str = "json"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents and save to file.
        
        Args:
            query: Query text
            output_file: Path to output file
            top_k: Number of results to return
            format: Output format ('json' or 'csv')
            
        Returns:
            List of retrieved candidates
        """
        candidates = self.retrieve(query, top_k)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "num_results": len(candidates),
                    "candidates": candidates
                }, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'rank', 'doc_id', 'similarity_score', 'distance',
                    'title', 'paragraph_index', 'text_preview'
                ])
                writer.writeheader()
                
                for candidate in candidates:
                    writer.writerow({
                        'rank': candidate['rank'],
                        'doc_id': candidate['doc_id'],
                        'similarity_score': candidate['similarity_score'],
                        'distance': candidate['distance'],
                        'title': candidate['metadata']['title'],
                        'paragraph_index': candidate['metadata']['paragraph_index'],
                        'text_preview': candidate['text'][:100] + "..." if len(candidate['text']) > 100 else candidate['text']
                    })
        
        logger.info(f"Saved {len(candidates)} candidates to {output_file}")
        
        return candidates


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Retrieve documents from the vector database"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top candidates to retrieve"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/retrieval_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=['json', 'csv'],
        help="Output format"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="personal_notes",
        help="Name of the Chroma collection"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./.chroma/",
        help="Directory for Chroma persistence"
    )
    
    args = parser.parse_args()
    
    # Initialize embedding manager and retriever
    embedding_manager = EmbeddingManager(
        persist_dir=args.persist_dir,
        collection_name=args.collection
    )
    
    retriever = Retriever(embedding_manager, top_k=args.topk)
    
    # Retrieve and save
    candidates = retriever.retrieve_and_save(
        query=args.query,
        output_file=args.output,
        format=args.format
    )
    
    # Print summary
    print(f"\nRetrieval Results for: '{args.query}'")
    print(f"Total candidates: {len(candidates)}")
    print("\nTop 5 results:")
    for candidate in candidates[:5]:
        print(f"  [{candidate['rank']}] {candidate['metadata']['title']} (score: {candidate['similarity_score']:.4f})")
        print(f"      {candidate['text'][:80]}...")


if __name__ == "__main__":
    main()
