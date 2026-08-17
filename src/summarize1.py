import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Summarization module for the Personal Knowledge Summary System.
Supports both Zhipu AI API and template-based local summarization.
"""

import json
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

from .logger import get_logger

logger = get_logger()


class SummarizationEngine:
    """Handles summarization using Zhipu AI or local templates."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "glm-4-flash",
        use_zhipu: bool = True,
        max_retries: int = 2
    ):
        """
        Initialize summarization engine.
        
        Args:
            api_key: Zhipu AI API key
            model_name: Name of the Zhipu AI model
            use_zhipu: Whether to use Zhipu AI (fallback to template if False)
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        self.model_name = model_name
        self.use_zhipu = use_zhipu and bool(self.api_key)
        self.max_retries = max_retries
        self.api_base = "https://open.bigmodel.cn/api/paas/v4"
        
        if self.use_zhipu:
            logger.info(f"Summarization engine initialized with Zhipu AI model: {model_name}")
        else:
            logger.info("Summarization engine initialized with template-based mode")
    
    def _call_zhipu_api(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Call Zhipu AI API for summarization.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        if not self.api_key:
            logger.warning("Zhipu API key not available")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.7,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Zhipu API (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"API returned status code: {response.status_code}")
                    logger.warning(f"Response: {response.text}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _build_prompt(
        self,
        query: str,
        top_k_texts: List[str],
        max_summary_length: int = 250
    ) -> str:
        """
        Build prompt for Zhipu AI.
        
        Args:
            query: Query text
            top_k_texts: List of top-k relevant texts
            max_summary_length: Maximum summary length
            
        Returns:
            Formatted prompt
        """
        context = "\n\n".join([f"[文献{i+1}]\n{text}" for i, text in enumerate(top_k_texts)])
        
        prompt = f"""请根据以下文献内容，对查询"{query}"进行详细总结。

{context}

请按照以下JSON格式返回结果（严格JSON格式，无其他文本）：
{{
    "summary": "详细总结内容",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "key_points": ["要点1", "要点2", "要点3"],
    "evidence": [
        {{"source": "文献1", "quote": "相关引用"}},
        {{"source": "文献2", "quote": "相关引用"}}
    ],
    "explanation": "为什么这些信息与查询相关"
}}

重要要求：
1. summary字段必须包含150-{max_summary_length}字的详细内容
2. 必须充分利用提供的文献内容进行深入总结
3. 总结要全面、详细，不要过于简短
4. 请确保summary字段的字数在150字以上"""
        
        return prompt
    
    def _template_summarize(
        self,
        query: str,
        top_k_texts: List[str],
        top_k_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary using template-based approach.
        
        Args:
            query: Query text
            top_k_texts: List of top-k texts
            top_k_metadata: Metadata for top-k texts
            
        Returns:
            Summary dictionary
        """
        try:
            # 导入新模板
            from .templates.summary_template import generate_template_summary

            # 准备文档数据
            documents = []
            for text, metadata in zip(top_k_texts, top_k_metadata):
                documents.append({
                    "text": text,
                    "metadata": metadata
                })

            # 生成摘要
            result = generate_template_summary(
                query=query,
                documents=documents,
                max_length=400
            )

            logger.info("使用高级模板生成摘要")
            return result

        except ImportError:
            # 回退到原有方法
            logger.info("使用基础模板生成摘要")
            return self._basic_template_summarize(query, top_k_texts, top_k_metadata)

    def _basic_template_summarize(self, query, top_k_texts, top_k_metadata):
        logger.info("Using template-based summarization")

        # Extract keywords from all texts
        all_keywords = []
        for metadata in top_k_metadata:
            all_keywords.extend(metadata.get("keywords", []))

        # Get unique keywords
        unique_keywords = list(set(all_keywords))[:5]

        # Create simple summary
        summary = f"关于'{query}'的总结：\n\n"
        summary += "主要观点：\n"
        for i, text in enumerate(top_k_texts, 1):
            summary += f"{i}. {text[:100]}...\n"

        result = {
            "summary": summary[:250],
            "keywords": unique_keywords,
            "key_points": [
                f"观点{i+1}: {text[:80]}..."
                for i, text in enumerate(top_k_texts[:3])
            ],
            "evidence": [
                {
                    "source": metadata.get("title", f"文献{i+1}"),
                    "para_index": metadata.get("paragraph_index", 0),
                    "quote": text[:100]
                }
                for i, (text, metadata) in enumerate(zip(top_k_texts, top_k_metadata))
            ],
            "explanation": f"这些信息与查询'{query}'相关，提供了多个角度的观点和证据。"
        }

        return result
    
    def summarize(
        self,
        query: str,
        top_k_results: List[Dict[str, Any]],
        max_summary_length: int = 250
    ) -> Dict[str, Any]:
        """
        Generate summary from top-k results.
        
        Args:
            query: Query text
            top_k_results: List of top-k reranked results
            max_summary_length: Maximum summary length
            
        Returns:
            Summary dictionary
        """
        if not top_k_results:
            logger.warning("No results to summarize")
            return {
                "summary": "没有找到相关内容",
                "keywords": [],
                "key_points": [],
                "evidence": [],
                "explanation": "查询没有返回任何结果"
            }
        
        logger.info(f"Generating summary for query: {query[:100]}...")
        
        # Extract texts and metadata
        top_k_texts = [result.get("text", "") for result in top_k_results]
        top_k_metadata = [result.get("metadata", {}) for result in top_k_results]
        
        # Try Zhipu AI first
        if self.use_zhipu:
            prompt = self._build_prompt(query, top_k_texts, max_summary_length)
            # Use larger max_tokens to ensure sufficient space for detailed summary
            response = self._call_zhipu_api(prompt, max_tokens=2000)
            
            if response:
                try:
                    # Parse JSON response
                    result = json.loads(response)
                    logger.info("Successfully generated summary using Zhipu AI")
                    return result
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Zhipu AI response as JSON")
                    logger.warning(f"Response: {response}")
        
        # Fallback to template-based summarization
        logger.info("Falling back to template-based summarization")
        return self._template_summarize(query, top_k_texts, top_k_metadata)
    
    def save_summary(
        self,
        summary: Dict[str, Any],
        query: str,
        output_dir: str = "./output/summaries/"
    ) -> str:
        """
        Save summary to JSON file.
        
        Args:
            summary: Summary dictionary
            query: Query text
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename from query
        query_id = query[:50].replace(" ", "_").replace("/", "_")
        output_file = os.path.join(output_dir, f"{query_id}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "summary": summary,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {output_file}")
        return output_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate summaries from reranked results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with reranked results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/summaries/",
        help="Output directory for summaries"
    )
    parser.add_argument(
        "--use-zhipu",
        action="store_true",
        default=True,
        help="Use Zhipu AI for summarization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="glm-4-flash",
        help="Zhipu AI model name"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=250,
        help="Maximum summary length"
    )
    
    args = parser.parse_args()
    
    # Load reranked results
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    query = data.get("query", "")
    results = data.get("reranked_results", [])
    
    # Initialize summarization engine
    engine = SummarizationEngine(
        model_name=args.model,
        use_zhipu=args.use_zhipu
    )
    
    # Generate summary
    summary = engine.summarize(query, results, args.max_length)
    
    # Save summary
    output_file = engine.save_summary(summary, query, args.output_dir)
    
    print(f"\nGenerated Summary:")
    print(f"Query: {query}")
    print(f"Summary: {summary.get('summary', '')[:200]}...")
    print(f"Keywords: {summary.get('keywords', [])}")
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
