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
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from functools import lru_cache

from .logger import get_logger

logger = get_logger()


class APICache:
    """Cache manager for API responses with memory and disk persistence."""

    def __init__(
            self,
            cache_dir: str = "./cache/zhipu_api/",
            max_memory_size: int = 100,
            disk_ttl: int = 86400  # 24 hours in seconds
    ):
        """
        Initialize API cache.

        Args:
            cache_dir: Directory for disk cache
            max_memory_size: Maximum number of items in memory cache
            disk_ttl: Time-to-live for disk cache in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_ttl = disk_ttl
        self.memory_cache = {}
        self.max_memory_size = max_memory_size
        self.memory_access_order = []  # Track access order for LRU

        logger.info(f"API cache initialized: memory={max_memory_size}, disk_ttl={disk_ttl}s")

    def _generate_cache_key(self, query: str, texts: List[str]) -> str:
        """
        Generate unique cache key from query and texts.

        Args:
            query: Query text
            texts: List of texts

        Returns:
            SHA256 hash as cache key
        """
        combined = query + "||".join(texts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get file path for disk cache."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid (not expired)."""
        if not cache_file.exists():
            return False

        file_time = cache_file.stat().st_mtime
        current_time = time.time()
        return (current_time - file_time) < self.disk_ttl

    def _update_lru(self, cache_key: str):
        """Update LRU access order."""
        if cache_key in self.memory_access_order:
            self.memory_access_order.remove(cache_key)
        self.memory_access_order.append(cache_key)

        # Enforce LRU eviction
        if len(self.memory_access_order) > self.max_memory_size:
            oldest_key = self.memory_access_order.pop(0)
            self.memory_cache.pop(oldest_key, None)

    def get(self, query: str, texts: List[str]) -> Optional[str]:
        """
        Get cached response if available.

        Args:
            query: Query text
            texts: List of texts

        Returns:
            Cached response or None
        """
        cache_key = self._generate_cache_key(query, texts)

        # Check memory cache first
        if cache_key in self.memory_cache:
            self._update_lru(cache_key)
            logger.debug(f"Memory cache hit for key: {cache_key[:16]}...")
            return self.memory_cache[cache_key]

        # Check disk cache
        cache_file = self._get_cache_file_path(cache_key)
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    response = cache_data.get('response')

                    if response:
                        # Load into memory cache
                        self.memory_cache[cache_key] = response
                        self._update_lru(cache_key)
                        logger.debug(f"Disk cache hit for key: {cache_key[:16]}...")
                        return response
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read cache file: {e}")

        logger.debug(f"Cache miss for key: {cache_key[:16]}...")
        return None

    def set(self, query: str, texts: List[str], response: str):
        """
        Store response in cache.

        Args:
            query: Query text
            texts: List of texts
            response: API response to cache
        """
        cache_key = self._generate_cache_key(query, texts)

        # Store in memory cache
        self.memory_cache[cache_key] = response
        self._update_lru(cache_key)

        # Store in disk cache
        cache_file = self._get_cache_file_path(cache_key)
        try:
            cache_data = {
                'response': response,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'cache_key': cache_key
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cached response to disk: {cache_key[:16]}...")
        except IOError as e:
            logger.warning(f"Failed to write cache file: {e}")

    def clear(self):
        """Clear all cache (memory and disk)."""
        # Clear memory cache
        self.memory_cache.clear()
        self.memory_access_order.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.json"))
        valid_disk_files = [f for f in disk_files if self._is_cache_valid(f)]

        return {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_limit': self.max_memory_size,
            'disk_cache_size': len(valid_disk_files),
            'cache_dir': str(self.cache_dir)
        }


class SummarizationEngine:
    """Handles summarization using Zhipu AI or local templates."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "glm-4-flash",
            use_zhipu: bool = True,
            max_retries: int = 2,
            enable_cache: bool = True,
            cache_dir: str = "./cache/zhipu_api/",
            cache_ttl: int = 86400
    ):
        """
        Initialize summarization engine.

        Args:
            api_key: Zhipu AI API key
            model_name: Name of the Zhipu AI model
            use_zhipu: Whether to use Zhipu AI (fallback to template if False)
            max_retries: Maximum number of retries for API calls
            enable_cache: Whether to enable API response caching
            cache_dir: Directory for cache storage
            cache_ttl: Cache time-to-live in seconds (default 24 hours)
        """
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        self.model_name = model_name
        self.use_zhipu = use_zhipu and bool(self.api_key)
        self.max_retries = max_retries
        self.api_base = "https://open.bigmodel.cn/api/paas/v4"
        self.enable_cache = enable_cache

        # Initialize cache if enabled
        self.cache = APICache(cache_dir=cache_dir, disk_ttl=cache_ttl) if enable_cache else None

        if self.use_zhipu:
            cache_status = "enabled" if enable_cache else "disabled"
            logger.info(f"Summarization engine initialized with Zhipu AI model: {model_name}, cache={cache_status}")
        else:
            logger.info("Summarization engine initialized with template-based mode")

    def _call_zhipu_api(
            self,
            prompt: str,
            query: str,
            texts: List[str],
            temperature: float = 0.7,
            max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Call Zhipu AI API for summarization with caching support.

        Args:
            prompt: Input prompt
            query: Query text (for cache key generation)
            texts: List of texts (for cache key generation)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text or None if failed
        """
        if not self.api_key:
            logger.warning("Zhipu API key not available")
            return None

        # Check cache first if enabled
        if self.enable_cache and self.cache:
            cached_response = self.cache.get(query, texts)
            if cached_response:
                logger.info("Cache hit - returning cached response")
                return cached_response

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
                        api_response = result["choices"][0]["message"]["content"]

                        # Cache the response if enabled
                        if self.enable_cache and self.cache:
                            self.cache.set(query, texts, api_response)
                            logger.info("Response cached for future use")

                        return api_response
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
        context = "\n\n".join([f"[文献{i + 1}]\n{text}" for i, text in enumerate(top_k_texts)])

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
                f"观点{i + 1}: {text[:80]}..."
                for i, text in enumerate(top_k_texts[:3])
            ],
            "evidence": [
                {
                    "source": metadata.get("title", f"文献{i + 1}"),
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
            response = self._call_zhipu_api(prompt, query, top_k_texts, max_tokens=2000)

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

    def clear_cache(self):
        """Clear API cache."""
        if self.cache:
            self.cache.clear()
            logger.info("API cache cleared")
        else:
            logger.warning("Cache is not enabled")

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics.

        Returns:
            Cache statistics dictionary or None if cache is disabled
        """
        if self.cache:
            return self.cache.get_stats()
        return None


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
    parser.add_argument(
        "--enable-cache",
        action="store_true",
        default=True,
        help="Enable API response caching"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/zhipu_api/",
        help="Directory for cache storage"
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=86400,
        help="Cache time-to-live in seconds (default 24 hours)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before processing"
    )
    parser.add_argument(
        "--show-cache-stats",
        action="store_true",
        help="Show cache statistics"
    )

    args = parser.parse_args()

    # Initialize summarization engine
    engine = SummarizationEngine(
        model_name=args.model,
        use_zhipu=args.use_zhipu,
        enable_cache=args.enable_cache,
        cache_dir=args.cache_dir,
        cache_ttl=args.cache_ttl
    )

    # Clear cache if requested
    if args.clear_cache:
        engine.clear_cache()

    # Show cache stats if requested
    if args.show_cache_stats:
        stats = engine.get_cache_stats()
        if stats:
            print(f"\nCache Statistics:")
            print(f"  Memory cache: {stats['memory_cache_size']}/{stats['memory_cache_limit']}")
            print(f"  Disk cache: {stats['disk_cache_size']} files")
            print(f"  Cache directory: {stats['cache_dir']}")
        else:
            print("\nCache is disabled")

    # Load reranked results
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    query = data.get("query", "")
    results = data.get("reranked_results", [])

    # Generate summary
    summary = engine.summarize(query, results, args.max_length)

    # Save summary
    output_file = engine.save_summary(summary, query, args.output_dir)

    print(f"\nGenerated Summary:")
    print(f"Query: {query}")
    print(f"Summary: {summary.get('summary', '')[:200]}...")
    print(f"Keywords: {summary.get('keywords', [])}")
    print(f"\nSummary saved to: {output_file}")

    # Show final cache stats
    if args.show_cache_stats:
        stats = engine.get_cache_stats()
        if stats:
            print(f"\nFinal Cache Statistics:")
            print(f"  Memory cache: {stats['memory_cache_size']}/{stats['memory_cache_limit']}")
            print(f"  Disk cache: {stats['disk_cache_size']} files")


if __name__ == "__main__":
    main()
