import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Embedding and indexing module for the Personal Knowledge Summary System.
Uses sentence-transformers for embedding and Chroma for vector storage.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config
from .logger import get_logger

logger = get_logger()


class EmbeddingManager:
    """Manages embedding generation and storage."""

    def __init__(
            self,
            model_name: str = "BAAI/bge-base-zh",  # 默认改为中文模型
            persist_dir: str = "./.chroma/",
            collection_name: str = "personal_notes",
            device: str = "cpu",
            use_cache: bool = True  # 新增参数，是否使用缓存
    ):
        """
        初始化嵌入管理器，支持中文模型
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.device = device
        self.use_cache = use_cache
        self.query_prefix = ""  # 初始化query_prefix属性

        logger.info(f"加载嵌入模型: {model_name}")

        try:
            # 检查模型缓存
            cache_path = self._get_model_cache_path(model_name)
            model_exists = self._check_model_cache(cache_path)

            if model_exists and use_cache:
                logger.info(f"使用缓存的模型: {cache_path}")
                # 从缓存加载
                self.model = SentenceTransformer(cache_path, device=device)
                # 设置查询前缀（需要根据模型类型重新设置）
                self._set_query_prefix_for_model(model_name)
            else:
                logger.info(f"从HuggingFace下载模型: {model_name}")
                # 针对中文模型的特殊处理
                self._set_query_prefix_for_model(model_name)

                # 加载模型
                self.model = SentenceTransformer(model_name, device=device)

                # 如果设置了缓存，保存模型
                if use_cache and not model_exists:
                    self._save_model_to_cache(cache_path)

            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"模型加载成功，嵌入维度: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 回退到默认模型
            logger.info("尝试加载默认模型...")
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                             device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.query_prefix = ""

        # Initialize Chroma client
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        # Initialize Chroma client
        self.client = PersistentClient(path=persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Embedding manager initialized with model: {model_name}")

    def _set_query_prefix_for_model(self, model_name: str) -> None:
        """
        根据模型名称设置查询前缀
        """
        model_name_lower = model_name.lower()

        if "bge" in model_name_lower and "zh" in model_name_lower:
            # 中文BGE模型需要查询前缀
            self.query_prefix = "为这个句子生成表示以用于检索相关文章："
            logger.info(f"设置BGE中文模型查询前缀: {self.query_prefix}")
        elif "m3e" in model_name_lower:
            self.query_prefix = ""
            logger.info("M3E模型不需要查询前缀")
        else:
            self.query_prefix = ""
            logger.info("其他模型不需要查询前缀")

    def _get_model_cache_path(self, model_name: str) -> str:
        """
        获取模型缓存路径

        Args:
            model_name: 模型名称

        Returns:
            缓存路径
        """
        # 将模型名称转换为安全的文件名
        safe_name = model_name.replace("/", "__").replace("\\", "__")
        cache_dir = Path.home() / ".cache" / "pks_models" / safe_name
        return str(cache_dir)

    def _check_model_cache(self, cache_path: str) -> bool:
        """
        检查模型是否已缓存
        改进：更宽松的检查逻辑，只要缓存目录存在且有模型文件就认为已缓存

        Args:
            cache_path: 缓存路径

        Returns:
            是否已缓存
        """
        cache_dir = Path(cache_path)

        if not cache_dir.exists():
            logger.debug(f"缓存目录不存在: {cache_path}")
            return False

        # 检查目录是否为空
        if not any(cache_dir.iterdir()):
            logger.debug(f"缓存目录为空: {cache_path}")
            return False

        # 更宽松的检查：只要目录存在且非空，就认为已缓存
        # 因为模型可能已经通过huggingface transformers下载到默认位置
        logger.info(f"检测到模型缓存目录: {cache_path}，目录包含 {len(list(cache_dir.iterdir()))} 个项目")

        # 额外检查是否有常见的模型文件
        model_files = list(cache_dir.glob("*.safetensors")) + \
                      list(cache_dir.glob("*.bin")) + \
                      list(cache_dir.glob("*.pth")) + \
                      list(cache_dir.glob("*.pt"))

        if model_files:
            logger.info(f"找到模型文件: {len(model_files)} 个")
            return True

        # 如果有配置文件，也认为可能已缓存
        config_files = list(cache_dir.glob("*.json")) + \
                       list(cache_dir.glob("*.yaml")) + \
                       list(cache_dir.glob("*.yml"))

        if config_files:
            logger.info(f"找到配置文件: {len(config_files)} 个")
            return True

        # 如果目录非空但找不到特定文件，可能是其他格式的模型
        # 让SentenceTransformer自己去判断
        return True

    def _save_model_to_cache(self, cache_path: str) -> bool:
        """
        保存模型到缓存
        改进：使用更可靠的方法获取模型路径

        Args:
            cache_path: 缓存路径

        Returns:
            是否保存成功
        """
        try:
            cache_dir = Path(cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"尝试将模型保存到缓存: {cache_path}")

            # 方法1：直接保存模型
            try:
                # SentenceTransformer有save方法
                self.model.save(cache_path)
                logger.info(f"模型已成功保存到: {cache_path}")
                return True
            except Exception as save_error:
                logger.warning(f"直接保存失败: {save_error}，尝试其他方法...")

            # 方法2：检查模型是否已经缓存到huggingface的默认位置
            from transformers import AutoModel, AutoTokenizer

            # 尝试获取模型和tokenizer
            try:
                # 保存tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                tokenizer.save_pretrained(cache_path)

                # 保存模型
                model = AutoModel.from_pretrained(self.model_name)
                model.save_pretrained(cache_path)

                logger.info(f"模型已通过transformers保存到缓存: {cache_path}")
                return True
            except Exception as transformers_error:
                logger.warning(f"通过transformers保存失败: {transformers_error}")

            # 方法3：如果以上方法都失败，只保存配置信息
            logger.warning("无法保存完整模型，仅创建标记文件")
            marker_file = cache_dir / ".model_cached"
            marker_file.touch()

            logger.info(f"创建标记文件: {marker_file}")
            return True

        except Exception as e:
            logger.error(f"保存模型到缓存失败: {e}")
            return False

    def embed_texts(
            self,
            texts: List[str],
            batch_size: int = 32,
            show_progress: bool = True,
            is_query: bool = False  # 新增参数，区分查询和文档
    ) -> np.ndarray:
        """
        生成文本嵌入

        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            is_query: 是否为查询文本（某些模型需要特殊处理）
        """
        logger.info(f"正在嵌入 {len(texts)} 个文本，是否为查询: {is_query}")

        # 处理查询前缀
        if is_query and self.query_prefix:
            texts = [self.query_prefix + text for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # 重要：归一化嵌入
        )
        return embeddings

    def add_documents(
            self,
            documents: List[Dict[str, Any]],
            batch_size: int = 32
    ) -> int:
        """
        Add documents to the Chroma collection.

        Args:
            documents: List of document dictionaries with 'text' field
            batch_size: Batch size for embedding

        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        # Extract texts
        texts = [doc.get("text", "") for doc in documents]

        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        # Prepare metadata
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            doc_id = doc.get("doc_id", f"doc_{i}")
            para_idx = doc.get("paragraph_index", 0)
            record_id = f"{doc_id}_para_{para_idx}"

            metadata = {
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "paragraph_index": str(para_idx),
                "keywords": json.dumps(doc.get("keywords", [])),
                "language": doc.get("language", ""),
                "length": str(doc.get("length", 0))
            }

            ids.append(record_id)
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            documents=texts
        )

        logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
        return len(documents)

    def load_from_jsonl(
            self,
            jsonl_file: str,
            batch_size: int = 32
    ) -> int:
        """
        Load documents from JSONL file and add to collection.

        Args:
            jsonl_file: Path to JSONL file
            batch_size: Batch size for embedding

        Returns:
            Number of documents added
        """
        logger.info(f"Loading documents from {jsonl_file}")

        documents = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from JSONL")

        return self.add_documents(documents, batch_size=batch_size)

    def query(
            self,
            query_text: str,
            top_k: int = 5,
            where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        查询集合中的相似文档
        """
        # 生成查询嵌入（注意 is_query=True）
        query_embedding = self.embed_texts([query_text], is_query=True)[0]

        # 查询集合
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "distances", "metadatas"]
        )

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Collection information
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection '{self.collection_name}'")

    def persist(self) -> None:
        """Persist the collection to disk (handled automatically by PersistentClient)."""
        logger.info("Collection persistence is handled automatically by PersistentClient.")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build vector index"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/processed/processed.jsonl",
        help="Input JSONL file with processed documents"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="personal_notes",
        help="Name of the Chroma collection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence-transformers model"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./.chroma/",
        help="Directory for Chroma persistence"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda'],
        help="Device for embedding"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable model caching"
    )

    args = parser.parse_args()

    # Initialize embedding manager
    manager = EmbeddingManager(
        model_name=args.model,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        device=args.device,
        use_cache=not args.no_cache
    )

    # Load documents and build index
    if Path(args.input).exists():
        count = manager.load_from_jsonl(args.input, batch_size=args.batch_size)

        # Print collection info
        info = manager.get_collection_info()
        print("\nCollection Information:")
        print(f"  Collection Name: {info['collection_name']}")
        print(f"  Document Count: {info['document_count']}")
        print(f"  Embedding Dimension: {info['embedding_dim']}")
        print(f"  Model: {info['model_name']}")
    else:
        print(f"Input file not found: {args.input}")


if __name__ == "__main__":
    main()