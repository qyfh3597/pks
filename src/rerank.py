import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Hybrid reranking module for the Personal Knowledge Summary System.
Implements feature-based reranking with manual and learned weights.
"""

import json
import pickle
import jieba
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import csv

from .embed import EmbeddingManager
from .logger import get_logger

logger = get_logger()


class HybridReranker:
    """Implements hybrid reranking with multiple features."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        weights: Optional[Dict[str, float]] = None,
        learned_model_path: Optional[str] = None
    ):
        """
        Initialize reranker.
        
        Args:
            embedding_manager: EmbeddingManager instance
            weights: Manual weights for features
            learned_model_path: Path to learned model
        """
        self.embedding_manager = embedding_manager
        
        # Default weights
        self.default_weights = {
            "semantic_sim": 0.55,
            "keyword_overlap": 0.25,
            "title_overlap": 0.08,
            "position_score": 0.07,
            "length_score": 0.05
        }
        
        self.weights = weights or self.default_weights
        self.learned_model = None
        self.scaler = None
        
        if learned_model_path and Path(learned_model_path).exists():
            self.load_model(learned_model_path)
    
    def compute_semantic_sim(
        self,
        query: str,
        candidate_text: str
    ) -> float:
        """
        Compute semantic similarity between query and candidate.
        
        Args:
            query: Query text
            candidate_text: Candidate text
            
        Returns:
            Similarity score (0-1)
        """
        query_emb = self.embedding_manager.model.encode(query, convert_to_numpy=True)
        candidate_emb = self.embedding_manager.model.encode(candidate_text, convert_to_numpy=True)
        
        # Cosine similarity
        similarity = np.dot(query_emb, candidate_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(candidate_emb) + 1e-8
        )
        
        return float(max(0, similarity))  # Ensure non-negative
    
    def compute_keyword_overlap(
        self,
        query: str,
        candidate_keywords: List[str],
        candidate_text: str
    ) -> float:
        """
        Compute keyword overlap between query and candidate.
        
        Args:
            query: Query text
            candidate_keywords: Keywords from candidate
            candidate_text: Candidate text
            
        Returns:
            Overlap score (0-1)
        """
        # Extract query keywords (simple word-based)
        # Use jieba for Chinese, split for English
        try:
            # 1. 提取查询关键词（改进版）
            query_keywords = self._extract_keywords(query, is_query=True)

            # 2. 候选关键词（来自预处理）
            candidate_set = set(candidate_keywords)

            # 3. 如果候选关键词太少，从文本补充
            if len(candidate_set) < 2:
                additional = self._extract_keywords(candidate_text, is_query=False)
                candidate_set.update(additional)

            # 4. 计算重叠
            if not query_keywords or not candidate_set:
                return 0.0

            overlap_count = 0
            total_weight = 0

            for q_word in query_keywords:
                word_weight = 1.0
                # 较长的查询词权重更高
                if len(q_word) >= 3:
                    word_weight = 1.2

                found = False
                for c_word in candidate_set:
                    # 完全匹配
                    if q_word.lower() == c_word.lower():
                        overlap_count += word_weight
                        found = True
                        break
                    # 部分匹配（包含关系）
                    elif q_word.lower() in c_word.lower() or c_word.lower() in q_word.lower():
                        overlap_count += word_weight * 0.5
                        found = True
                        break

                total_weight += word_weight

            # 归一化
            if total_weight > 0:
                score = overlap_count / total_weight
                return min(score, 1.0)
            return 0.0

        except Exception as e:
            logger.warning(f"Error computing keyword overlap: {e}")
            return 0.0

    def _extract_keywords(self, text: str, is_query: bool = False) -> List[str]:
        """提取关键词的辅助方法"""
        if not text:
            return []

        # 简单的停用词列表
        stopwords_cn = {'的', '了', '是', '在', '和', '有', '一', '这', '不', '人'}
        stopwords_en = {'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'for', 'on'}

        keywords = []

        # 检测语言（简单版）
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

        if has_chinese:
            # 中文分词
            import jieba
            words = jieba.cut(text)
            for word in words:
                word = word.strip()
                if (len(word) > 1 and
                        word not in stopwords_cn and
                        any('\u4e00' <= char <= '\u9fff' for char in word)):
                    keywords.append(word)
        else:
            # 英文处理
            import re
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            for word in words:
                if word not in stopwords_en:
                    keywords.append(word)

        # 查询关键词取前5个，文档关键词取前10个
        max_keywords = 5 if is_query else 10
        return keywords[:max_keywords]
    
    def compute_title_overlap(
        self,
        query: str,
        title: str
    ) -> float:
        """
        Compute overlap between query and document title.
        
        Args:
            query: Query text
            title: Document title
            
        Returns:
            Overlap score (0-1)
        """
        # Use jieba for Chinese, split for English
        if not title or not query:
            return 0.0

            # 转换为小写，去掉标点
        import re
        query_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query.lower())
        title_clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', title.lower())

        # 分词
        if any('\u4e00' <= char <= '\u9fff' for char in query_clean):
            # 中文分词
            import jieba
            query_words = set(jieba.cut(query_clean))
            title_words = set(jieba.cut(title_clean))
        else:
            # 英文分词
            query_words = set(query_clean.split())
            title_words = set(title_clean.split())

        # 过滤停用词
        stopwords = {'的', '与', '和', '在', '是', '了', '有', '一', '这', '不'}
        query_words = {w for w in query_words if w not in stopwords and len(w) > 1}
        title_words = {w for w in title_words if w not in stopwords and len(w) > 1}

        if not query_words or not title_words:
            return 0.0

        # 计算重合度
        overlap = len(query_words & title_words)
        return overlap / len(query_words)
    
    def compute_position_score(
        self,
        paragraph_index: int,
        total_paragraphs: int =None,
        doc_length: int = None
    ) -> float:
        """
        Compute position score (earlier paragraphs get higher scores).
        
        Args:
            paragraph_index: Index of the paragraph
            total_paragraphs: Total number of paragraphs in document
            
        Returns:
            Position score (0-1)
        """
        if paragraph_index == 0 and doc_length:
            # 使用文档长度来估算"位置"
            # 假设较长的文档内容靠后，较短的靠前
            if doc_length < 500:
                return 0.9  # 短文档，位置靠前
            elif doc_length < 1500:
                return 0.7  # 中等长度
            elif doc_length < 3000:
                return 0.5  # 较长文档
            else:
                return 0.3  # 很长文档
        elif paragraph_index > 0:
            # 正常计算（如果有多个段落）
            if total_paragraphs is None:
                total_paragraphs = max(paragraph_index + 1, 10)

            normalized_pos = paragraph_index / max(total_paragraphs - 1, 1)
            score = np.exp(-1.5 * normalized_pos)
            return float(score)
        else:
            # 默认情况
            return 0.5
    
    def compute_length_score(
        self,
        text_length: int,
        mean_length: float = None,
        sigma: float = None
    ) -> float:
        """
        Compute length score (Gaussian distribution around mean).
        
        Args:
            text_length: Length of text
            mean_length: Mean length
            sigma: Standard deviation
            
        Returns:
            Length score (0-1)
        """
        try:
            # 针对你的数据调整参数
            if mean_length is None:
                # 根据你的文档调整（假设平均长度800-1500字）
                mean_length = 1200

            if sigma is None:
                # 标准差设为均值的一半
                sigma = mean_length * 0.5

            if text_length <= 0:
                return 0.0

            # 归一化处理
            # 使用sigmoid函数而不是高斯，更稳定
            # score = 1 / (1 + exp(-(text_length - mean_length)/sigma))
            z = (text_length - mean_length) / max(sigma, 1)
            score = 1.0 / (1.0 + np.exp(-z))

            # 确保在合理范围内
            return float(max(0.1, min(score, 1.0)))

        except Exception as e:
            logger.warning(f"Error computing length score: {e}")
            return 0.5  # 默认返回中等分数
    
    def extract_features(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, float]]:
        """
        Extract features for all candidates.
        
        Args:
            query: Query text
            candidates: List of candidate documents
            query_embedding: Pre-computed query embedding
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for candidate in candidates:
            text = candidate.get("text", "")
            metadata = candidate.get("metadata", {})

            # The metadata already contains 'keywords' and 'title' from preprocess.py
            doc_length = metadata.get("length", 0)
            para_index = metadata.get("paragraph_index", 0)
            features = {
                "semantic_sim": candidate.get("similarity_score", 0.0),
                "keyword_overlap": self.compute_keyword_overlap(
                    query,
                    metadata.get("keywords", []), # Keywords from preprocess.py
                    text
                ),
                "title_overlap": self.compute_title_overlap(
                    query,
                    metadata.get("title", "") # Title from preprocess.py
                ),
                "position_score": self.compute_position_score(
                    paragraph_index=para_index,
                    total_paragraphs=None,
                    doc_length=doc_length
                ),
                "length_score": self.compute_length_score(doc_length)
            }

            features_list.append(features)

        return features_list
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using hybrid features.
        
        Args:
            query: Query text
            candidates: List of candidate documents
            top_k: Number of top results to return
            mode: Reranking mode ('hybrid', 'learned', or 'manual')
            
        Returns:
            Reranked candidates with scores and explanations
        """
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates using mode: {mode}")
        
        # Extract features
        features_list = self.extract_features(query, candidates)
        
        # Compute reranking scores
        reranked = []
        
        for i, (candidate, features) in enumerate(zip(candidates, features_list)):
            if mode == "learned" and self.learned_model:
                # Use learned model
                feature_vector = np.array([
                    features["semantic_sim"],
                    features["keyword_overlap"],
                    features["title_overlap"],
                    features["position_score"],
                    features["length_score"]
                ]).reshape(1, -1)
                
                if self.scaler:
                    feature_vector = self.scaler.transform(feature_vector)
                
                score = float(self.learned_model.predict(feature_vector)[0])
            else:
                # Use manual weights
                score = sum(
                    features.get(feat, 0) * self.weights.get(feat, 0)
                    for feat in self.weights.keys()
                )
            
            # Compute feature contributions
            contributions = {
                feat: features.get(feat, 0) * self.weights.get(feat, 0)
                for feat in self.weights.keys()
            }
            
            reranked_candidate = {
                **candidate,
                "rerank_score": float(score),
                "features": features,
                "contributions": contributions,
                "weights_used": self.weights
            }
            
            reranked.append(reranked_candidate)
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Update ranks
        for i, candidate in enumerate(reranked[:top_k]):
            candidate["rerank"] = i + 1
        
        logger.info(f"Reranking complete. Top-{top_k} selected.")
        
        return reranked[:top_k]
    
    def train_model(
        self,
        training_data: List[Dict[str, Any]],
        labels: List[float],
        model_type: str = "linear"
    ) -> Tuple[float, float]:
        """
        Train a reranking model on labeled data.
        
        Args:
            training_data: List of training candidates
            labels: List of relevance labels
            model_type: Type of model ('linear' or 'mlp')
            
        Returns:
            Tuple of (train_score, validation_score)
        """
        logger.info(f"Training reranking model with {len(training_data)} samples")
        
        # Extract features
        features_list = []
        for candidate in training_data:
            features = {
                "semantic_sim": candidate.get("similarity_score", 0.0),
                "keyword_overlap": candidate.get("keyword_overlap", 0.0),
                "title_overlap": candidate.get("title_overlap", 0.0),
                "position_score": candidate.get("position_score", 0.0),
                "length_score": candidate.get("length_score", 0.0)
            }
            features_list.append(features)
        
        # Convert to feature matrix
        feature_names = ["semantic_sim", "keyword_overlap", "title_overlap", "position_score", "length_score"]
        X = np.array([[f.get(name, 0) for name in feature_names] for f in features_list])
        y = np.array(labels)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if model_type == "linear":
            self.learned_model = LinearRegression()
            self.learned_model.fit(X_scaled, y)
            
            # Extract weights
            learned_weights = {}
            for name, weight in zip(feature_names, self.learned_model.coef_):
                learned_weights[name] = float(weight)
            
            self.weights = learned_weights
            logger.info(f"Learned weights: {learned_weights}")
        
        # Compute score
        train_score = self.learned_model.score(X_scaled, y)
        logger.info(f"Model R² score: {train_score:.4f}")
        
        return train_score, train_score
    
    def save_model(self, model_path: str) -> None:
        """Save the learned model to disk."""
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.learned_model,
            "scaler": self.scaler,
            "weights": self.weights
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a learned model from disk."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.learned_model = model_data.get("model")
        self.scaler = model_data.get("scaler")
        self.weights = model_data.get("weights", self.default_weights)
        
        logger.info(f"Model loaded from {model_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Rerank retrieval results using hybrid features"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with retrieval results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/rerank_results.json",
        help="Output file path"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top results to return"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=['hybrid', 'learned', 'manual'],
        help="Reranking mode"
    )
    
    args = parser.parse_args()
    
    # Load retrieval results
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    query = data.get("query", "")
    candidates = data.get("candidates", [])
    
    # Initialize embedding manager and reranker
    embedding_manager = EmbeddingManager()
    reranker = HybridReranker(embedding_manager)
    
    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=args.topk, mode=args.mode)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "query": query,
            "num_results": len(reranked),
            "reranked_results": reranked
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Reranked results saved to {args.output}")


if __name__ == "__main__":
    main()
