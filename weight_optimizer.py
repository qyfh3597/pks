#!/usr/bin/env python3
"""
权重优化工具 - 简化版
通过简单的相关性评分自动学习最优权重
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import minimize

from src.embed import EmbeddingManager
from src.retrieval import Retriever
from src.rerank import HybridReranker
from src.logger import get_logger

logger = get_logger()


class WeightOptimizer:
    """权重优化器"""

    def __init__(self):
        """初始化优化器"""
        self.embedding_manager = EmbeddingManager()
        self.retriever = Retriever(self.embedding_manager)
        logger.info("Weight optimizer initialized")

    def collect_training_data(
            self,
            queries: List[str],
            output_file: str = "./data/labels/training_data.json"
    ):
        """
        收集训练数据（交互式标注）

        Args:
            queries: 查询列表
            output_file: 输出文件路径
        """
        training_data = []

        print("\n" + "=" * 60)
        print("📝 Interactive Labeling for Weight Optimization")
        print("=" * 60)
        print("\nFor each result, rate its relevance (0-5):")
        print("  0 = Not relevant")
        print("  1 = Slightly relevant")
        print("  2 = Moderately relevant")
        print("  3 = Relevant")
        print("  4 = Very relevant")
        print("  5 = Highly relevant")
        print("\nPress Ctrl+C to stop and save.\n")

        try:
            for query in queries:
                print(f"\n{'=' * 60}")
                print(f"Query: {query}")
                print('=' * 60)

                # 检索候选文档
                candidates = self.retriever.retrieve(query, top_k=10)

                if not candidates:
                    print("No candidates found. Skipping...")
                    continue

                query_data = {
                    "query": query,
                    "candidates": []
                }

                for i, candidate in enumerate(candidates, 1):
                    print(f"\n[{i}/{len(candidates)}]")
                    print(f"Text: {candidate['text'][:200]}...")
                    print(f"Title: {candidate['metadata'].get('title', 'N/A')}")

                    while True:
                        try:
                            score = input("Relevance (0-5, or 's' to skip): ").strip()

                            if score.lower() == 's':
                                break

                            score = int(score)
                            if 0 <= score <= 5:
                                query_data["candidates"].append({
                                    "text": candidate["text"],
                                    "metadata": candidate["metadata"],
                                    "similarity_score": candidate.get("similarity_score", 0),
                                    "relevance_score": score
                                })
                                break
                            else:
                                print("Please enter a number between 0 and 5")
                        except ValueError:
                            print("Invalid input. Please enter a number or 's'")

                if query_data["candidates"]:
                    training_data.append(query_data)
                    print(f"\n✅ Labeled {len(query_data['candidates'])} candidates for this query")

        except KeyboardInterrupt:
            print("\n\n⏸️  Labeling interrupted")

        # 保存训练数据
        if training_data:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Training data saved: {output_file}")
            print(f"   Total queries: {len(training_data)}")
            print(f"   Total labels: {sum(len(q['candidates']) for q in training_data)}")
        else:
            print("\n⚠️  No training data collected")

    def optimize_weights(
            self,
            training_file: str = "./data/labels/training_data.json",
            output_file: str = "./output/models/optimized_weights.json"
    ) -> Dict[str, float]:
        """
        优化权重

        Args:
            training_file: 训练数据文件
            output_file: 输出权重文件

        Returns:
            优化后的权重字典
        """
        print("\n🔧 Optimizing weights...\n")

        # 加载训练数据
        with open(training_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        # 准备特征和标签
        X = []  # 特征矩阵
        y = []  # 标签向量

        reranker = HybridReranker(self.embedding_manager)

        for query_data in training_data:
            query = query_data["query"]
            candidates = query_data["candidates"]

            # 提取特征
            features_list = reranker.extract_features(query, candidates)

            for features, candidate in zip(features_list, candidates):
                feature_vector = [
                    features["semantic_sim"],
                    features["keyword_overlap"],
                    features["title_overlap"],
                    features["position_score"],
                    features["length_score"]
                ]
                X.append(feature_vector)
                y.append(candidate["relevance_score"])

        X = np.array(X)
        y = np.array(y)

        print(f"Training samples: {len(X)}")

        # 定义优化目标函数（最小化预测误差）
        def objective(weights):
            # 确保权重为正且和为1
            if np.any(weights < 0) or np.sum(weights) == 0:
                return 1e10

            weights = weights / np.sum(weights)  # 归一化

            # 计算预测分数
            y_pred = X @ weights

            # 计算均方误差
            mse = np.mean((y_pred - y) ** 2)

            return mse

        # 初始权重
        initial_weights = np.array([0.5, 0.15, 0.1, 0.15, 0.1])

        # 约束：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # 边界：权重在0到1之间
        bounds = [(0, 1) for _ in range(5)]

        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # 提取优化后的权重
        optimized_weights = result.x
        optimized_weights = optimized_weights / np.sum(optimized_weights)  # 归一化

        weights_dict = {
            "semantic_sim": float(optimized_weights[0]),
            "keyword_overlap": float(optimized_weights[1]),
            "title_overlap": float(optimized_weights[2]),
            "position_score": float(optimized_weights[3]),
            "length_score": float(optimized_weights[4])
        }

        # 计算优化前后的性能
        initial_mse = objective(initial_weights)
        optimized_mse = result.fun

        print("\n📊 Optimization Results:")
        print(f"  Initial MSE: {initial_mse:.4f}")
        print(f"  Optimized MSE: {optimized_mse:.4f}")
        print(f"  Improvement: {(1 - optimized_mse / initial_mse) * 100:.2f}%")

        print("\n🎯 Optimized Weights:")
        for feature, weight in weights_dict.items():
            print(f"  {feature}: {weight:.4f}")

        # 保存权重
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "weights": weights_dict,
                "initial_mse": float(initial_mse),
                "optimized_mse": float(optimized_mse),
                "improvement": float((1 - optimized_mse / initial_mse) * 100)
            }, f, indent=2)

        print(f"\n✅ Weights saved: {output_file}")

        return weights_dict

    def quick_optimize(
            self,
            queries: List[str],
            num_samples_per_query: int = 5
    ) -> Dict[str, float]:
        """
        快速优化（自动标注 + 优化）

        Args:
            queries: 查询列表
            num_samples_per_query: 每个查询标注的样本数

        Returns:
            优化后的权重
        """
        print("\n🚀 Quick Optimization Mode\n")
        print("This will automatically collect labels and optimize weights.")
        print(f"Queries: {len(queries)}")
        print(f"Samples per query: {num_samples_per_query}")

        # 收集训练数据
        training_file = "./data/labels/training_data.json"
        self.collect_training_data(queries, training_file)

        # 优化权重
        if Path(training_file).exists():
            weights = self.optimize_weights(training_file)
            return weights
        else:
            print("\n❌ No training data available for optimization")
            return {}

    def apply_weights(
            self,
            weights_file: str = "./output/models/optimized_weights.json",
            config_file: str = "./config.json"
    ):
        """
        应用优化后的权重到配置文件

        Args:
            weights_file: 权重文件路径
            config_file: 配置文件路径
        """
        print(f"\n📝 Applying weights to config...\n")

        # 加载权重
        with open(weights_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            weights = data["weights"]

        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 更新权重
        config["reranking"]["manual_weights"] = weights

        # 保存配置
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print("✅ Weights applied to config.json")
        print("\n🎯 New Weights:")
        for feature, weight in weights.items():
            print(f"  {feature}: {weight:.4f}")
        print()


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="Weight Optimizer for Reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect training data
  python weight_optimizer.py collect --queries queries.txt

  # Optimize weights
  python weight_optimizer.py optimize

  # Apply optimized weights
  python weight_optimizer.py apply

  # Quick optimization (all-in-one)
  python weight_optimizer.py quick --queries queries.txt
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # collect命令
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--queries', required=True, help='File with queries (one per line)')
    collect_parser.add_argument('--output', default='./data/labels/training_data.json', help='Output file')

    # optimize命令
    optimize_parser = subparsers.add_parser('optimize', help='Optimize weights')
    optimize_parser.add_argument('--input', default='./data/labels/training_data.json', help='Training data file')
    optimize_parser.add_argument('--output', default='./output/models/optimized_weights.json', help='Output file')

    # apply命令
    apply_parser = subparsers.add_parser('apply', help='Apply optimized weights to config')
    apply_parser.add_argument('--weights', default='./output/models/optimized_weights.json', help='Weights file')
    apply_parser.add_argument('--config', default='./config.json', help='Config file')

    # quick命令
    quick_parser = subparsers.add_parser('quick', help='Quick optimization (collect + optimize + apply)')
    quick_parser.add_argument('--queries', required=True, help='File with queries (one per line)')
    quick_parser.add_argument('--samples', type=int, default=5, help='Samples per query')

    args = parser.parse_args()

    optimizer = WeightOptimizer()

    if args.command == 'collect':
        # 读取查询
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        optimizer.collect_training_data(queries, args.output)

    elif args.command == 'optimize':
        optimizer.optimize_weights(args.input, args.output)

    elif args.command == 'apply':
        optimizer.apply_weights(args.weights, args.config)

    elif args.command == 'quick':
        # 读取查询
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        weights = optimizer.quick_optimize(queries, args.samples)

        if weights:
            # 自动应用权重
            weights_file = './output/models/optimized_weights.json'
            optimizer.apply_weights(weights_file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
