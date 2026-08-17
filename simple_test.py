#!/usr/bin/env python3
"""
测试连接复用效果
演示第一次调用建立连接后，后续调用复用连接变快
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import time
import os

# 导入项目模块
from src.config import get_config
from src.summarize1 import SummarizationEngine


def test_connection_reuse():
    """测试连接复用效果"""
    print("=" * 80)
    print("连接复用测试")
    print("=" * 80)

    # 检查API密钥
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到ZHIPU_API_KEY环境变量")
        return False

    print(f"✓ API密钥已设置")

    # 加载配置
    config = get_config("config.json")
    print(f"✓ 配置加载成功")

    # 初始化引擎（启用连接池）
    summarizer = SummarizationEngine(
        model_name=config.get("summarization.zhipu_model"),
        use_zhipu=True,
        max_retries=2,
        timeout=30,
        enable_cache=False  # 禁用缓存，只测试连接复用
    )

    print(f"✓ 引擎初始化成功（连接池已启用）")
    print()

    # 准备不同的测试查询
    test_queries = [
        "什么是机器学习",
        "什么是深度学习",
        "什么是人工智能",
        "什么是神经网络",
        "什么是监督学习"
    ]

    test_results = []

    # 测试不同的查询
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"测试 {i}/{len(test_queries)}: {query}")
        print(f"{'=' * 80}")

        # 准备测试数据（使用相同的数据结构，但查询不同）
        test_data = [
            {
                "text": f"这是关于{query}的详细说明。{query}是计算机科学的重要分支。",
                "metadata": {
                    "title": f"{query}介绍",
                    "keywords": [query, "计算机科学"],
                    "paragraph_index": 0
                },
                "rerank_score": 0.95
            }
        ]

        start_time = time.time()

        try:
            summary = summarizer.summarize(
                query=query,
                top_k_results=test_data,
                max_summary_length=250
            )

            elapsed_time = time.time() - start_time

            test_results.append({
                "query": query,
                "time": elapsed_time,
                "success": True
            })

            print(f"✓ 成功")
            print(f"  耗时: {elapsed_time:.3f}秒")
            print(f"  摘要: {summary.get('summary', '')[:80]}...")

        except Exception as e:
            elapsed_time = time.time() - start_time
            test_results.append({
                "query": query,
                "time": elapsed_time,
                "success": False,
                "error": str(e)
            })
            print(f"✗ 失败: {e}")

        # 请求间隔
        if i < len(test_queries):
            time.sleep(1)

    # 分析结果
    print("\n" + "=" * 80)
    print("测试结果分析")
    print("=" * 80)

    successful = [r for r in test_results if r['success']]

    if len(successful) < 2:
        print("⚠️  成功的请求太少，无法分析连接复用效果")
        return False

    print(f"\n总请求数: {len(test_results)}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(test_results) - len(successful)}")

    print(f"\n响应时间:")
    for i, result in enumerate(successful, 1):
        marker = "🔥" if i == 1 else "✓"
        print(f"  {marker} 第{i}次 ({result['query']}): {result['time']:.3f}秒")

    # 计算统计
    times = [r['time'] for r in successful]
    first_time = times[0]
    avg_subsequent_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else 0
    improvement = ((first_time - avg_subsequent_time) / first_time * 100) if first_time > 0 else 0

    print(f"\n统计:")
    print(f"  第一次请求: {first_time:.3f}秒")
    print(f"  后续平均: {avg_subsequent_time:.3f}秒")

    if improvement > 0:
        print(f"  ⚡ 速度提升: {improvement:.1f}%")
    elif improvement < 0:
        print(f"  ⚠️  后续反而慢了 {-improvement:.1f}% (可能是API服务器负载变化)")
    else:
        print(f"  = 速度相同")

    # 分析连接复用效果
    print("\n" + "=" * 80)
    print("连接复用效果分析")
    print("=" * 80)

    if improvement > 10:
        print("✅ 连接复用效果明显！")
        print("   第一次请求建立了TCP连接和TLS握手，后续请求复用了连接。")
        print(f"   节省了约{first_time - avg_subsequent_time:.2f}秒的连接建立时间。")
    elif improvement > 0:
        print("✓ 连接复用有一定效果")
        print("   后续请求略有提升，但可能受到API服务器响应时间影响。")
    else:
        print("⚠️  连接复用效果不明显")
        print("   可能原因:")
        print("   - API服务器响应时间波动较大")
        print("   - 网络连接已经优化过")
        print("   - 连接复用的优势被其他因素掩盖")

    # 获取性能统计
    stats = summarizer.get_performance_stats()
    print(f"\n性能统计:")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  平均时间: {stats['avg_time']:.3f}秒")
    print(f"  最快: {stats['min_time']:.3f}秒")
    print(f"  最慢: {stats['max_time']:.3f}秒")

    # 关闭会话
    summarizer.close()
    print(f"\n✓ 会话已关闭")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_connection_reuse()
    sys.exit(0 if success else 1)