#!/usr/bin/env python3
"""
验证所有导入和方法调用是否正确
"""
import sys
from pathlib import Path

print("=" * 60)
print("PKS Import Verification")
print("=" * 60)
print()

errors = []

# 测试1: 导入核心模块
print("1. Testing core module imports...")
try:
    from src.config import get_config
    from src.embed import EmbeddingManager
    from src.retrieval import Retriever
    from src.rerank import HybridReranker
    from src.summarize1 import SummarizationEngine
    from src.explain import ExplainabilityEngine
    from src.logger import get_logger
    print("   ✅ All core modules imported successfully")
except Exception as e:
    errors.append(f"Core module import failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试2: 检查配置
print("\n2. Testing configuration...")
try:
    config_obj = get_config()
    config = config_obj.config
    
    # 检查关键配置项
    assert "embedding" in config
    assert "chroma" in config
    assert "summarization" in config
    assert "persist_dir" in config["chroma"]
    assert "collection_name" in config["chroma"]
    assert "model_name" in config["embedding"]
    
    print("   ✅ Configuration structure is correct")
except Exception as e:
    errors.append(f"Configuration check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试3: 检查EmbeddingManager初始化参数
print("\n3. Testing EmbeddingManager initialization...")
try:
    import inspect
    sig = inspect.signature(EmbeddingManager.__init__)
    params = list(sig.parameters.keys())
    
    required_params = ['self', 'model_name', 'persist_dir', 'collection_name', 'device']
    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
    
    print(f"   ✅ Parameters: {', '.join(params[1:])}")
except Exception as e:
    errors.append(f"EmbeddingManager check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试4: 检查Retriever初始化参数
print("\n4. Testing Retriever initialization...")
try:
    sig = inspect.signature(Retriever.__init__)
    params = list(sig.parameters.keys())
    
    required_params = ['self', 'embedding_manager', 'top_k']
    for param in required_params:
        assert param in params, f"Missing parameter: {param}"
    
    print(f"   ✅ Parameters: {', '.join(params[1:])}")
except Exception as e:
    errors.append(f"Retriever check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试5: 检查SummarizationEngine初始化参数
print("\n5. Testing SummarizationEngine initialization...")
try:
    sig = inspect.signature(SummarizationEngine.__init__)
    params = list(sig.parameters.keys())
    
    # 检查参数名
    assert 'model_name' in params, "Missing 'model_name' parameter"
    assert 'use_zhipu' in params, "Missing 'use_zhipu' parameter"
    assert 'max_summary_length' not in params, "'max_summary_length' should not be in __init__"
    
    print(f"   ✅ Parameters: {', '.join(params[1:])}")
except Exception as e:
    errors.append(f"SummarizationEngine check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试6: 检查SummarizationEngine.summarize参数
print("\n6. Testing SummarizationEngine.summarize method...")
try:
    sig = inspect.signature(SummarizationEngine.summarize)
    params = list(sig.parameters.keys())
    
    # 检查参数名
    assert 'query' in params, "Missing 'query' parameter"
    assert 'top_k_results' in params, "Missing 'top_k_results' parameter (not 'top_results'!)"
    assert 'max_summary_length' in params, "Missing 'max_summary_length' parameter"
    
    print(f"   ✅ Parameters: {', '.join(params[1:])}")
except Exception as e:
    errors.append(f"SummarizationEngine.summarize check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试7: 检查ExplainabilityEngine.generate_html_report参数
print("\n7. Testing ExplainabilityEngine.generate_html_report method...")
try:
    sig = inspect.signature(ExplainabilityEngine.generate_html_report)
    params = list(sig.parameters.keys())
    
    # 检查参数名
    assert 'query' in params, "Missing 'query' parameter"
    assert 'summary' in params, "Missing 'summary' parameter"
    assert 'reranked_results' in params, "Missing 'reranked_results' parameter"
    
    print(f"   ✅ Parameters: {', '.join(params[1:])}")
except Exception as e:
    errors.append(f"ExplainabilityEngine.generate_html_report check failed: {e}")
    print(f"   ❌ Error: {e}")

# 测试8: 检查环境变量
print("\n8. Testing environment variables...")
import os
if 'ZHIPU_API_KEY' in os.environ:
    print("   ✅ ZHIPU_API_KEY is set")
else:
    print("   ⚠️  ZHIPU_API_KEY is not set (required for AI summarization)")

# 总结
print("\n" + "=" * 60)
if errors:
    print(f"❌ Verification FAILED with {len(errors)} error(s):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    sys.exit(1)
else:
    print("✅ All verifications PASSED!")
    print("=" * 60)
    print("\nThe system is ready to use!")
    print("\nNext steps:")
    print("  1. Set ZHIPU_API_KEY if not already set")
    print("  2. Add documents to data/raw/")
    print("  3. Run: python pks.py init")
    print("  4. Run: python pks.py query \"your question\"")
    print()
