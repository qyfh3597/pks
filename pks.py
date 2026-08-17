#!/usr/bin/env python3
"""
Personal Knowledge Summary System - 统一CLI工具
提供一键式的知识库管理和查询功能
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import webbrowser
from pathlib import Path

# 导入自动管理器
from auto_manager import KnowledgeBaseManager

# 导入核心模块
from src.embed import EmbeddingManager
from src.retrieval import Retriever
from src.rerank import HybridReranker
from src.summarize1 import SummarizationEngine
from src.explain import ExplainabilityEngine
from src.config import get_config
from src.logger import get_logger

logger = get_logger()


class PKSCli:
    """PKS命令行工具"""
    
    def __init__(self):
        """初始化CLI"""
        # 使用get_config获取Config对象
        self.config_obj = get_config()
        # 获取配置字典
        self.config = self.config_obj.config
        self.kb_manager = KnowledgeBaseManager()
    
    def init(self):
        """初始化知识库"""
        print("\n" + "=" * 60)
        print("🚀 Initializing Personal Knowledge Summary System")
        print("=" * 60)
        print()
        
        # 检查数据目录
        raw_dir = Path(self.config["data"]["raw_data_dir"])
        if not raw_dir.exists() or not list(raw_dir.glob("*")):
            print("⚠️  Warning: No files found in data/raw/")
            print("   Please add your documents to data/raw/ first")
            return
        
        print(f"📁 Found {len(list(raw_dir.glob('*')))} files in data/raw/")
        print()
        
        # 同步知识库
        print("⏳ Processing documents...")
        self.kb_manager.sync()
        
        print()
        print("=" * 60)
        print("✅ Initialization complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run: python pks.py query \"your question\"")
        print("  2. Or: python pks.py interactive")
        print()
    
    def query(self, query_text: str, top_k: int = 5, no_open: bool = False):
        """执行查询"""
        print("\n" + "=" * 60)
        print(f"🔍 Query: {query_text}")
        print("=" * 60)
        print()
        
        # 加载模型 - 使用正确的配置键名和参数
        print("⏳ Loading models...")
        
        # EmbeddingManager初始化
        embedding_manager = EmbeddingManager(
            model_name=self.config["embedding"]["model_name"],
            persist_dir=self.config["chroma"]["persist_dir"],
            collection_name=self.config["chroma"]["collection_name"],
            device=self.config["embedding"].get("device", "cpu")
        )
        
        # Retriever初始化
        retriever = Retriever(
            embedding_manager,
            top_k=self.config["retrieval"]["top_candidates_after_retrieval"]
        )
        
        # HybridReranker初始化
        reranker = HybridReranker(embedding_manager)
        
        # SummarizationEngine初始化 - 使用正确的参数名
        summarizer = SummarizationEngine(
            model_name=self.config["summarization"].get("zhipu_model", "glm-4-flash"),
            use_zhipu=self.config["summarization"]["use_zhipu_ai"]
        )
        
        # ExplainabilityEngine初始化
        explainer = ExplainabilityEngine()
        
        # 检索
        print("🔎 Retrieving relevant documents...")
        candidates = retriever.retrieve(query_text)
        print(f"   Found {len(candidates)} candidates")
        
        if not candidates:
            print("\n⚠️  No relevant documents found. Please check:")
            print("   1. Have you run 'python pks.py init' first?")
            print("   2. Are there documents in data/raw/?")
            print()
            return
        
        # 重排序
        print("📊 Reranking results...")
        reranked = reranker.rerank(
            query=query_text,
            candidates=candidates,
            top_k=top_k
        )
        print(f"   Top {len(reranked)} results selected")
        
        # 生成摘要 - 使用正确的参数名 top_k_results
        print("✍️  Generating summary...")
        summary_result = summarizer.summarize(
            query=query_text,
            top_k_results=reranked,  # ⚠️ 正确的参数名
            max_summary_length=self.config["summarization"]["max_summary_length"]
        )
        
        # 保存摘要到文件
        summary_file = summarizer.save_summary(
            summary=summary_result,
            query=query_text
        )
        print("\n" + "=" * 60)
        print("📊 详细特征分析")
        print("=" * 60)
        # 获取使用的权重
        weights = reranker.weights
        print(f"使用的权重配置: {weights}")

        for i, result in enumerate(reranked[:3], 1):
            features = result.get("features", {})
            contributions = result.get("contributions", {})
            total_score = sum(contributions.values())

            print(f"\n🔍 第 {i} 名: {result['metadata'].get('title', '未知')}")
            print(f"   总重排序分数: {result['rerank_score']:.4f}")
            print(f"   原始相似度: {result.get('similarity_score', 0):.4f}")

            if total_score > 0:
                print("   ┌────────────────┬─────────┬─────────┬─────────┬─────────┐")
                print("   │ 特征           │ 原始值  │ 权重    │ 贡献值  │ 占比    │")
                print("   ├────────────────┼─────────┼─────────┼─────────┼─────────┤")

                feature_order = ["semantic_sim", "keyword_overlap", "title_overlap", "position_score", "length_score"]
                feature_names = {
                    "semantic_sim": "语义相似度",
                    "keyword_overlap": "关键词重合度",
                    "title_overlap": "标题重合度",
                    "position_score": "位置分数",
                    "length_score": "长度分数"
                }

                for feat_name in feature_order:
                    feat_value = features.get(feat_name, 0)
                    weight = weights.get(feat_name, 0)
                    contribution = contributions.get(feat_name, 0)
                    percentage = (contribution / total_score * 100) if total_score > 0 else 0
                    display_name = feature_names.get(feat_name, feat_name)

                    print(
                        f"   │ {display_name:<14} │ {feat_value:>7.3f} │ {weight:>7.3f} │ {contribution:>7.3f} │ {percentage:>6.1f}% │")

                print("   └────────────────┴─────────┴─────────┴─────────┴─────────┘")
            else:
                print("   ⚠️ 无特征数据")
        # 生成可解释性报告 - 使用正确的参数名
        print("📈 Generating explainability report...")
        
        # 提取特征数据用于可视化
        feature_importance = {}
        if reranked:
            # 计算每个特征的平均值
            feature_names = ["semantic_sim", "keyword_overlap", "title_overlap", "position_score", "length_score"]
            for feat_name in feature_names:
                values = [r.get("features", {}).get(feat_name, 0.0) for r in reranked]
                feature_importance[feat_name] = sum(values) / len(values) if values else 0.0
        
        feature_explanation = {
            "feature_importance": feature_importance
        }
        
        # 生成HTML内容
        html_content = explainer.generate_html_report(
            query=query_text,
            summary=summary_result,  # ⚠️ 传递整个结果字典
            reranked_results=reranked,  # ⚠️ 正确的参数名
            feature_explanation=feature_explanation  # ⚠️ 传递特征数据
        )
        
        # 保存HTML到文件
        html_path = explainer.save_report(
            html_content=html_content,
            query=query_text
        )
        
        # 显示结果
        print()
        print("=" * 60)
        print("📝 SUMMARY")
        print("=" * 60)
        print()
        print(summary_result["summary"])
        print()
        
        if summary_result.get("keywords"):
            print(f"🔑 Keywords: {', '.join(summary_result['keywords'])}")
            print()
        
        print("=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"✅ Summary saved: {summary_file}")
        print(f"📄 HTML report: {html_path}")
        print()
        
        # 不自动打开浏览器，用户可以手动打开HTML文件
        
        print()
    
    def interactive(self):
        """交互式查询模式"""
        print("\n" + "=" * 60)
        print("💬 Interactive Query Mode")
        print("=" * 60)
        print()
        print("Enter your queries (type 'exit' or 'quit' to stop)")
        print()
        
        while True:
            try:
                query = input("Query> ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                self.query(query, no_open=False)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Query error: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")
    
    def add(self, file_path: str):
        """添加文件到知识库"""
        print(f"\n📥 Adding file: {file_path}")
        self.kb_manager.add_file(file_path)
        print("✅ File added. Run 'python pks.py sync' to update the knowledge base.")
        print()
    
    def remove(self, filename: str):
        """从知识库删除文件"""
        print(f"\n🗑️  Removing file: {filename}")
        self.kb_manager.remove_file(filename)
        print("✅ File removed. Run 'python pks.py sync' to update the knowledge base.")
        print()
    
    def sync(self):
        """同步知识库"""
        print("\n🔄 Syncing knowledge base...")
        self.kb_manager.sync()
        print("✅ Knowledge base synced!")
        print()
    
    def list_files(self):
        """列出所有文件"""
        print("\n📋 Files in knowledge base:")
        print()
        self.kb_manager.list_files()
        print()
    
    def stats(self):
        """显示统计信息"""
        print("\n📊 Knowledge Base Statistics:")
        print()
        self.kb_manager.show_stats()
        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Personal Knowledge Summary System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize knowledge base
  python pks.py init
  
  # Query
  python pks.py query "What is machine learning?"
  
  # Interactive mode
  python pks.py interactive
  
  # Add file
  python pks.py add document.pdf
  
  # Sync knowledge base
  python pks.py sync
  
  # List files
  python pks.py list
  
  # Show statistics
  python pks.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init命令
    subparsers.add_parser('init', help='Initialize knowledge base')
    
    # query命令
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('text', help='Query text')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of top results')
    query_parser.add_argument('--no-open', action='store_true', help='Do not open report in browser')
    
    # interactive命令
    subparsers.add_parser('interactive', help='Interactive query mode')
    
    # add命令
    add_parser = subparsers.add_parser('add', help='Add file to knowledge base')
    add_parser.add_argument('file', help='File path')
    
    # remove命令
    remove_parser = subparsers.add_parser('remove', help='Remove file from knowledge base')
    remove_parser.add_argument('file', help='Filename')
    
    # sync命令
    subparsers.add_parser('sync', help='Sync knowledge base')
    
    # list命令
    subparsers.add_parser('list', help='List all files')
    
    # stats命令
    subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        cli = PKSCli()
        
        if args.command == 'init':
            cli.init()
        elif args.command == 'query':
            cli.query(args.text, args.top_k, args.no_open)
        elif args.command == 'interactive':
            cli.interactive()
        elif args.command == 'add':
            cli.add(args.file)
        elif args.command == 'remove':
            cli.remove(args.file)
        elif args.command == 'sync':
            cli.sync()
        elif args.command == 'list':
            cli.list_files()
        elif args.command == 'stats':
            cli.stats()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Please check the log file for details.")
        return 1


if __name__ == "__main__":
    main()
