#!/usr/bin/env python3
"""
知识库自动管理器
支持文件变化检测、增量更新和自动同步
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
import json
import hashlib
import argparse
from datetime import datetime
from typing import List, Dict, Any, Set
import time

from src.preprocess import preprocess_directory, process_file
from src.embed import EmbeddingManager
from src.logger import get_logger

logger = get_logger()


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(
            self,
            raw_dir: str = "./data/raw/",
            processed_file: str = "./data/processed/processed.jsonl",
            index_file: str = "./data/processed/file_index.json"
    ):
        """
        初始化管理器

        Args:
            raw_dir: 原始数据目录
            processed_file: 处理后的数据文件
            index_file: 文件索引文件
        """
        self.raw_dir = Path(raw_dir)
        self.processed_file = Path(processed_file)
        self.index_file = Path(index_file)

        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载文件索引
        self.file_index = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """加载文件索引"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """保存文件索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_index, f, indent=2, ensure_ascii=False)
        logger.info(f"File index saved: {len(self.file_index)} files tracked")

    def _calculate_hash(self, file_path: Path) -> str:
        """计算文件MD5哈希"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变化

        Returns:
            包含added, modified, deleted文件列表的字典
        """
        changes = {
            "added": [],
            "modified": [],
            "deleted": []
        }

        # 获取当前文件列表
        current_files = set()
        if self.raw_dir.exists():
            for file_path in self.raw_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(self.raw_dir))
                    current_files.add(rel_path)

        # 检测新增和修改的文件
        for rel_path in current_files:
            file_path = self.raw_dir / rel_path
            current_hash = self._calculate_hash(file_path)

            if rel_path not in self.file_index:
                # 新增文件
                changes["added"].append(rel_path)
                self.file_index[rel_path] = {
                    "hash": current_hash,
                    "last_modified": datetime.now().isoformat(),
                    "status": "new"
                }
            elif self.file_index[rel_path]["hash"] != current_hash:
                # 修改的文件
                changes["modified"].append(rel_path)
                self.file_index[rel_path]["hash"] = current_hash
                self.file_index[rel_path]["last_modified"] = datetime.now().isoformat()
                self.file_index[rel_path]["status"] = "modified"

        # 检测删除的文件
        indexed_files = set(self.file_index.keys())
        for rel_path in indexed_files - current_files:
            changes["deleted"].append(rel_path)
            del self.file_index[rel_path]

        return changes

    def sync(self):
        """同步知识库"""
        logger.info("Starting knowledge base synchronization...")

        # 检测变化
        changes = self.detect_changes()

        total_changes = sum(len(files) for files in changes.values())

        if total_changes == 0:
            logger.info("No changes detected. Knowledge base is up to date.")
            print("✅ No changes detected. Knowledge base is up to date.")
            return

        logger.info(f"Detected changes: {changes}")
        print(f"\n📊 Detected changes:")
        print(f"  Added: {len(changes['added'])} files")
        print(f"  Modified: {len(changes['modified'])} files")
        print(f"  Deleted: {len(changes['deleted'])} files")
        print()

        # 预处理文档
        logger.info("Preprocessing documents...")
        print("⏳ Preprocessing documents...")
        preprocess_directory(
            input_dir=str(self.raw_dir),
            output_file=str(self.processed_file)
        )

        # 保存索引
        self._save_index()

        # 重建向量索引
        logger.info("Rebuilding vector embeddings...")
        print("⏳ Building vector embeddings...")
        try:
            embedding_manager = EmbeddingManager()

            # 清空现有集合
            embedding_manager.clear_collection()

            # 使用load_from_jsonl方法而不是build_index_from_jsonl
            num_docs = embedding_manager.load_from_jsonl(str(self.processed_file))

            logger.info(f"Vector embeddings built: {num_docs} documents")
            print(f"✅ Vector embeddings built: {num_docs} documents")
        except Exception as e:
            logger.error(f"Error rebuilding embeddings: {e}")
            print(f"❌ Error rebuilding embeddings: {e}")
            raise

        logger.info("Knowledge base synchronization complete")
        print("\n✅ Knowledge base synchronized successfully!")

    def add_file(self, file_path: str):
        """
        添加文件到知识库

        Args:
            file_path: 文件路径
        """
        src_path = Path(file_path)
        if not src_path.exists():
            logger.error(f"File not found: {file_path}")
            print(f"❌ File not found: {file_path}")
            return

        # 复制到raw目录
        dst_path = self.raw_dir / src_path.name

        import shutil
        shutil.copy2(src_path, dst_path)

        logger.info(f"File added: {src_path.name}")
        print(f"✅ File added: {src_path.name}")
        print("   Run 'sync' to update the knowledge base")

    def remove_file(self, filename: str):
        """
        从知识库删除文件

        Args:
            filename: 文件名
        """
        file_path = self.raw_dir / filename

        if not file_path.exists():
            logger.error(f"File not found: {filename}")
            print(f"❌ File not found: {filename}")
            return

        file_path.unlink()

        logger.info(f"File removed: {filename}")
        print(f"✅ File removed: {filename}")
        print("   Run 'sync' to update the knowledge base")

    def list_files(self):
        """列出所有文件"""
        if not self.raw_dir.exists():
            print("No files found")
            return

        files = sorted(self.raw_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            print("No files found")
            return

        print(f"Total files: {len(files)}\n")
        for i, file_path in enumerate(files, 1):
            rel_path = file_path.relative_to(self.raw_dir)
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size / 1024:.1f} KB"

            status = "✓"
            if str(rel_path) in self.file_index:
                status_info = self.file_index[str(rel_path)].get("status", "synced")
                if status_info == "new":
                    status = "🆕"
                elif status_info == "modified":
                    status = "📝"

            print(f"  {status} [{i:3d}] {rel_path} ({size_str})")

    def show_stats(self):
        """显示统计信息"""
        # 文件统计
        files = list(self.raw_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        total_size = sum(f.stat().st_size for f in files)

        print(f"Raw files: {len(files)}")
        print(f"Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")

        # 索引统计
        print(f"Indexed files: {len(self.file_index)}")

        # 处理后的文档统计
        if self.processed_file.exists():
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                num_docs = sum(1 for line in f if line.strip())
            print(f"Processed documents: {num_docs}")

        # 向量数据库统计
        try:
            embedding_manager = EmbeddingManager()
            info = embedding_manager.get_collection_info()
            print(f"Vector embeddings: {info['document_count']}")
        except Exception as e:
            print(f"Vector embeddings: N/A (error: {e})")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description="Knowledge Base Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync knowledge base
  python auto_manager.py sync

  # Add file
  python auto_manager.py add document.pdf

  # Remove file
  python auto_manager.py remove document.pdf

  # List files
  python auto_manager.py list

  # Show statistics
  python auto_manager.py stats

  # Detect changes
  python auto_manager.py detect
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # sync命令
    subparsers.add_parser('sync', help='Sync knowledge base')

    # add命令
    add_parser = subparsers.add_parser('add', help='Add file')
    add_parser.add_argument('file', help='File path')

    # remove命令
    remove_parser = subparsers.add_parser('remove', help='Remove file')
    remove_parser.add_argument('file', help='Filename')

    # list命令
    subparsers.add_parser('list', help='List all files')

    # stats命令
    subparsers.add_parser('stats', help='Show statistics')

    # detect命令
    subparsers.add_parser('detect', help='Detect changes')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = KnowledgeBaseManager()

    if args.command == 'sync':
        manager.sync()
    elif args.command == 'add':
        manager.add_file(args.file)
    elif args.command == 'remove':
        manager.remove_file(args.file)
    elif args.command == 'list':
        manager.list_files()
    elif args.command == 'stats':
        manager.show_stats()
    elif args.command == 'detect':
        changes = manager.detect_changes()
        print("\n📊 Detected changes:")
        print(f"  Added: {len(changes['added'])} files")
        if changes['added']:
            for f in changes['added']:
                print(f"    + {f}")
        print(f"  Modified: {len(changes['modified'])} files")
        if changes['modified']:
            for f in changes['modified']:
                print(f"    ~ {f}")
        print(f"  Deleted: {len(changes['deleted'])} files")
        if changes['deleted']:
            for f in changes['deleted']:
                print(f"    - {f}")
        print()


if __name__ == "__main__":
    main()
