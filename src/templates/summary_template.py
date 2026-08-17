"""
高级模板摘要生成器
"""


def generate_template_summary(query: str, documents: list, max_length: int = 500) -> dict:
    """
    生成高质量的模板摘要

    Args:
        query: 查询问题
        documents: 文档列表，每个包含text和metadata
        max_length: 最大长度

    Returns:
        摘要字典
    """
    if not documents:
        return get_empty_summary(query)

    # 1. 准备数据
    texts = [doc.get('text', '') for doc in documents]
    metadatas = [doc.get('metadata', {}) for doc in documents]

    # 2. 提取核心信息
    keywords = extract_keywords(texts, metadatas)
    key_points = extract_key_points(texts, metadatas)
    sources = extract_sources(metadatas)

    # 3. 生成结构化摘要
    summary = build_structured_summary(query, keywords, key_points, sources, max_length)

    return {
        "summary": summary,
        "keywords": keywords[:8],
        "key_points": key_points[:6],
        "evidence": build_evidence(texts, metadatas),
        "explanation": f"基于{len(documents)}篇相关文献的综合分析"
    }


def extract_keywords(texts, metadatas):
    """提取关键词"""
    import jieba
    from collections import Counter

    all_keywords = []

    for metadata in metadatas:
        # 使用预提取的关键词
        keywords = metadata.get('keywords', [])
        all_keywords.extend(keywords)

    # 如果没有预提取的关键词，从文本中提取
    if not all_keywords:
        for text in texts:
            words = jieba.cut(text)
            # 过滤停用词
            stopwords = {'的', '了', '是', '在', '和', '有', '一', '这', '不', '人'}
            keywords = [w for w in words if w not in stopwords and len(w) > 1]
            all_keywords.extend(keywords[:10])

    # 统计频率
    counter = Counter(all_keywords)
    return [kw for kw, _ in counter.most_common(10)]


def extract_key_points(texts, metadatas, max_points=6):
    """提取关键点"""
    import re

    key_points = []

    for i, text in enumerate(texts[:4]):  # 只处理前4个文档
        # 分句
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if not sentences:
            continue

        # 策略1：取开头句（通常是主题句）
        if len(sentences[0]) > 20:
            point = sentences[0]
            if len(point) > 60:
                point = point[:57] + "..."
            key_points.append(f"{i + 1}. {point}")

        # 策略2：取包含关键词的句子
        title = metadatas[i].get('title', '')
        if title and len(sentences) > 1:
            for sentence in sentences[1:4]:  # 检查前几个句子
                if any(kw in sentence for kw in title.split('_')):
                    if len(sentence) > 20:
                        point = sentence[:80] + "..." if len(sentence) > 80 else sentence
                        key_points.append(f"{i + 1}. {point}")
                        break

        # 策略3：如果还没有足够的关键点，取有内容的句子
        if len(key_points) <= i and len(sentences) > 1:
            for sentence in sentences[1:]:
                if len(sentence) > 30 and len(sentence) < 100:
                    key_points.append(f"{i + 1}. {sentence}...")
                    break

    return key_points[:max_points]


def extract_sources(metadatas):
    """提取来源信息"""
    sources = []
    for i, meta in enumerate(metadatas[:4], 1):
        title = meta.get('title', f'文档{i}')
        # 清理标题
        title = title.replace('doc_', '').replace('_', ' ')
        sources.append(title)
    return sources


def build_structured_summary(query, keywords, key_points, sources, max_length):
    """构建结构化摘要"""
    parts = []

    # 标题
    parts.append(f"📋 **关于「{query}」的分析总结**")
    parts.append("=" * 40)

    # 概述
    parts.append(f"\n🔍 **概述**")
    parts.append(f"基于{len(sources)}个相关文档的分析，{query}涉及以下核心内容：")

    # 关键点
    if key_points:
        parts.append(f"\n📌 **核心要点**")
        for point in key_points:
            parts.append(f"  • {point}")

    # 关键技术
    if keywords:
        parts.append(f"\n🔧 **关键技术/概念**")
        parts.append(f"  {', '.join(keywords[:6])}")

    # 主要来源
    if sources:
        parts.append(f"\n📚 **主要参考文档**")
        for i, source in enumerate(sources[:3], 1):
            parts.append(f"  {i}. {source}")

    # 完整摘要
    full_text = "\n".join(parts)

    # 限制长度
    if len(full_text) > max_length:
        full_text = full_text[:max_length - 3] + "..."

    return full_text


def build_evidence(texts, metadatas):
    """构建证据链"""
    evidence = []
    for i, (text, meta) in enumerate(zip(texts[:3], metadatas[:3])):
        evidence.append({
            "source": meta.get('title', f'文献{i + 1}'),
            "para_index": meta.get('paragraph_index', 0),
            "quote": text[:70] + "..." if len(text) > 70 else text
        })
    return evidence


def get_empty_summary(query):
    """空结果处理"""
    return {
        "summary": f"未找到与「{query}」直接相关的文档。\n请尝试不同的查询词或添加更多相关文档到知识库。",
        "keywords": [],
        "key_points": [],
        "evidence": [],
        "explanation": "查询没有返回相关结果"
    }