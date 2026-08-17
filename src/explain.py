import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Explainability module for the Personal Knowledge Summary System.
Generates interpretable explanations and HTML reports.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64

from .logger import get_logger

logger = get_logger()


class ExplainabilityEngine:
    """Generates interpretable explanations for summarization results."""
    
    def __init__(self):
        """Initialize explainability engine."""
        logger.info("Explainability engine initialized")
    
    def generate_feature_explanation(
        self,
        reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate feature-level explanations.
        
        Args:
            reranked_results: List of reranked results with features
            
        Returns:
            Feature explanation dictionary
        """
        explanations = {
            "feature_importance": {},
            "top_features_per_result": []
        }
        
        # Aggregate feature importance using original feature values
        feature_sums = {}
        feature_counts = {}
        
        for result in reranked_results:
            features = result.get("features", {})
            contributions = result.get("contributions", {})
            
            # Use original feature values for importance calculation
            for feature_name, feature_value in features.items():
                if feature_name not in feature_sums:
                    feature_sums[feature_name] = 0
                    feature_counts[feature_name] = 0
                
                feature_sums[feature_name] += feature_value
                feature_counts[feature_name] += 1
        
        # Compute average importance (using original feature values)
        for feature_name in feature_sums:
            avg_value = feature_sums[feature_name] / max(feature_counts[feature_name], 1)
            explanations["feature_importance"][feature_name] = float(avg_value)
        
        # Get top features for each result
        for i, result in enumerate(reranked_results):
            contributions = result.get("contributions", {})
            sorted_features = sorted(
                contributions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            explanations["top_features_per_result"].append({
                "rank": i + 1,
                "top_3_features": [
                    {"feature": name, "contribution": float(value)}
                    for name, value in sorted_features[:3]
                ]
            })
        
        return explanations
    
    def generate_evidence_explanation(
        self,
        summary: Dict[str, Any],
        reranked_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate evidence-level explanations.
        
        Args:
            summary: Summary dictionary
            reranked_results: List of reranked results
            
        Returns:
            Evidence explanation dictionary
        """
        evidence_explanations = {
            "evidence_sources": [],
            "keyword_support": {}
        }
        
        keywords = summary.get("keywords", [])
        evidence = summary.get("evidence", [])
        
        # Map evidence to sources
        for i, result in enumerate(reranked_results):
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            
            evidence_explanations["evidence_sources"].append({
                "rank": i + 1,
                "source": metadata.get("title", f"Source {i+1}"),
                "paragraph_index": metadata.get("paragraph_index", 0),
                "text_preview": text[:150] + "..." if len(text) > 150 else text,
                "rerank_score": result.get("rerank_score", 0)
            })
            
            # Count keyword occurrences
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    if keyword not in evidence_explanations["keyword_support"]:
                        evidence_explanations["keyword_support"][keyword] = []
                    
                    evidence_explanations["keyword_support"][keyword].append({
                        "source": metadata.get("title", f"Source {i+1}"),
                        "rank": i + 1
                    })
        
        return evidence_explanations
    
    def generate_html_report(
        self,
        query: str,
        summary: Dict[str, Any],
        reranked_results: List[Dict[str, Any]],
        feature_explanation: Optional[Dict[str, Any]] = None,
        evidence_explanation: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate HTML report with visualizations.
        
        Args:
            query: Query text
            summary: Summary dictionary
            reranked_results: List of reranked results
            feature_explanation: Feature-level explanations
            evidence_explanation: Evidence-level explanations
            
        Returns:
            HTML content
        """
        # Generate feature importance chart data
        if feature_explanation:
            features = feature_explanation.get("feature_importance", {})
            # Ensure feature names are in a consistent order for display
            feature_names = [
                "semantic_sim", 
                "keyword_overlap", 
                "title_overlap", 
                "position_score", 
                "length_score"
            ]
            feature_values = [features.get(name, 0.0) for name in feature_names]
            
            # Map feature names to Chinese for display
            feature_name_map = {
                "semantic_sim": "语义相似度",
                "keyword_overlap": "关键词重合度",
                "title_overlap": "标题重合度",
                "position_score": "位置分数",
                "length_score": "长度分数"
            }
            feature_names = [feature_name_map.get(name, name) for name in feature_names]
            feature_values = [features.get(name, 0.0) for name in [
                "semantic_sim", 
                "keyword_overlap", 
                "title_overlap", 
                "position_score", 
                "length_score"
            ]]
        else:
            feature_names = []
            feature_values = []
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Knowledge Summary Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 16px;
            opacity: 0.9;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .summary-box {{
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin-bottom: 15px;
            border-radius: 4px;
        }}
        
        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        
        .keyword {{
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
        }}
        
        .result-item {{
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            border-left: 4px solid #764ba2;
        }}
        
        .result-item .title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }}
        
        .result-item .score {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }}
        
        .result-item .text {{
            color: #666;
            font-size: 14px;
            margin: 10px 0;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        .feature-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        .feature-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        .feature-table tr:hover {{
            background: #f5f5f5;
        }}
        
        .footer {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 个人知识摘要系统报告</h1>
            <p>查询: <strong>{query}</strong></p>
        </div>
        
        <div class="section">
            <h2>📝 摘要</h2>
            <div class="summary-box">
                <p>{summary.get('summary', '暂无摘要')}</p>
            </div>
            
            <h3 style="margin-top: 15px; color: #333;">关键词</h3>
            <div class="keywords">
"""
        
        for keyword in summary.get("keywords", []):
            html += f'                <span class="keyword">{keyword}</span>\n'
        
        html += """            </div>
        </div>
        
        <div class="section">
            <h2>🎯 主要观点</h2>
"""
        
        for point in summary.get("key_points", []):
            html += f'            <p style="margin-bottom: 10px;">• {point}</p>\n'
        
        html += """        </div>
        
        <div class="section">
            <h2>📊 特征重要性</h2>
"""
        
        if feature_names:
            html += f"""            <div class="chart-container">
                <canvas id="featureChart"></canvas>
            </div>
            <script>
                const ctx = document.getElementById('featureChart').getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(feature_names)},
                        datasets: [{{
                            label: '特征平均值',
                            data: {json.dumps(feature_values)},
                            backgroundColor: 'rgba(102, 126, 234, 0.6)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            </script>
"""
        
        html += """        </div>
        
        <div class="section">
            <h2>📚 证据来源</h2>
"""
        
        for i, result in enumerate(reranked_results):
            metadata = result.get("metadata", {})
            score = result.get("rerank_score", 0)
            text = result.get("text", "")
            
            html += f"""            <div class="result-item">
                <div class="title">
                    [{i+1}] {metadata.get('title', f'Source {i+1}')}
                    <span class="score">Score: {score:.4f}</span>
                </div>
                <div class="text">{text[:200]}...</div>
            </div>
"""
        
        html += """        </div>
        
        <div class="section">
            <h2>🔍 可解释性分析</h2>
            <p>本报告通过以下方式提供可解释性：</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>特征重要性分析：展示各个特征的平均值，反映其在排序中的实际表现</li>
                <li>证据来源追踪：清晰标注每个摘要内容的来源和排序分数</li>
                <li>关键词支持：突出显示支持摘要的关键词及其出现位置</li>
                <li>排序过程透明化：展示混合重排序的各个特征值</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>© 2024 Personal Knowledge Summary System | Generated by Manus AI</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def save_report(
        self,
        html_content: str,
        query: str,
        output_dir: str = "./output/reports/"
    ) -> str:
        """
        Save HTML report to file.
        
        Args:
            html_content: HTML content
            query: Query text
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create filename from query
        query_id = query[:50].replace(" ", "_").replace("/", "_")
        output_file = os.path.join(output_dir, f"{query_id}.html")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")
        return output_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate explainability reports"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to summary JSON file"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to reranked results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/reports/",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    # Load files
    with open(args.summary, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    with open(args.results, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    summary = summary_data.get("summary", {})
    results = results_data.get("reranked_results", [])
    
    # Initialize explainability engine
    engine = ExplainabilityEngine()
    
    # Generate explanations
    feature_explanation = engine.generate_feature_explanation(results)
    evidence_explanation = engine.generate_evidence_explanation(summary, results)
    
    # Generate HTML report
    html_content = engine.generate_html_report(
        args.query,
        summary,
        results,
        feature_explanation,
        evidence_explanation
    )
    
    # Save report
    output_file = engine.save_report(html_content, args.query, args.output_dir)
    
    print(f"\nHTML report saved to: {output_file}")


if __name__ == "__main__":
    main()
