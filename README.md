混合重排序 + 可解释生成式个人知识自动摘要系统 | ChromaDB × BGE × Zhipu GLM-4
# Personal Knowledge Summary System (PKS)
> 🧠 基于混合重排序与可解释生成式摘要的个人知识自动问答系统                          
> 特性	说明
混合重排序	5维特征融合（语义相似度 55%、关键词重叠 25%、标题重叠 8%、位置得分 7%、长度得分 5%），支持手动权重和学习模式
增量知识库	文件变化检测 + hash 索引，sync 命令一键更新
双模式摘要	Zhipu GLM-4 API / 本地模板回退，无需 API 也能运行
可解释性报告	HTML 报告展示 reranking 分数分布、特征贡献
# 1. 克隆仓库
git clone https://github.com/qyfh3597/pks.git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 放入你的文档
mkdir -p data/raw
# put your .txt

# 4. 初始化知识库
python pks.py init

# 5. 查询
python pks.py query "什么是机器学习？"

# 6. 交互模式
python pks.py interactive

可选：启用智谱 AI 摘要

set ZHIPU_API_KEY=你的密钥

init       – 初始化知识库（处理文档 → 向量化 → 建索引）

query      – 单次查询

sync       – 增量同步（检测新增/修改/删除文件）

add        – 添加单个文件

remove     – 移除文件

list       – 列出知识库中的文档

stats      – 知识库统计信息

interactive – 交互式问答模式

配置说明

config.json 控制所有参数：

检索：top_k=50（粗召回候选数）

重排序：final_top_k=5，5维权重可调

摘要：max_length=250，use_zhipu_ai=true + glm-4-flash

评估：rouge_types、ndcg_k、map_k

模型：BAAI/bge-base-zh + cpu

技术栈

组件	技术

嵌入模型	BAAI/bge-base-zh (Sentence-Transformers)
向量数据库	ChromaDB (HNSW, Cosine)
重排序	scikit-learn LinearRegression + 5维手工特征
摘要	智谱 GLM-4 Flash API / 本地模板
NLP	jieba + nltk
文档解析	pypdf + python-docx + BeautifulSoup4
