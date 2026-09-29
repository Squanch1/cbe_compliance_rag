# 跨境电商多平台规则与税务合规 RAG 问答系统

Cross-border E-commerce Multi-platform Rules and Tax Compliance RAG Q&A System

面向跨境电商卖家的合规问答系统，覆盖多平台（Amazon、TikTok Shop、Shopee、Temu 等）平台规则与各国税务合规要求，基于检索增强生成（Retrieval-Augmented Generation, RAG）提供带出处引用的问答能力。

**核心主张：所有回答必须附带可追溯的原文引用；无法从语料中找到依据时，明确声明依据不足，而不是给出推测。**

## 当前状态

首期范围锁定 Amazon 平台与欧盟增值税（VAT）/ 进口一站式服务（IOSS）。规范已定稿，进入实现阶段。

- [x] 仓库初始化
- [x] 项目级 CLAUDE.md 与 Spec 开发边界约束规范
- [x] 技术选型与架构设计
- [x] 配置层实现
- [ ] 外部服务连通性验证
- [ ] 知识库数据采集与处理
- [ ] 检索与问答链路实现
- [ ] 评测集与效果评估

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.10 |
| 服务层 | FastAPI |
| 界面 | Streamlit |
| 向量库 | Milvus 2.6（稠密 + 稀疏混合检索） |
| 关系库 | MySQL（文档元数据、分块、维度表） |
| 文档库 | MongoDB（原始解析产物） |
| 缓存 | Redis（会话上下文、查询缓存） |
| 嵌入模型 | bge-m3（进程内加载，fp16，输出稠密 1024 维与稀疏两路） |
| 生成模型 | 阿里云百炼（OpenAI 兼容模式） |

## 目录结构

```
.
├── CLAUDE.md                 # 开发约定
├── .env.example              # 配置模板，只含占位符
├── pyproject.toml            # 项目元数据与 pytest 配置
├── docs/
│   ├── spec/                 # Spec 开发边界约束规范
│   └── adr/                  # 架构决策记录
├── src/cbe_rag/
│   ├── config/               # 配置加载与校验
│   ├── storage/              # Milvus / MySQL / MongoDB / Redis 适配层
│   ├── ingestion/            # 采集与解析（fetcher / parser / chunker）
│   ├── indexing/             # 向量化与写入索引
│   ├── retrieval/            # 召回、过滤与融合
│   ├── generation/           # 提示词组装与答案生成
│   ├── api/                  # FastAPI 路由与数据模型
│   └── observability/        # 日志与追踪
├── apps/web/                 # Streamlit 前端
├── scripts/                  # 运维脚本
├── tests/                    # unit / integration / fixtures
├── data/                     # 原始与处理数据，不入库
└── models/                   # 本地模型文件，不入库
```

## 快速开始

```bash
# 1. 复制配置模板并填入真实凭据
cp .env.example .env

# 2. 运行单元测试（不依赖外部服务）
python -m pytest
```

外部服务（Milvus、Redis、MongoDB、MySQL）与本地模型文件的准备方式，见 `docs/spec/03-data-model.md` 第 7 节。

## 文档

| 文档 | 内容 |
|---|---|
| `docs/spec/00-overview.md` | 项目目标与用户故事 |
| `docs/spec/01-scope.md` | 范围边界：做什么、不做什么 |
| `docs/spec/02-architecture.md` | 分层架构与模块职责 |
| `docs/spec/03-data-model.md` | 数据模型与存储分工 |
| `docs/parsing-decisions.md` | 解析决策记录：阈值来源、候选方案对比与被否决的选项 |

## 许可证

待定。
