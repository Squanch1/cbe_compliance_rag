# 03 数据模型与存储分工

## 0. 总则

- 存储职责划分依据见 `02-architecture.md` 第 5 节，本文档只落实具体结构
- 所有标识符（`doc_id`、`chunk_id`、`session_id`）在四个存储间**保持同一套取值**，不做本地化转换
- 字符集统一 `utf8mb4`，排序规则 `utf8mb4_0900_ai_ci`
- 时间字段统一 `DATETIME`，存本地时间；日期字段用 `DATE`
- 所有表必须有 `created_at`，可变的表必须有 `updated_at`

## 1. 标识符约定

| 标识符 | 格式 | 生成方 | 说明 |
|---|---|---|---|
| `doc_id` | UUID v4，36 字符 | 入库时生成 | 文档级唯一标识 |
| `parent_id` | `{doc_id}_p{序号:04d}` | 切分程序 | 父块标识 |
| `chunk_id` | `{doc_id}_c{序号:04d}` | 切分程序 | 子块标识，Milvus 主键 |
| `session_id` | UUID v4 | 接口层 | 会话标识 |

**序号从 0 开始，左侧补零到 4 位。** 补零是为了让字符串排序与数值排序一致，便于调试时按序查看。同一文档内父块与子块各自独立编号。

## 2. Milvus

### 2.1 命名空间

| 项 | 值 |
|---|---|
| Database | `cbe_compliance` |
| Collection | `cbe_chunks_v1` |

**不得写入 `default` 库。** Collection 名带版本后缀，schema 变更时建 `cbe_chunks_v2` 而非原地改，旧集合保留到新集合验证通过再删。

### 2.2 字段定义

| 字段 | Milvus 类型 | 约束 | 说明 |
|---|---|---|---|
| `chunk_id` | `VARCHAR(64)` | 主键 | 子块标识 |
| `dense_vector` | `FLOAT_VECTOR` | dim = 1024 | bge-m3 稠密输出 |
| `sparse_vector` | `SPARSE_FLOAT_VECTOR` | 不指定维度 | bge-m3 稀疏输出，维度由词元索引隐式决定 |
| `doc_id` | `VARCHAR(36)` | 非空 | 关联 MySQL |
| `parent_id` | `VARCHAR(64)` | 非空 | 召回后折叠用（见 `02-architecture.md` 5.2） |
| `chunk_index` | `INT32` | 非空 | 子块在文档内的序号 |
| `country` | `VARCHAR(16)` | 非空 | 过滤维度 |
| `doc_type` | `VARCHAR(32)` | 非空 | 过滤维度 |
| `publisher` | `VARCHAR(64)` | 非空 | 过滤维度 |

**只有子块进入本集合**，父块不建立向量索引。

**字段长度留有余量**：`VARCHAR` 的 max_length 必须显式指定且不可改，宁大勿小。超长写入会直接报错，不会截断。

### 2.3 索引

**稠密向量索引**

| 参数 | 值 |
|---|---|
| 索引类型 | `FLAT` |
| 度量方式 | `COSINE` |

**稀疏向量索引**

| 参数 | 值 |
|---|---|
| 索引类型 | `SPARSE_INVERTED_INDEX` |
| 度量方式 | `IP` |

稀疏向量只支持内积（`IP`）作为度量方式，其他度量对该类型无定义。

选 `FLAT` 而非 `HNSW` 的理由：首期语料 30 至 50 篇，子块总量预计在数千量级，暴力检索耗时在毫秒级，而 `FLAT` 给出**精确最近邻**。用精确检索可以避免评测结果被近似算法的召回损失污染——如果检索本身漏了，就无法判断是切分问题还是模型问题。

**升级触发条件**：子块总量超过 10 万，或检索 P99 延迟超过 100 毫秒时，稠密索引改用 `HNSW`（`M=16`、`efConstruction=200`，查询时 `ef=64`）。稀疏索引在同等规模下改用 `SPARSE_WAND`。

**升级不是原地修改**：变更索引类型需要 `release_collection` → `drop_index` → `create_index` → `load_collection` 四步，向量数据本身不必重灌，但索引要重建。

**标量索引**：`country`、`doc_type`、`publisher` 三个过滤字段各建 `INVERTED` 索引。

**稠密度量选 `COSINE` 而非 `IP`**：bge-m3 的稠密输出已 L2 归一化，两者数学等价；用 `COSINE` 是为了让意图显式——不依赖「上游一定归一化过」这个隐含前提。稀疏向量则相反，只能也必须用 `IP`。

### 2.4 双路融合

两路召回结果由 Milvus 的 `hybrid_search` 融合，采用加权求和（`WeightedRanker`）。选它的理由见 `02-architecture.md` 6.6。

**权重走配置项**，分稠密权重与稀疏权重两个参数。首期取值在评测集上标定后写入，不凭直觉设定。建议稠密权重不低于稀疏权重，因为跨语言的语义匹配完全由稠密路承担。

**每路的候选数量单独配置**，不共用同一个 `limit`。稀疏召回通常需要更大的候选集，因为其分数分布比稠密更陡。

**标量过滤条件必须附加到两路请求上**，漏掉任一路都会让筛选结果出现不该出现的文档。

### 2.5 拒答阈值

检索结果低于配置阈值时触发拒答路径（见 `02-architecture.md` 6.3）。

**可选判据有两个，需在评测集上比较后确定。**

| 判据 | 优点 | 缺点 |
|---|---|---|
| 稠密路最高余弦相似度 | 有绝对含义（0 到 1），与语料规模无关，纯粹反映语义相关性 | 需额外执行一次稠密检索。`FLAT` 索引下耗时毫秒级，代价可接受 |
| 融合分数 | 无需额外查询，`hybrid_search` 已返回 | 混入了词元匹配的贡献。文档仅因共享缩写或数字而稀疏得分偏高时，融合分数会被抬高，可能掩盖语义上并不相关的事实 |

**倾向稠密余弦**。拒答要回答的问题是「语料里到底有没有讲这件事」，这本质是语义判断。稀疏路的词元重合是相关性的辅助证据，不宜作为「有没有讲」的判据。

阈值走配置项，首期取值在评测集上标定后写入，不凭直觉设定。

## 3. MySQL

### 3.1 命名空间

Database：`cbe_compliance`

**不得写入其他项目的库。**

### 3.2 `documents` 文档元数据

```sql
CREATE TABLE documents (
    doc_id         CHAR(36)      NOT NULL COMMENT '文档唯一标识，UUID',
    title          VARCHAR(512)  NOT NULL COMMENT '文档标题',
    source_url     VARCHAR(1024) NULL     COMMENT '原文出处，缺失则无法进入向量库',
    publisher      VARCHAR(64)   NOT NULL COMMENT '发布机构代码，关联 dim_publisher',
    platform       VARCHAR(32)   NOT NULL COMMENT '平台代码，首期固定 amazon',
    country        VARCHAR(16)   NOT NULL COMMENT '国家代码，EU 表示欧盟整体',
    doc_type       VARCHAR(32)   NOT NULL COMMENT '文档类型代码，关联 dim_doc_type',
    effective_date DATE          NULL     COMMENT '生效日期，允许为空',
    collected_date DATE          NOT NULL COMMENT '采集日期',
    status         VARCHAR(16)   NOT NULL COMMENT 'pending/indexed/needs_manual/failed/superseded，仅 indexed 参与检索',
    parse_attempts JSON          NULL     COMMENT '各解析层的尝试记录，全部失败时供人工排查',
    missing_fields JSON          NULL     COMMENT '缺失的必填字段名清单',
    raw_path       VARCHAR(1024) NOT NULL COMMENT '原始文件绝对路径，见 CLAUDE.md 5.2 路径约定',
    content_hash   CHAR(64)      NOT NULL COMMENT '原始文件 SHA-256，用于去重与变更检测',
    created_at     DATETIME      NOT NULL,
    updated_at     DATETIME      NOT NULL,
    PRIMARY KEY (doc_id),
    UNIQUE KEY uk_documents_content_hash (content_hash),
    KEY idx_documents_status (status),
    KEY idx_documents_filter (country, doc_type, publisher)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档元数据';
```

**`status` 状态机**

```
登记（元数据齐备）──> 解析与索引 ──┬─ 成功 ──────────> indexed ──出现新版本──> superseded
                                   ├─ 各层全不合格 ──> needs_manual
                                   └─ 过程出错 ──────> failed

登记（元数据不齐）──> pending ──补齐必填元数据后重跑导入──> 回到「解析与索引」
```

**只有 `indexed` 参与检索**，其余四个状态都不会被召回。检索层的过滤条件写成 `status == 'indexed'`，而不是逐个排除其余状态——以后新增状态时检索代码不必跟着改，也就不会漏。

必填元数据指 `source_url`、`country`、`doc_type`、`publisher`。缺任一项时 `status = pending`，`missing_fields` 记录缺了哪几个。门禁细节见 `02-architecture.md` 6.2.1。

**`needs_manual` 与 `failed` 是两回事**：

- `needs_manual` 表示**所有解析层都不合格**，机器处理不了，需要人工介入。此时 `parse_attempts` 记录了每一层「用哪个工具、为什么没过」
- `failed` 表示解析通过但**后续步骤出错**（如写向量库失败），通常是环境或代码问题，重跑可能就好

分开的理由是处理方式完全不同：前者要人去读文档，后者重跑即可。合并成一个状态会让人无从判断该做什么。

**`parse_attempts` 用 JSON 而非文本**：人工排查时需要按层遍历、按字段比较，JSON 可以直接读出结构化数据，一段拼接好的日志只能靠肉眼找。

**`missing_fields` 用 JSON 而非逗号分隔字符串**：需要按字段名查询「哪些文档缺 source_url」，JSON 类型可以建函数索引，字符串做不到。

**`content_hash` 加唯一约束**：同一份文件重复导入时不会产生第二条记录。

判重分两步：**先按 hash 查库，命中就跳过**；唯一约束是并发场景下的兜底，不是常规判重手段。顺序不能颠倒——`doc_id` 是每次运行新生成的 UUID，若直接插入靠约束拦截，同一份文档会白白消耗一批 UUID；更危险的是，一旦防护有漏，新 `doc_id` 会让 `chunk_id` 另起一套前缀，向量库里同一份文档出现两组向量。**跳过时必须复用库里原有的 `doc_id`。**

**hash 算的是文件字节，不是解析后的正文。** 这样判重发生在解析之前，重复文件省下一次完整解析。副作用是同一份内容的 PDF 版与 HTML 版会被当作两份独立文档——这是对的，它们的来源与解析结构本就不同，引用时需区分。

文档内容更新时 hash 变化，走「新记录入库 + 旧记录标 `superseded`」的流程。旧记录保留，便于追溯这份文档换过几次、何时换的；但不再参与检索，避免过时内容被引用。

### 3.3 `chunks` 分块表

父子两级块存于同一张表，子块通过 `parent_id` 自引用父块。

```sql
CREATE TABLE chunks (
    chunk_id     CHAR(64)   NOT NULL COMMENT '父块 {doc_id}_pXXXX 或子块 {doc_id}_cXXXX',
    doc_id       CHAR(36)   NOT NULL COMMENT '所属文档',
    parent_id    CHAR(64)   NULL     COMMENT '父块标识，父块自身为空',
    level        VARCHAR(8) NOT NULL COMMENT 'parent 或 child',
    chunk_index  INT        NOT NULL COMMENT '同级内的序号，从 0 开始',
    text         MEDIUMTEXT NOT NULL COMMENT '块的正文',
    token_count  INT        NOT NULL COMMENT 'token 数，便于排查切分异常',
    start_offset INT        NULL     COMMENT '在父块正文中的起始字符位置，子块必填',
    end_offset   INT        NULL     COMMENT '在父块正文中的结束字符位置，左闭右开，子块必填',
    created_at   DATETIME   NOT NULL,
    PRIMARY KEY (chunk_id),
    KEY idx_chunks_doc (doc_id, level, chunk_index),
    KEY idx_chunks_parent (parent_id),
    CONSTRAINT fk_chunks_doc FOREIGN KEY (doc_id)
        REFERENCES documents (doc_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='父子两级分块';
```

**子块记录相对父块的字符偏移**，用于界面在父块全文中高亮命中的子块。

不在渲染时用字符串查找代替偏移量：同一段文字在一个父块内可能重复出现（法条与政策文本尤其常见），查找会定位到错误的那一处。

**入库时校验 `parent.text[start_offset:end_offset] == child.text`**，不满足即拒绝。这条保证「子块是父块的连续子串」（见 `02-architecture.md` 6.5），是「父块进提示词不丢信息」与「界面高亮定位」两个能力的前提。

**取父块正文的查询**（在线链路第 5 步）：

```sql
SELECT c.chunk_id, c.text AS child_text, p.text AS parent_text
FROM chunks c
LEFT JOIN chunks p ON c.parent_id = p.chunk_id
WHERE c.chunk_id IN (...)
```

**外键用 `ON DELETE CASCADE`**：删除文档时其所有块自动清理，避免孤儿数据。

**`level` 字段是冗余的**（可从 `parent_id` 是否为空推断），但保留它可以让按层级查询用上索引，不必写函数判断。

### 3.4 维度表

三张枚举表，供界面渲染筛选下拉框，同时约束元数据取值。

```sql
CREATE TABLE dim_country (
    code      VARCHAR(16) NOT NULL COMMENT 'ISO 3166-1 alpha-2，或 EU',
    name_zh   VARCHAR(64) NOT NULL,
    name_en   VARCHAR(64) NOT NULL,
    is_active TINYINT(1)  NOT NULL DEFAULT 1 COMMENT '是否在首期范围内',
    PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='国家维度';

CREATE TABLE dim_doc_type (
    code      VARCHAR(32) NOT NULL COMMENT 'guideline/regulation/policy',
    name_zh   VARCHAR(64) NOT NULL,
    name_en   VARCHAR(64) NOT NULL,
    is_active TINYINT(1)  NOT NULL DEFAULT 1,
    PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档类型维度';

CREATE TABLE dim_publisher (
    code         VARCHAR(64)  NOT NULL COMMENT 'amazon/eu_commission/...',
    name_zh      VARCHAR(64)  NOT NULL,
    name_en      VARCHAR(64)  NOT NULL,
    official_url VARCHAR(512) NULL COMMENT '官方站点，用于校验来源权威性',
    is_active    TINYINT(1)   NOT NULL DEFAULT 1,
    PRIMARY KEY (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='发布机构维度';
```

`is_active` 的用途：首期只启用 Amazon 与欧盟，其余取值预先登记但标为不启用。这样扩充范围时只改数据不改表结构，与 `01-scope.md`「架构预留扩展位」的要求一致。

**`official_url` 不只是展示用**：采集语料时可据此校验来源是否权威，避免把货代博客当官方文档录进来。

### 3.5 `eval_cases` 评测集

```sql
CREATE TABLE eval_cases (
    case_id           CHAR(36)   NOT NULL COMMENT '用例唯一标识',
    question          TEXT       NOT NULL COMMENT '测试问题',
    expected_behavior VARCHAR(16) NOT NULL COMMENT 'answer 或 refuse',
    category          VARCHAR(32) NOT NULL COMMENT '分类，见下',
    expected_doc_ids  JSON       NULL COMMENT '期望命中的文档标识列表',
    notes             TEXT       NULL COMMENT '出题依据与备注',
    created_at        DATETIME   NOT NULL,
    PRIMARY KEY (case_id),
    KEY idx_eval_cases_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评测用例';
```

**`category` 取值与 `01-scope.md` 第 3 节的灰色地带一一对应**：

| 取值 | 对应场景 | 期望行为 |
|---|---|---|
| `factual` | 有明确原文支撑的事实型提问 | `answer` |
| `comparison` | 概念比较类（见 3.1 节） | `answer`，且需逐条挂引用 |
| `calculation` | 计算类（见 3.2 节） | `refuse` |
| `out_of_scope` | 超出语料范围（见 3.3 节） | `refuse` |
| `conflict` | 多文档规定冲突（见 3.4 节） | `answer`，且需标注冲突 |
| `advice` | 结论性提问（见 3.5 节） | `answer`，且不得给指令性表述 |

这样设计的目的是**让拒答能力可被度量**。只统计「答对了多少」会鼓励系统勉强作答，把「该拒答时拒答」纳入评测才能约束住。

**评测结果表暂不建。** 首期评测以脚本输出报告的形式进行，待指标稳定后再考虑落库。

## 4. Redis

### 4.1 命名规范

```
cbe:{域}:{实体}:{标识}[:{子项}]
```

**统一 `cbe:` 前缀是强制要求。** Redis 实例由多个项目共用，无前缀会导致键冲突和误删。

### 4.2 键定义

| 键 | 类型 | 内容 | TTL |
|---|---|---|---|
| `cbe:session:{session_id}:msgs` | LIST | 会话消息，按时间顺序，元素为 JSON 字符串 | 1 小时，每次追加后刷新 |
| `cbe:session:{session_id}:meta` | HASH | 会话元信息：创建时间、最后活跃时间、轮次数 | 同上 |
| `cbe:cache:answer:{sha1}` | STRING | 问答结果 JSON。`sha1` 为「规范化问题 + 筛选条件」的摘要 | 由配置项决定，首期 10 分钟 |
| `cbe:lock:index:{doc_id}` | STRING | 索引构建互斥锁，值为进程标识 | 5 分钟，防止索引进程崩溃后死锁 |

**会话消息设 TTL 而非永久保留**：会话上下文只服务于进行中的对话，长期存档应落 MySQL（本期不做）。

**缓存键必须包含筛选条件**：同一个问题在不同筛选条件下的答案不同，只按问题文本做键会返回错误结果。`sha1` 的输入格式为 `问题文本 + "\x1f" + 排序后的筛选条件`，用不可见分隔符避免文本拼接歧义。

**锁的 TTL 是兜底，不是正常释放路径**：正常流程结束后主动删除键；TTL 只用于进程异常退出时自动解锁。

### 4.3 序列化

会话消息与缓存值统一用 JSON 字符串，**不用 pickle 或任何可执行序列化格式**。Redis 中的数据可能跨进程、跨语言读取，可执行格式是安全隐患。

## 5. MongoDB（M2 阶段）

首期不接入。M2 阶段用于存放原始解析产物，服务于解析质量回溯与调试。

```
Database   cbe_compliance
Collection raw_parsed

文档结构
{
  doc_id:         string,      // 关联 MySQL documents
  parser_version: string,      // 解析器版本，便于定位解析质量问题
  source_format:  string,      // html 或 pdf
  parsed_at:      datetime,
  blocks: [                    // 解析出的结构化块，保留原始层级
    { type: string, level: number, text: string, order: number }
  ]
}
```

`parser_version` 是必要的：解析器升级后重跑同一份文档，结果可能不同。留版本号才能在发现检索质量下降时判断是不是解析变更引起的。

## 6. 跨存储一致性

| 关系 | 依据 | 不一致时怎么办 |
|---|---|---|
| Milvus ↔ MySQL | MySQL 为准，Milvus 存派生数据 | 以 `doc_id` 重建该文档的向量 |
| Redis ↔ MySQL | Redis 只是缓存与临时会话 | 直接清空重建 |
| MongoDB ↔ MySQL | MongoDB 存原始产物，不参与检索 | 无一致性要求 |

**不做分布式事务。** 重建成本远低于引入两阶段提交的复杂度，见 `02-architecture.md` 第 9 节。

## 7. 实现前置条件

以下事项未完成前，本文档中的数据模型无法落地验证：

| 事项 | 说明 |
|---|---|
| 升级 `pymilvus` 到 2.6.x | 当前 2.5.4，混合检索所需的 `hybrid_search`、`AnnSearchRequest`、`RRFRanker` 支持不完整 |
| 放入 bge-m3 模型文件 | `models/bge-m3/`，必须为 HuggingFace 格式。Ollama 的 GGUF 格式不可用 |
| 在 Milvus 建独立 database | `cbe_compliance`，不得写入 `default`（其中已有其他项目的集合） |
| 在 MySQL 建独立 database | `cbe_compliance` |
| 标定拒答阈值 | 需先有评测集与已索引的语料，取值来自实测而非估计 |
