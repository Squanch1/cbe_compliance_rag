"""存储结构定义。

MySQL 的建表语句、维度表的初始数据、Milvus 集合的字段与索引参数。
全部依据 docs/spec/03-data-model.md，改动前先改那份规范再同步这里。

本模块只有数据没有逻辑，因此集中放一处；
执行它们的方法在各自的适配器里（MysqlStore / MilvusStore）。
"""

from __future__ import annotations

# ============================================================
# MySQL
# ============================================================

MYSQL_TABLES: tuple[tuple[str, str], ...] = (
    # 表名与建表语句成对出现，便于建表时逐条报进度
    (
        "documents",
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id         CHAR(36)      NOT NULL COMMENT '文档唯一标识，UUID',
            title          VARCHAR(512)  NOT NULL COMMENT '文档标题',
            source_url     VARCHAR(1024) NULL     COMMENT '原文出处，缺失则无法进入向量库',
            publisher      VARCHAR(64)   NOT NULL COMMENT '发布机构代码，关联 dim_publisher',
            platform       VARCHAR(32)   NOT NULL COMMENT '平台代码，首期固定 amazon',
            country        VARCHAR(16)   NOT NULL COMMENT '国家代码，EU 表示欧盟整体',
            doc_type       VARCHAR(32)   NOT NULL COMMENT '文档类型代码，关联 dim_doc_type',
            effective_date DATE          NULL     COMMENT '生效日期，允许为空',
            collected_date DATE          NOT NULL COMMENT '采集日期',
            status         VARCHAR(16)   NOT NULL COMMENT 'pending/ready/indexed/failed',
            missing_fields JSON          NULL     COMMENT '缺失的必填字段名清单',
            raw_path       VARCHAR(1024) NOT NULL COMMENT '原始文件绝对路径',
            content_hash   CHAR(64)      NOT NULL COMMENT '原始文件 SHA-256，用于去重',
            created_at     DATETIME      NOT NULL,
            updated_at     DATETIME      NOT NULL,
            PRIMARY KEY (doc_id),
            UNIQUE KEY uk_documents_content_hash (content_hash),
            KEY idx_documents_status (status),
            KEY idx_documents_filter (country, doc_type, publisher)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档元数据'
        """,
    ),
    (
        "chunks",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     CHAR(64)   NOT NULL COMMENT '父块 {doc_id}_pXXXX 或子块 {doc_id}_cXXXX',
            doc_id       CHAR(36)   NOT NULL COMMENT '所属文档',
            parent_id    CHAR(64)   NULL     COMMENT '父块标识，父块自身为空',
            level        VARCHAR(8) NOT NULL COMMENT 'parent 或 child',
            chunk_index  INT        NOT NULL COMMENT '同级内的序号，从 0 开始',
            text         MEDIUMTEXT NOT NULL COMMENT '块的正文',
            token_count  INT        NOT NULL COMMENT 'token 数',
            start_offset INT        NULL     COMMENT '在父块正文中的起始字符位置，子块必填',
            end_offset   INT        NULL     COMMENT '在父块正文中的结束字符位置，左闭右开',
            created_at   DATETIME   NOT NULL,
            PRIMARY KEY (chunk_id),
            KEY idx_chunks_doc (doc_id, level, chunk_index),
            KEY idx_chunks_parent (parent_id),
            CONSTRAINT fk_chunks_doc FOREIGN KEY (doc_id)
                REFERENCES documents (doc_id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='父子两级分块'
        """,
    ),
    (
        "dim_country",
        """
        CREATE TABLE IF NOT EXISTS dim_country (
            code      VARCHAR(16) NOT NULL COMMENT 'ISO 3166-1 alpha-2，或 EU',
            name_zh   VARCHAR(64) NOT NULL,
            name_en   VARCHAR(64) NOT NULL,
            is_active TINYINT(1)  NOT NULL DEFAULT 1 COMMENT '是否在首期范围内',
            PRIMARY KEY (code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='国家维度'
        """,
    ),
    (
        "dim_doc_type",
        """
        CREATE TABLE IF NOT EXISTS dim_doc_type (
            code      VARCHAR(32) NOT NULL COMMENT 'guideline/regulation/policy/faq',
            name_zh   VARCHAR(64) NOT NULL,
            name_en   VARCHAR(64) NOT NULL,
            is_active TINYINT(1)  NOT NULL DEFAULT 1,
            PRIMARY KEY (code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文档类型维度'
        """,
    ),
    (
        "dim_publisher",
        """
        CREATE TABLE IF NOT EXISTS dim_publisher (
            code         VARCHAR(64)  NOT NULL COMMENT 'amazon/eu_commission/...',
            name_zh      VARCHAR(64)  NOT NULL,
            name_en      VARCHAR(64)  NOT NULL,
            official_url VARCHAR(512) NULL COMMENT '官方站点，用于校验来源权威性',
            is_active    TINYINT(1)   NOT NULL DEFAULT 1,
            PRIMARY KEY (code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='发布机构维度'
        """,
    ),
    (
        "eval_cases",
        """
        CREATE TABLE IF NOT EXISTS eval_cases (
            case_id           CHAR(36)   NOT NULL COMMENT '用例唯一标识',
            question          TEXT       NOT NULL COMMENT '测试问题',
            expected_behavior VARCHAR(16) NOT NULL COMMENT 'answer 或 refuse',
            category          VARCHAR(32) NOT NULL COMMENT '分类，与 01-scope 的灰色地带对应',
            expected_doc_ids  JSON       NULL     COMMENT '期望命中的文档标识列表',
            notes             TEXT       NULL     COMMENT '出题依据与备注',
            created_at        DATETIME   NOT NULL,
            PRIMARY KEY (case_id),
            KEY idx_eval_cases_category (category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评测用例'
        """,
    ),
)

# 国家维度初始数据：(代码, 中文名, 英文名, 是否首期启用)
#
# 首期只启用 EU 与 Amazon 欧盟站点覆盖的成员国；其余预先登记但标为
# 不启用，扩充范围时只改数据不改表结构（见 01-scope 的「架构预留扩展位」）。
COUNTRY_SEED: tuple[tuple[str, str, str, bool], ...] = (
    ("EU", "欧盟", "European Union", True),
    ("DE", "德国", "Germany", True),
    ("FR", "法国", "France", True),
    ("IT", "意大利", "Italy", True),
    ("ES", "西班牙", "Spain", True),
    ("NL", "荷兰", "Netherlands", True),
    ("PL", "波兰", "Poland", True),
    ("BE", "比利时", "Belgium", False),
    ("AT", "奥地利", "Austria", False),
    ("SE", "瑞典", "Sweden", False),
    ("CZ", "捷克", "Czechia", False),
)

DOC_TYPE_SEED: tuple[tuple[str, str, str, bool], ...] = (
    ("guideline", "官方指南", "Guideline", True),
    ("regulation", "法规条文", "Regulation", True),
    ("policy", "平台政策", "Policy", True),
    ("faq", "常见问题", "FAQ", True),
)

# 发布机构：(代码, 中文名, 英文名, 官方站点, 是否首期启用)
#
# official_url 不只是展示用：采集语料时可据此校验来源是否权威，
# 避免把货代博客当成官方文档录进来。
PUBLISHER_SEED: tuple[tuple[str, str, str, str | None, bool], ...] = (
    ("amazon", "亚马逊", "Amazon", "https://sellercentral.amazon.com", True),
    (
        "eu_commission",
        "欧盟委员会",
        "European Commission",
        "https://europa.eu",
        True,
    ),
)


# ============================================================
# Milvus
# ============================================================

# 集合名由配置项 CBE_MILVUS__COLLECTION 提供，不在此处再写一份——
# 写两份会出现「改了这里不生效」的陷阱。
#
# VARCHAR 的 max_length 必须显式指定且建成后不可改，宁大勿小——
# 超长写入会直接报错，不会静默截断。
MILVUS_VARCHAR_LENGTHS: dict[str, int] = {
    "chunk_id": 64,
    "doc_id": 36,
    "parent_id": 64,
    "country": 16,
    "doc_type": 32,
    "publisher": 64,
}

# 需要建倒排索引的标量字段，即检索时的过滤维度
MILVUS_SCALAR_INDEX_FIELDS: tuple[str, ...] = ("country", "doc_type", "publisher")


# 稀疏向量只支持内积，其他度量对该类型无定义。
MILVUS_DENSE_INDEX: tuple[str, str] = ("FLAT", "COSINE")
MILVUS_SPARSE_INDEX: tuple[str, str] = ("SPARSE_INVERTED_INDEX", "IP")
MILVUS_SCALAR_INDEX_TYPE = "INVERTED"

# 集合的字段定义：(字段名, 类型名, 附加参数)
#
# **不含 dense_vector**：它的维度必须与嵌入模型一致，属于嵌入配置
# （CBE_EMBEDDING__DENSE_DIM），由调用方传给 create_collection，
# 在这里写死会形成第二个真相来源。
#
# 新增字段一律追加在末尾。
MILVUS_FIELDS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("chunk_id", "VARCHAR", {"is_primary": True, "max_length": 64}),
    ("sparse_vector", "SPARSE_FLOAT_VECTOR", {}),
    ("doc_id", "VARCHAR", {"max_length": 36}),
    ("parent_id", "VARCHAR", {"max_length": 64}),
    ("chunk_index", "INT32", {}),
    ("country", "VARCHAR", {"max_length": 16}),
    ("doc_type", "VARCHAR", {"max_length": 32}),
    ("publisher", "VARCHAR", {"max_length": 64}),
)
