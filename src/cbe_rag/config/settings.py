"""应用配置。

所有配置只从环境变量或 .env 文件读取。缺失必填项时加载即失败，
不带默认值静默降级（见 CLAUDE.md 5.2）。

环境变量命名规则：CBE_<段名>__<字段名>，段名与字段名均为大写。
例如 Milvus 的 host 对应 CBE_MILVUS__HOST。

新增配置项时必须同步更新 .env.example。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# 配置文件按本模块位置推算绝对路径，而不是靠当前工作目录。
# 靠工作目录会在 IDE 里出问题：PyCharm 运行脚本时的工作目录不一定是项目根，
# 于是 Settings() 找不到 .env，报出一堆「字段缺失」，而真正的原因是文件没找到。
# 源码布局为 <项目根>/src/cbe_rag/config/settings.py，故向上四级即项目根。
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 配置模板与真实配置的位置，供脚本判断文件是否存在并给出准确提示
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

_DEFAULT_ENV_FILE = ENV_FILE_PATH


class _Section(BaseModel):
    """配置段的基类。

    禁止未声明的字段：环境变量名写错时应当在启动阶段就报错，
    而不是静默套用默认值继续运行。
    """

    model_config = ConfigDict(extra="forbid")


class MilvusConfig(_Section):
    """Milvus 向量库连接参数。"""

    host: str = Field(min_length=1, description="服务地址")
    port: int = Field(default=19530, ge=1, le=65535, description="服务端口")
    database: str = Field(
        default="cbe_compliance",
        min_length=1,
        description="独立 database 名，不得写入 default",
    )
    collection: str = Field(
        default="chunks_v1",
        min_length=1,
        description="子块向量集合名，带版本后缀",
    )
    timeout_seconds: float = Field(default=10.0, gt=0, description="网络调用超时秒数")


class RedisConfig(_Section):
    """Redis 连接与键生命周期参数。"""

    host: str = Field(min_length=1)
    port: int = Field(default=6379, ge=1, le=65535)
    password: SecretStr = Field(description="必填，实例开启了认证")
    key_prefix: str = Field(
        default="cbe",
        min_length=1,
        description="键前缀，实例由多个项目共用，不可为空",
    )
    session_ttl_seconds: int = Field(default=3600, gt=0, description="会话上下文过期秒数")
    cache_ttl_seconds: int = Field(default=600, gt=0, description="问答缓存过期秒数")
    lock_ttl_seconds: int = Field(default=300, gt=0, description="索引构建锁过期秒数")
    timeout_seconds: float = Field(default=5.0, gt=0)


class MongodbConfig(_Section):
    """MongoDB 连接参数。存放原始解析产物，M2 阶段启用。"""

    host: str = Field(min_length=1)
    port: int = Field(default=27017, ge=1, le=65535)
    user: str = Field(min_length=1)
    password: SecretStr
    database: str = Field(default="cbe_compliance", min_length=1)
    auth_source: str = Field(
        default="admin",
        min_length=1,
        description="认证库。账号建在 admin 库时保持默认，建在业务库时改为业务库名",
    )
    timeout_seconds: float = Field(default=10.0, gt=0)


class MysqlConfig(_Section):
    """MySQL 连接参数。存放文档元数据与维度表。"""

    host: str = Field(min_length=1)
    port: int = Field(default=3306, ge=1, le=65535)
    user: str = Field(min_length=1)
    password: SecretStr
    database: str = Field(default="cbe_compliance", min_length=1)
    charset: str = Field(default="utf8mb4", min_length=1)
    timeout_seconds: float = Field(default=10.0, gt=0)


class LlmConfig(_Section):
    """阿里云百炼生成模型参数，走 OpenAI 兼容模式。"""

    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        min_length=1,
    )
    api_key: SecretStr
    model: str = Field(
        min_length=1,
        description="模型名无默认值，必须显式指定",
    )
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_tokens: int = Field(default=2048, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class EmbeddingConfig(_Section):
    """bge-m3 嵌入模型参数，在 Python 进程内加载。"""

    model_path: Path = Field(
        default=Path("models/bge-m3"),
        description="HuggingFace 格式模型目录，相对项目根或绝对路径",
    )
    use_fp16: bool = Field(
        default=True,
        description="固定 fp16，与已建索引的精度必须一致，不得中途切换",
    )
    device: str = Field(default="cuda", min_length=1)
    dense_dim: int = Field(default=1024, gt=0, description="稠密向量维度")
    batch_size: int = Field(default=8, gt=0, description="推理批大小，受显存限制")


class RetrievalConfig(_Section):
    """检索与融合参数。"""

    dense_limit: int = Field(default=20, gt=0, description="稠密路候选数量")
    sparse_limit: int = Field(
        default=40,
        gt=0,
        description="稀疏路候选数量，通常大于稠密路，因其分数分布更陡",
    )
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="融合时稠密路权重")
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="融合时稀疏路权重")
    refuse_threshold: float | None = Field(
        default=None,
        description=(
            "拒答阈值，判据为稠密路最高余弦相似度。"
            "None 表示尚未标定，消费方遇到 None 必须显式报错，不得放行"
        ),
    )


class Settings(BaseSettings):
    """全部配置的入口。各段通过环境变量注入。"""

    model_config = SettingsConfigDict(
        env_prefix="CBE_",
        env_nested_delimiter="__",
        env_file=_DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="forbid",
    )

    milvus: MilvusConfig
    redis: RedisConfig
    mongodb: MongodbConfig
    mysql: MysqlConfig
    llm: LlmConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的配置实例。

    缓存是因为嵌入模型路径等配置会在多处读取，且加载过程包含文件访问。
    测试中需要重新加载时调用 get_settings.cache_clear()。
    """
    return Settings()
