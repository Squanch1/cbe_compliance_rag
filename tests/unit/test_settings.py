"""配置加载的单元测试。

不依赖任何外部服务，全部通过注入环境变量完成。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from cbe_rag.config.settings import (
    ENV_EXAMPLE_PATH as SETTINGS_ENV_EXAMPLE,
)
from cbe_rag.config.settings import (
    ENV_FILE_PATH as SETTINGS_ENV_FILE,
)
from cbe_rag.config.settings import (
    PROJECT_ROOT as SETTINGS_PROJECT_ROOT,
)
from cbe_rag.config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

# 必填项的最小集合。改这里的键名时，settings.py 中对应字段的必填性也必须一致。
REQUIRED_ENV: dict[str, str] = {
    "CBE_MILVUS__HOST": "192.168.88.101",
    "CBE_REDIS__HOST": "192.168.88.101",
    "CBE_REDIS__PASSWORD": "fake-redis-password",
    "CBE_MONGODB__HOST": "192.168.88.101",
    "CBE_MONGODB__USER": "cbe",
    "CBE_MONGODB__PASSWORD": "fake-mongo-password",
    "CBE_MYSQL__HOST": "127.0.0.1",
    "CBE_MYSQL__USER": "cbe",
    "CBE_MYSQL__PASSWORD": "fake-mysql-password",
    "CBE_LLM__API_KEY": "sk-fake-key",
    "CBE_LLM__MODEL": "qwen-plus",
}


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """注入一套完整可用的环境变量。"""
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def build_settings() -> Settings:
    """构造 Settings，禁用 .env 文件以免读到开发机上的真实配置。"""
    return Settings(_env_file=None)


class TestLoadSuccess:
    def test_loads_with_all_required_env(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        assert settings.milvus.host == "192.168.88.101"
        assert settings.mysql.host == "127.0.0.1"
        assert settings.llm.model == "qwen-plus"

    def test_optional_sections_are_not_required(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        # 这些字段缺失时不应阻止加载，但也不应提供业务默认值掩盖问题
        assert settings.retrieval.refuse_threshold is None


class TestDefaults:
    def test_connection_defaults(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        assert settings.milvus.port == 19530
        assert settings.redis.port == 6379
        assert settings.mongodb.port == 27017
        assert settings.mysql.port == 3306

    def test_mongodb_auth_source_defaults_to_admin(self, env: pytest.MonkeyPatch) -> None:
        # 账号默认建在 admin 库；若建在业务库需显式改这一项
        settings = build_settings()

        assert settings.mongodb.auth_source == "admin"

    def test_namespace_defaults(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        # Milvus 与 MySQL 必须使用独立 database，不得落在 default
        assert settings.milvus.database == "cbe_compliance"
        assert settings.mysql.database == "cbe_compliance"

    def test_redis_key_prefix_is_present(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        # Redis 实例由多个项目共用，前缀不可为空
        assert settings.redis.key_prefix == "cbe"

    def test_embedding_defaults(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        assert settings.embedding.dense_dim == 1024
        assert settings.embedding.use_fp16 is True


class TestMissingRequired:
    @pytest.mark.parametrize(
        "missing_key",
        [
            "CBE_REDIS__PASSWORD",
            "CBE_MYSQL__PASSWORD",
            "CBE_LLM__API_KEY",
            "CBE_LLM__MODEL",
            "CBE_MILVUS__HOST",
        ],
    )
    def test_missing_required_key_fails_fast(
        self, env: pytest.MonkeyPatch, missing_key: str
    ) -> None:
        env.delenv(missing_key)

        with pytest.raises(ValidationError):
            build_settings()

    def test_empty_string_is_rejected(self, env: pytest.MonkeyPatch) -> None:
        # 空字符串等于没配，不能当成有效值放行
        env.setenv("CBE_LLM__MODEL", "")

        with pytest.raises(ValidationError):
            build_settings()


class TestSecretHandling:
    def test_password_is_not_exposed_in_str(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        assert "fake-redis-password" not in str(settings.redis)
        assert "fake-mysql-password" not in str(settings.mysql)
        assert "sk-fake-key" not in str(settings.llm)

    def test_secret_can_be_read_explicitly(self, env: pytest.MonkeyPatch) -> None:
        settings = build_settings()

        assert settings.redis.password.get_secret_value() == "fake-redis-password"
        assert settings.llm.api_key.get_secret_value() == "sk-fake-key"


class TestEnvFileResolution:
    """配置文件的定位。

    回归：早期实现靠当前工作目录找 .env，在 PyCharm 里运行脚本时
    工作目录不是项目根，于是 Settings() 报出一堆「字段缺失」，
    而真正的原因是文件根本没找到。
    """

    def test_project_root_is_the_repository_root(self) -> None:
        # 源码布局若变化导致向上级数算错，这条会失败
        assert (SETTINGS_PROJECT_ROOT / "pyproject.toml").is_file()
        assert (SETTINGS_PROJECT_ROOT / "src" / "cbe_rag").is_dir()

    def test_env_file_path_is_absolute(self) -> None:
        assert SETTINGS_ENV_FILE.is_absolute()

    def test_env_file_sits_in_project_root(self) -> None:
        assert SETTINGS_ENV_FILE.parent == SETTINGS_PROJECT_ROOT
        assert SETTINGS_ENV_FILE.name == ".env"

    def test_env_example_sits_next_to_env_file(self) -> None:
        assert SETTINGS_ENV_EXAMPLE.parent == SETTINGS_ENV_FILE.parent
        assert SETTINGS_ENV_EXAMPLE.name == ".env.example"

    def test_default_env_file_does_not_depend_on_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert Settings.model_config["env_file"] == SETTINGS_ENV_FILE


class TestEnvExampleTemplate:
    """模板完整性。

    新增配置项却忘了同步 .env.example 时，这里会失败。
    这正是 CLAUDE.md 5.2 要求的「新增配置项必须同步更新 .env.example」。
    """

    def test_template_file_exists(self) -> None:
        assert ENV_EXAMPLE_PATH.is_file()

    def test_template_can_be_fully_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 清空环境变量，强制 Settings 只能从模板文件取值
        for key in list(os.environ):
            if key.startswith("CBE_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=ENV_EXAMPLE_PATH)

        assert settings.milvus.database == "cbe_compliance"
        assert settings.mysql.database == "cbe_compliance"
        assert settings.redis.key_prefix == "cbe"

    def test_template_leaves_threshold_uncalibrated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in list(os.environ):
            if key.startswith("CBE_"):
                monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=ENV_EXAMPLE_PATH)

        # 模板里这一项是注释掉的，表示阈值尚未标定
        assert settings.retrieval.refuse_threshold is None
