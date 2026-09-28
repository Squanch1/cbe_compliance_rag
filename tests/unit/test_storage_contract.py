"""四个存储适配器的接口一致性测试。

连通性脚本、健康检查接口、FastAPI 生命周期钩子都需要统一遍历所有适配器，
因此每个适配器都必须实现 StorageAdapter 协议。任何一个漏掉方法，
调用方就被迫写 hasattr 判断——这是本文件要防住的情况。

（这个问题真实发生过：Redis 适配器最初漏了 close()，
直到四个适配器放在一起跑才暴露。）
"""

from __future__ import annotations

import inspect

import pytest

from cbe_rag.storage import (
    EmbeddingStore,
    MilvusStore,
    MongoStore,
    MysqlStore,
    RedisStore,
)
from cbe_rag.storage.health import StorageAdapter

ADAPTER_CLASSES = [
    MilvusStore,
    MongoStore,
    MysqlStore,
    RedisStore,
    EmbeddingStore,
]


def class_name(cls: type) -> str:
    return cls.__name__


class TestUniformInterface:
    @pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES, ids=class_name)
    def test_has_health_check(self, adapter_cls: type) -> None:
        assert callable(getattr(adapter_cls, "health_check", None)), (
            "%s 缺少 health_check，调用方无法统一遍历适配器" % adapter_cls.__name__
        )

    @pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES, ids=class_name)
    def test_has_close(self, adapter_cls: type) -> None:
        assert callable(getattr(adapter_cls, "close", None)), (
            "%s 缺少 close，调用方无法统一释放资源" % adapter_cls.__name__
        )

    @pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES, ids=class_name)
    def test_protocol_methods_accept_only_self(self, adapter_cls: type) -> None:
        # 参数不一致会让统一调用失效
        for method_name in ("health_check", "close"):
            signature = inspect.signature(getattr(adapter_cls, method_name))
            assert list(signature.parameters) == ["self"], (
                "%s.%s 的参数不是仅 self：%s"
                % (adapter_cls.__name__, method_name, list(signature.parameters))
            )


class CompleteAdapter:
    """两个方法都实现了。"""

    def health_check(self) -> None:
        return None

    def close(self) -> None:
        return None


class AdapterMissingClose:
    """只实现了 health_check。"""

    def health_check(self) -> None:
        return None


class TestProtocolActuallyEnforces:
    """验证 StorageAdapter 协议确实要求两个方法都在。

    这决定了漏写 close() 能否在类型检查阶段被发现，
    而不是等到调用方崩溃。
    """

    def test_complete_implementation_satisfies_protocol(self) -> None:
        assert isinstance(CompleteAdapter(), StorageAdapter)

    def test_missing_close_violates_protocol(self) -> None:
        assert not isinstance(AdapterMissingClose(), StorageAdapter)
