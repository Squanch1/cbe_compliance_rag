"""嵌入适配器的单元测试。

注入假的编码器，不加载真实的 2.3GB 模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scipy.sparse import csr_matrix

from cbe_rag.config.settings import EmbeddingConfig
from cbe_rag.storage.embedding import EmbeddingStore

FAKE_DENSE_DIM = 1024
FAKE_SPARSE_DIM = 100


def make_config(model_path: Path, *, batch_size: int = 8) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_path=model_path,
        use_fp16=True,
        device="cpu",
        dense_dim=FAKE_DENSE_DIM,
        batch_size=batch_size,
    )


class FakeEncoder:
    """假的编码器。

    稀疏向量用真实的 scipy CSR 矩阵返回，让适配器里的格式转换
    接受真实结构的检验，而不是一个简化过的替身。
    """

    def __init__(
        self,
        *,
        dense_dim: int = FAKE_DENSE_DIM,
        sparse_dim: int = FAKE_SPARSE_DIM,
        fail_with: Exception | None = None,
    ) -> None:
        self._dense_dim = dense_dim
        self._sparse_dim = sparse_dim
        self._fail_with = fail_with
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> dict[str, Any]:
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append(list(texts))

        count = len(texts)
        dense = [[0.1] * self._dense_dim for _ in range(count)]

        # 每条给两个非零项，索引各不相同，便于验证转换没串行
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for index in range(count):
            rows.extend([index, index])
            cols.extend([index, index + 10])
            values.extend([0.5, 0.25])

        sparse = csr_matrix(
            (values, (rows, cols)), shape=(count, self._sparse_dim)
        )
        return {"dense": dense, "sparse": sparse}

    @property
    def dim(self) -> dict[str, int]:
        return {"dense": self._dense_dim, "sparse": self._sparse_dim}


class RecordingFactory:
    """记录编码器工厂被调用了几次。"""

    def __init__(self, encoder: FakeEncoder | None = None) -> None:
        self._encoder = encoder if encoder is not None else FakeEncoder()
        self.created: list[FakeEncoder] = []

    def __call__(self) -> FakeEncoder:
        self.created.append(self._encoder)
        return self._encoder

    @property
    def encoder(self) -> FakeEncoder:
        return self._encoder


def build_model_dir(tmp_path: Path, *files: str) -> Path:
    """建一个模型目录并放入指定文件。"""
    model_dir = tmp_path / "bge-m3"
    model_dir.mkdir(exist_ok=True)
    for name in files:
        (model_dir / name).write_text("stub", encoding="utf-8")
    return model_dir


def complete_model_dir(tmp_path: Path) -> Path:
    """建一个文件齐全的模型目录。"""
    return build_model_dir(tmp_path, "config.json", "model.safetensors", "tokenizer.json")


def build_store(
    tmp_path: Path, *, batch_size: int = 8, encoder: FakeEncoder | None = None
) -> tuple[EmbeddingStore, RecordingFactory]:
    factory = RecordingFactory(encoder)
    store = EmbeddingStore(make_config(tmp_path, batch_size=batch_size), factory)
    return store, factory


class TestConstruction:
    def test_does_not_load_model(self, tmp_path: Path) -> None:
        # 加载需数十秒并占用显存，构造时不能做这件事
        store, factory = build_store(tmp_path)

        assert factory.created == []

    def test_health_check_does_not_load_model(self, tmp_path: Path) -> None:
        # 否则每次连通性检查都要卡上十几秒
        model_dir = complete_model_dir(tmp_path)
        store, factory = build_store(model_dir)

        store.health_check()

        assert factory.created == []


class TestEncode:
    def test_returns_one_dense_vector_per_input(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path)

        result = store.encode(["a", "b", "c"])

        assert len(result.dense) == 3

    def test_dense_vector_has_configured_dimension(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path)

        result = store.encode(["a"])

        assert len(result.dense[0]) == FAKE_DENSE_DIM

    def test_returns_one_sparse_dict_per_input(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path)

        result = store.encode(["a", "b", "c"])

        assert len(result.sparse) == 3

    def test_sparse_dict_keys_are_int_and_values_are_float(self, tmp_path: Path) -> None:
        # Milvus 插入稀疏向量要求 {词元ID: 权重} 这种格式
        store, _ = build_store(tmp_path)

        result = store.encode(["a"]).sparse[0]

        assert result, "稀疏向量不应为空"
        for key, value in result.items():
            assert isinstance(key, int)
            assert isinstance(value, float)

    def test_sparse_rows_are_not_shifted(self, tmp_path: Path) -> None:
        # 每条文本的稀疏项索引不同，若转换时错位即可被发现
        store, _ = build_store(tmp_path)

        result = store.encode(["a", "b", "c"])

        assert 0 in result.sparse[0]
        assert 1 in result.sparse[1]
        assert 2 in result.sparse[2]

    def test_empty_input_returns_empty_result(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path)

        result = store.encode([])

        assert result.dense == []
        assert result.sparse == []

    def test_empty_input_does_not_call_encoder(self, tmp_path: Path) -> None:
        store, factory = build_store(tmp_path)

        store.encode([])

        assert factory.encoder.calls == []

    def test_splits_input_into_batches(self, tmp_path: Path) -> None:
        store, factory = build_store(tmp_path, batch_size=2)

        store.encode(["a", "b", "c", "d", "e"])

        assert [len(call) for call in factory.encoder.calls] == [2, 2, 1]

    def test_batch_order_is_preserved(self, tmp_path: Path) -> None:
        store, factory = build_store(tmp_path, batch_size=2)

        store.encode(["a", "b", "c"])

        assert factory.encoder.calls == [["a", "b"], ["c"]]

    def test_encoder_is_created_once_across_calls(self, tmp_path: Path) -> None:
        store, factory = build_store(tmp_path)

        store.encode(["a"])
        store.encode(["b"])

        assert len(factory.created) == 1

    def test_dimension_mismatch_is_rejected(self, tmp_path: Path) -> None:
        # 维度与配置不符会让建索引和查询落在不同的向量空间里，
        # 相似度全错却不报错——必须当场失败
        mismatched = FakeEncoder(dense_dim=768)
        store, _ = build_store(tmp_path, encoder=mismatched)

        with pytest.raises(ValueError, match="768"):
            store.encode(["a"])

    def test_dimension_mismatch_message_names_both_values(self, tmp_path: Path) -> None:
        mismatched = FakeEncoder(dense_dim=768)
        store, _ = build_store(tmp_path, encoder=mismatched)

        with pytest.raises(ValueError) as excinfo:
            store.encode(["a"])

        assert "768" in str(excinfo.value)
        assert str(FAKE_DENSE_DIM) in str(excinfo.value)

    def test_encoder_failure_propagates(self, tmp_path: Path) -> None:
        # 与 health_check 不同，encode 是真实操作，失败必须让调用方知道，
        # 不能吞掉后返回空结果
        failing = FakeEncoder(fail_with=RuntimeError("cuda out of memory"))
        store, _ = build_store(tmp_path, encoder=failing)

        with pytest.raises(RuntimeError, match="cuda out of memory"):
            store.encode(["a"])


class TestHealthCheck:
    def test_missing_directory_is_reported(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path / "not-there")

        result = store.health_check()

        assert result.ok is False
        assert "不存在" in result.detail

    def test_missing_config_json_is_reported(self, tmp_path: Path) -> None:
        model_dir = build_model_dir(tmp_path, "model.safetensors")
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert result.ok is False
        assert "config.json" in result.detail

    def test_missing_weight_file_is_reported(self, tmp_path: Path) -> None:
        model_dir = build_model_dir(tmp_path, "config.json")
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert result.ok is False
        assert "权重" in result.detail

    @pytest.mark.parametrize("weight_file", ["model.safetensors", "pytorch_model.bin"])
    def test_complete_model_passes(self, tmp_path: Path, weight_file: str) -> None:
        model_dir = build_model_dir(tmp_path, "config.json", weight_file)
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert result.ok is True
        assert result.service == "bge-m3"

    def test_reports_unloaded_state(self, tmp_path: Path) -> None:
        model_dir = complete_model_dir(tmp_path)
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert "未加载" in result.detail

    def test_reports_loaded_state_after_encode(self, tmp_path: Path) -> None:
        model_dir = complete_model_dir(tmp_path)
        store, _ = build_store(model_dir)
        store.encode(["a"])

        result = store.health_check()

        assert "已加载" in result.detail

    def test_reports_precision(self, tmp_path: Path) -> None:
        model_dir = complete_model_dir(tmp_path)
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert "fp16" in result.detail

    def test_path_is_included_for_troubleshooting(self, tmp_path: Path) -> None:
        model_dir = complete_model_dir(tmp_path)
        store, _ = build_store(model_dir)

        result = store.health_check()

        assert str(model_dir) in result.detail


class TestClose:
    def test_close_before_any_use_is_safe(self, tmp_path: Path) -> None:
        store, _ = build_store(tmp_path)

        store.close()

    def test_close_releases_so_next_encode_reloads(self, tmp_path: Path) -> None:
        store, factory = build_store(tmp_path)
        store.encode(["a"])

        store.close()
        store.encode(["b"])

        # 关闭后应重新加载，而不是复用已释放的模型
        assert len(factory.created) == 2

    def test_health_check_after_close_reports_unloaded(self, tmp_path: Path) -> None:
        model_dir = complete_model_dir(tmp_path)
        store, _ = build_store(model_dir)
        store.encode(["a"])

        store.close()

        assert "未加载" in store.health_check().detail
