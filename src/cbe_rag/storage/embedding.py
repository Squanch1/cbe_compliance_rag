"""bge-m3 嵌入适配器。

模型在 Python 进程内加载，一次调用同时产出稠密与稀疏两路向量
（见 docs/spec/02-architecture.md 第 6.6 节）。

业务层通过本类编码文本，不直接 import milvus_model 或 FlagEmbedding
（见 CLAUDE.md 5.3）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from cbe_rag.config.settings import EmbeddingConfig
from cbe_rag.storage.health import HealthResult

# HuggingFace 格式模型的必需文件。Ollama 的 GGUF 文件不含这些，
# 因此在进程内加载的场景下不可用。
REQUIRED_MODEL_FILES = ("config.json",)

# 权重文件二选一。safetensors 是较新的格式；pytorch_model.bin 是
# 较早的 pickle 格式，bge-m3 的官方发布仍在用它。
WEIGHT_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True)
class EmbeddingResult:
    """一次编码的产出，两路向量。

    sparse 已从 scipy 矩阵转成 {词元ID: 权重} 字典，
    调用方可以直接拿它喂给 Milvus，不必接触 scipy。
    """

    dense: list[list[float]]
    sparse: list[dict[int, float]]


class EncoderProtocol(Protocol):
    """编码器需要提供的能力。

    定义成协议是为了让单元测试注入假实现，不加载真实的 2.3GB 模型。
    真实的 BGEM3EmbeddingFunction 在结构上满足本协议。
    """

    def __call__(self, texts: list[str]) -> dict[str, Any]:
        """编码一批文本，返回含 dense 与 sparse 两个键的字典。"""
        ...

    @property
    def dim(self) -> dict[str, int]:
        """返回各路的维度，至少含 dense 与 sparse。"""
        ...


def sparse_matrix_to_dicts(matrix: Any) -> list[dict[int, float]]:
    """把稀疏矩阵转成 Milvus 需要的 {词元ID: 权重} 字典列表。

    按行遍历 CSR 的 indptr 区间，避免逐行 getrow 带来的开销。
    """
    csr = matrix.tocsr()
    rows: list[dict[int, float]] = []
    for index in range(csr.shape[0]):
        start, end = csr.indptr[index], csr.indptr[index + 1]
        rows.append(
            {
                int(token): float(weight)
                for token, weight in zip(
                    csr.indices[start:end], csr.data[start:end]
                )
            }
        )
    return rows


class EmbeddingStore:
    """bge-m3 访问入口。

    构造本对象不加载模型；模型在首次 encode 时建立并复用。
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        encoder_factory: Callable[[], EncoderProtocol] | None = None,
    ) -> None:
        """初始化。

        encoder_factory 仅用于测试注入；生产路径下由配置构造真实编码器。
        """
        self._config = config
        self._encoder_factory: Callable[[], EncoderProtocol] = (
            encoder_factory if encoder_factory is not None else self._load_encoder
        )
        self._encoder: EncoderProtocol | None = None

    def _load_encoder(self) -> EncoderProtocol:
        """加载真实的 bge-m3 模型。

        导入放在函数内而不是模块顶部：milvus_model 会连带拉起
        FlagEmbedding 与 torch，放在顶部会让 import cbe_rag.storage
        变慢数秒，而多数场景根本不需要加载模型。
        """
        from milvus_model.hybrid import BGEM3EmbeddingFunction

        return BGEM3EmbeddingFunction(
            model_name_or_path=str(self._config.model_path),
            use_fp16=self._config.use_fp16,
            device=self._config.device,
        )

    def _get_encoder(self) -> EncoderProtocol:
        """返回已加载的编码器，没有就先加载。

        加载后立即校验稠密维度。维度与配置不符会让建索引和查询
        落在不同的向量空间里，相似度全错却不报错——必须当场失败。
        """
        if self._encoder is None:
            encoder = self._encoder_factory()
            actual = encoder.dim.get("dense")
            if actual != self._config.dense_dim:
                raise ValueError(
                    "模型稠密维度与配置不符：模型 %s，配置 %s。"
                    "维度不一致会让索引与查询产生错误的相似度。"
                    % (actual, self._config.dense_dim)
                )
            self._encoder = encoder
        return self._encoder

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """编码一批文本，返回稠密与稀疏两路向量。

        按配置的批大小分批调用，避免一次性把全部文本压进显存。

        与 health_check 不同，这里**不吞异常**：编码是真实操作，
        失败必须让调用方知道，否则会拿空结果继续往下走。
        """
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])

        encoder = self._get_encoder()
        batch_size = self._config.batch_size

        dense_all: list[list[float]] = []
        sparse_all: list[dict[int, float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            output = encoder(batch)
            dense_all.extend([float(value) for value in vector] for vector in output["dense"])
            sparse_all.extend(sparse_matrix_to_dicts(output["sparse"]))

        return EmbeddingResult(dense=dense_all, sparse=sparse_all)

    def health_check(self) -> HealthResult:
        """检查模型是否可用。

        只检查文件结构与当前加载状态，**不触发加载**——首次加载需数秒
        至数十秒（本机实测约 10 秒）并占用显存，放进连通性检查会让
        每次执行都卡住。真正的加载验证由首次 encode 完成。
        """
        started = time.perf_counter()
        model_dir = Path(self._config.model_path)
        precision = "fp16" if self._config.use_fp16 else "fp32"

        if not model_dir.is_dir():
            ok = False
            detail = "模型目录不存在：%s" % model_dir
        else:
            missing = [
                name
                for name in REQUIRED_MODEL_FILES
                if not (model_dir / name).is_file()
            ]
            if missing:
                ok = False
                detail = "缺少文件 %s（目录 %s）" % ("、".join(missing), model_dir)
            elif not any(
                (model_dir / name).is_file() for name in WEIGHT_FILE_CANDIDATES
            ):
                ok = False
                detail = "缺少权重文件，需 %s 之一（目录 %s）" % (
                    " 或 ".join(WEIGHT_FILE_CANDIDATES),
                    model_dir,
                )
            else:
                state = (
                    "已加载"
                    if self._encoder is not None
                    else "未加载（首次 encode 时加载）"
                )
                ok = True
                detail = "路径=%s 精度=%s 模型=%s" % (model_dir, precision, state)

        return HealthResult(
            service="bge-m3",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放模型引用。

        torch 的缓存分配器可能仍持有显存直到垃圾回收触发。
        需要立刻归还显存时，调用方在 close() 之后显式执行
        gc.collect() 与 torch.cuda.empty_cache()。
        """
        self._encoder = None
