"""连通性脚本中纯函数的单元测试。

脚本本身要连外部服务，但其中的文件检查与错误渲染是纯逻辑，可以单独测试。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from cbe_rag.config.settings import Settings
from check_services import check_embedding_model, report_config_error


def make_model_dir(tmp_path: Path, *files: str) -> Path:
    """建一个模型目录并放入指定文件。"""
    model_dir = tmp_path / "bge-m3"
    model_dir.mkdir()
    for name in files:
        (model_dir / name).write_text("stub", encoding="utf-8")
    return model_dir


class TestEmbeddingModelCheck:
    def test_missing_directory(self, tmp_path: Path) -> None:
        result = check_embedding_model(tmp_path / "not-there")

        assert result.ok is False
        assert "目录不存在" in result.detail

    def test_missing_config_json(self, tmp_path: Path) -> None:
        model_dir = make_model_dir(tmp_path, "model.safetensors")

        result = check_embedding_model(model_dir)

        assert result.ok is False
        assert "config.json" in result.detail

    def test_missing_weight_file(self, tmp_path: Path) -> None:
        # 只有配置没有权重，是复制模型时中断的典型结果
        model_dir = make_model_dir(tmp_path, "config.json", "tokenizer.json")

        result = check_embedding_model(model_dir)

        assert result.ok is False
        assert "权重文件" in result.detail

    @pytest.mark.parametrize(
        "weight_file",
        ["model.safetensors", "pytorch_model.bin"],
    )
    def test_complete_model_passes(self, tmp_path: Path, weight_file: str) -> None:
        model_dir = make_model_dir(tmp_path, "config.json", weight_file)

        result = check_embedding_model(model_dir)

        assert result.ok is True
        assert result.service == "bge-m3"

    def test_passes_path_through_to_detail(self, tmp_path: Path) -> None:
        model_dir = make_model_dir(tmp_path, "config.json", "model.safetensors")

        result = check_embedding_model(model_dir)

        assert str(model_dir) in result.detail


class TestConfigErrorReporting:
    def test_reports_field_names(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None)

        report_config_error(excinfo.value)

        output = capsys.readouterr().out
        assert "milvus" in output
        assert "配置加载失败" in output

    def test_does_not_leak_secret_values(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 回归：pydantic 的 ValidationError 会附带完整的输入字典，
        # 里面含密码。渲染时必须只取字段路径与错误类型，不能整条打印。
        for key in list(os.environ):
            if key.startswith("CBE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CBE_REDIS__PASSWORD", "super-secret-value")

        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None)

        report_config_error(excinfo.value)

        output = capsys.readouterr().out
        assert "super-secret-value" not in output
