"""连通性脚本中纯函数的单元测试。

探测逻辑已经下沉到 storage/ 的适配器里，本文件只覆盖脚本自身的
配置错误渲染。模型文件检查的测试见 test_embedding_store.py。
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from cbe_rag.config.settings import Settings
from check_services import report_config_error


class TestConfigErrorReporting:
    def test_reports_field_names(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None)

        report_config_error(excinfo.value)

        output = capsys.readouterr().out
        assert "milvus" in output
        assert "配置加载失败" in output

    def test_points_at_the_template_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None)

        report_config_error(excinfo.value)

        assert ".env.example" in capsys.readouterr().out

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

        assert "super-secret-value" not in capsys.readouterr().out
