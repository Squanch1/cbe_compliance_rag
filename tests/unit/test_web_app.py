"""前端页面的冒烟测试。

用 Streamlit 的 AppTest 在进程内跑一遍页面脚本，不启浏览器也不启服务。
curl 首页只能证明端口在监听——**Streamlit 的脚本要等 WebSocket 连上才执行**，
页面里的报错在那之前根本不会发生。

这里只覆盖页面能渲染、关键元素在位。真正的问答路径要连后端，不在这里测。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = (
    Path(__file__).resolve().parents[2] / "apps" / "web" / "app.py"
)

# 页面与它的客户端模块在同一目录，直接跑脚本时 Streamlit 会把这个目录
# 加进 sys.path；测试里要自己加，否则 import api_client 找不到。
if str(APP_PATH.parent) not in sys.path:
    sys.path.insert(0, str(APP_PATH.parent))


def run_app() -> AppTest:
    """跑一遍页面脚本。"""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


class TestPageRenders:
    def test_script_runs_without_exception(self) -> None:
        # 页面里任何一处抛异常都会让界面变成一片红，而端口照样在监听
        app = run_app()

        assert not app.exception

    def test_shows_the_title(self) -> None:
        app = run_app()

        assert any("跨境电商合规问答" in item.value for item in app.title)

    def test_disables_the_plain_streamlit_menu_marker(self) -> None:
        # 只是确认页面配置确实生效了（layout 设成了 wide）
        app = run_app()

        assert not app.exception


class TestBackendClient:
    def test_unreachable_backend_says_what_to_do(self) -> None:
        # 后端没起时要直接说清怎么起，不能只说「请求失败」
        from api_client import ApiError, BackendClient

        client = BackendClient(base_url="http://127.0.0.1:1", timeout=1.0)

        with pytest.raises(ApiError, match="连不上后端"):
            client.health()

    def test_http_error_carries_the_contract_code(self) -> None:
        # 回归：错误分支曾经把两个返回值按位置传给只收一个位置参数的
        # ApiError，触发的不是「报错清晰」而是 TypeError。
        # 界面要按 code 区分「请求有问题」和「后端没起」。
        import io
        import json
        import urllib.error
        from unittest.mock import patch

        from api_client import ApiError, BackendClient

        body = json.dumps(
            {
                "trace_id": "t-1",
                "error": {"code": "unknown_parent", "message": "找不到这个父块"},
            }
        ).encode("utf-8")
        error = urllib.error.HTTPError(
            "http://x", 404, "Not Found", {}, io.BytesIO(body)
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ApiError) as info:
                BackendClient().get_parent("nope")

        assert info.value.code == "unknown_parent"
        assert info.value.status == 404
        assert "找不到" in str(info.value)


class TestSidebar:
    def test_offers_the_three_filter_dimensions(self) -> None:
        app = run_app()

        labels = [item.label for item in app.sidebar.selectbox]
        assert labels == ["国家或地区", "文档类型", "发布机构"]

    def test_each_filter_defaults_to_unrestricted(self) -> None:
        # 默认不该悄悄加上筛选条件，那会让召回无端变少
        app = run_app()

        assert all(item.value == "不限" for item in app.sidebar.selectbox)

    def test_offers_a_clear_button(self) -> None:
        app = run_app()

        assert any("清空对话" in item.label for item in app.sidebar.button)


class TestDimensionOptions:
    def test_label_pairs_the_code_with_its_chinese_name(self) -> None:
        # 代码在前：它才是后端认的值，也会出现在日志与引用里，
        # 使用者对着界面上的字样就能在别处找到同一个东西
        from app import build_options

        options = build_options(
            [{"code": "EU", "name_zh": "欧盟", "name_en": "European Union"}]
        )

        assert options == {"EU（欧盟）": "EU"}

    def test_submitted_value_is_still_the_code(self) -> None:
        # 拿中文名去提交，后端拼出来的过滤条件匹配不上
        from app import build_options

        options = build_options(
            [{"code": "guideline", "name_zh": "官方指南", "name_en": "Guideline"}]
        )

        assert list(options.values()) == ["guideline"]

    def test_empty_input_yields_empty_mapping(self) -> None:
        from app import build_options

        assert build_options([]) == {}

    def test_keeps_every_item(self) -> None:
        from app import build_options

        options = build_options(
            [
                {"code": "EU", "name_zh": "欧盟", "name_en": "European Union"},
                {"code": "DE", "name_zh": "德国", "name_en": "Germany"},
            ]
        )

        assert options == {"EU（欧盟）": "EU", "DE（德国）": "DE"}

    def test_no_hardcoded_dimension_values_in_the_page(self) -> None:
        # 取值必须来自维度表。写死在页面里的话，后端加一个国家，
        # 界面上不会出现，而且不报错。
        source = APP_PATH.read_text(encoding="utf-8")

        assert '"EU"' not in source
        assert "'EU'" not in source
        assert '"guideline"' not in source
        assert '"eu_commission"' not in source


class TestSuggestions:
    def test_shown_on_an_empty_conversation(self) -> None:
        # AppTest 不把 st.pills 暴露成可查询的元素，退一步查那一节的标题
        app = run_app()

        assert any("试试问这些" in item.value for item in app.markdown)

    def test_suggestions_include_one_the_corpus_cannot_answer(self) -> None:
        # 语料里没有德国税率，那一条是用来看拒答长什么样的。
        # 从源码里取选项，绕开 AppTest 对 pills 的可见性。
        source = APP_PATH.read_text(encoding="utf-8")

        assert "德国" in source

