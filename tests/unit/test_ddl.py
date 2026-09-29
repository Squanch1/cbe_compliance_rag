"""存储结构定义的测试。

ddl.py 里有两处写的是同一件事：`DocumentStatus` 枚举，以及 documents
建表语句里 status 列的注释。两处漏改一处不会报错，只会在很久以后被人
发现「表上的注释是旧的」，因此用测试把它们钉在一起。
"""

from __future__ import annotations

import re

from cbe_rag.storage.ddl import MYSQL_TABLES, DocumentStatus


def documents_ddl() -> str:
    """取 documents 表的建表语句。"""
    return dict(MYSQL_TABLES)["documents"]


def status_comment() -> str:
    """从建表语句里取出 status 列的注释内容。"""
    match = re.search(
        r"status\s+VARCHAR\(\d+\)\s+NOT NULL COMMENT '([^']*)'", documents_ddl()
    )
    assert match is not None, "建表语句里找不到 status 列的注释"
    return match.group(1)


def values_in_comment() -> set[str]:
    """注释里用斜杠分隔的那一段，即状态值清单。

    注释形如 `pending/indexed/...，仅 indexed 参与检索`，
    逗号之后的说明文字不算状态值——那里面的 indexed 是叙述。
    """
    head = status_comment().split("，")[0]
    return set(head.split("/"))


class TestDocumentStatusMatchesDdl:
    def test_comment_lists_exactly_the_enum_values(self) -> None:
        # 双向比对：枚举加了值而注释没改，或注释还留着已删的值，都会失败
        assert values_in_comment() == {status.value for status in DocumentStatus}

    def test_comment_states_that_only_indexed_is_retrievable(self) -> None:
        # 「只有 indexed 参与检索」是检索层的实现依据，写在表注释里
        # 才不会被接下来看表结构的人漏掉
        assert "indexed" in status_comment()

    def test_every_status_value_is_lowercase(self) -> None:
        # 库里存小写，与 spec 和检索时的过滤条件一致
        assert all(
            status.value == status.value.lower() for status in DocumentStatus
        )
