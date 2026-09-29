"""文件哈希的单元测试。

哈希算错的表现不是崩溃，而是「同一份文件每次算出来都不一样」
或「不同文件算出同一个值」——前者导致重复入库，后者导致丢文档，
两种都不会报错。因此边界要测到。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cbe_rag.indexing.hashing import HashingError, file_content_hash

TEXT = "Import One-Stop Shop 进口一站式服务\n"


def write(tmp_path: Path, name: str, content: bytes) -> Path:
    """在临时目录写一个文件，返回绝对路径。"""
    path = tmp_path / name
    path.write_bytes(content)
    return path


class TestResultShape:
    def test_same_file_hashes_the_same(self, tmp_path: Path) -> None:
        path = write(tmp_path, "a.txt", b"content")

        assert file_content_hash(path) == file_content_hash(path)

    def test_result_is_64_lowercase_hex_chars(self, tmp_path: Path) -> None:
        # 对不上 DDL 的 CHAR(64) 会在插入时被截断或报错
        path = write(tmp_path, "a.txt", b"content")

        result = file_content_hash(path)

        assert len(result) == 64
        assert all(char in "0123456789abcdef" for char in result)

    def test_matches_hashlib_computed_directly(self, tmp_path: Path) -> None:
        # 与标准库直接算的结果对齐，确认既没多读也没少读字节
        payload = TEXT.encode("utf-8")
        path = write(tmp_path, "a.txt", payload)

        assert file_content_hash(path) == hashlib.sha256(payload).hexdigest()


class TestContentSensitivity:
    def test_different_content_yields_different_hash(self, tmp_path: Path) -> None:
        first = write(tmp_path, "a.txt", b"content one")
        second = write(tmp_path, "b.txt", b"content two")

        assert file_content_hash(first) != file_content_hash(second)

    def test_renaming_does_not_change_the_hash(self, tmp_path: Path) -> None:
        # 改名不该被当成新文档——这正是用内容而非文件名做判重的理由
        original = write(tmp_path, "faq.html", TEXT.encode("utf-8"))
        renamed = write(tmp_path, "faq-2026.html", TEXT.encode("utf-8"))

        assert file_content_hash(original) == file_content_hash(renamed)

    def test_one_byte_difference_is_detected(self, tmp_path: Path) -> None:
        # 文档更新常常只改一处，必须能识别出来
        first = write(tmp_path, "a.txt", b"rate 19%")
        second = write(tmp_path, "b.txt", b"rate 20%")

        assert file_content_hash(first) != file_content_hash(second)


class TestChunkBoundaries:
    def test_file_larger_than_chunk_size(self, tmp_path: Path) -> None:
        # 分块读最容易在最后一块出错：漏读会少字节，重复读会多字节，
        # 两种情况算出的哈希都不对，但都不会报错
        payload = b"0123456789abcdefghij"  # 20 字节，8 字节的块要读三次
        path = write(tmp_path, "a.txt", payload)

        assert file_content_hash(path, chunk_size=8) == hashlib.sha256(
            payload
        ).hexdigest()

    def test_size_exactly_a_multiple_of_chunk_size(self, tmp_path: Path) -> None:
        # 恰好整除时，最后一块读完立即碰到文件尾，循环要多跑一次空读才退出
        payload = b"01234567" * 3  # 24 字节，块大小 8
        path = write(tmp_path, "a.txt", payload)

        assert file_content_hash(path, chunk_size=8) == hashlib.sha256(
            payload
        ).hexdigest()

    def test_chunk_size_larger_than_file(self, tmp_path: Path) -> None:
        payload = b"short"
        path = write(tmp_path, "a.txt", payload)

        assert file_content_hash(path, chunk_size=1024) == hashlib.sha256(
            payload
        ).hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        # 空文件的 SHA-256 是个固定值，不是空字符串
        path = write(tmp_path, "empty.txt", b"")

        assert file_content_hash(path) == hashlib.sha256(b"").hexdigest()

    def test_chunk_size_of_zero_raises(self, tmp_path: Path) -> None:
        # read(0) 永远返回空，循环立刻结束，会静默算出空文件的哈希
        path = write(tmp_path, "a.txt", b"content")

        with pytest.raises(HashingError, match="chunk_size"):
            file_content_hash(path, chunk_size=0)


class TestInputValidation:
    def test_relative_path_raises(self) -> None:
        with pytest.raises(HashingError, match="绝对路径"):
            file_content_hash(Path("data/raw/x.pdf"))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(HashingError, match="不存在"):
            file_content_hash(tmp_path / "absent.pdf")

    def test_directory_raises(self, tmp_path: Path) -> None:
        # 目录不是文件，应报「不存在」而不是拿它去 open
        with pytest.raises(HashingError, match="不存在"):
            file_content_hash(tmp_path)
