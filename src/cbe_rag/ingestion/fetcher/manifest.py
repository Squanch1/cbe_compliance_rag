"""语料清单的读取与校验。

清单是手工维护的 CSV（见 data/README.md），一行一个文档。
本模块负责把它读成结构化数据，并在取值有问题时报出**行号**——
手工表格最怕的就是「有一处不对但不知道在哪一行」。

本模块只做读取与校验，不检查文件是否存在、不生成 doc_id，
那些在 resolve.py 里做。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# 清单的完整列。顺序无关，按列名取值。
MANIFEST_COLUMNS: tuple[str, ...] = (
    "file_name",
    "title",
    "source_url",
    "publisher",
    "country",
    "doc_type",
    "effective_date",
    "platform",
    "notes",
)

# notes 标注为「导入工具不读」，因此不强制它在表头里存在
OPTIONAL_COLUMNS: tuple[str, ...] = ("notes",)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    name for name in MANIFEST_COLUMNS if name not in OPTIONAL_COLUMNS
)

# 取值白名单。
# 首期硬编码在这里；docs/spec/03-data-model.md 里的 dim_* 维度表建好后，
# 改为从库读取，那时新增取值只需插入一行数据。
ALLOWED_COUNTRIES: frozenset[str] = frozenset(
    {"EU", "DE", "FR", "IT", "ES", "NL", "PL"}
)
# 不含 faq：文档类型判的是材料的权威性与来源性质，不是内容的组织形式。
# 问答体另有一套（FAQ 直出），不进这个维度。
ALLOWED_DOC_TYPES: frozenset[str] = frozenset(
    {"guideline", "regulation", "policy"}
)
ALLOWED_PUBLISHERS: frozenset[str] = frozenset({"amazon", "eu_commission"})
ALLOWED_PLATFORMS: frozenset[str] = frozenset({"amazon"})

_DATE_FORMAT = "YYYY-MM-DD"


class ManifestError(Exception):
    """清单内容有问题。消息里带行号与列名，便于定位。"""


@dataclass(frozen=True)
class ManifestRow:
    """清单里的一行，已通过取值校验。

    取值允许为空，表示「尚未填写」——门禁据此决定该文档能否
    进入向量库（见 docs/spec/02-architecture.md 第 6.2 节）。
    """

    line_number: int
    file_name: str
    title: str
    platform: str
    source_url: str | None
    publisher: str | None
    country: str | None
    doc_type: str | None
    effective_date: date | None
    notes: str | None


def _clean(raw: str | None) -> str | None:
    """去掉首尾空白；空字符串视为未填写。"""
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _check_whitelist(
    raw: str | None,
    *,
    column: str,
    allowed: frozenset[str],
    line_number: int,
) -> str | None:
    """校验取值是否在白名单内。为空时返回 None。

    报错时把可选值一并列出——只说「取值非法」对填表的人没有帮助。
    """
    text = _clean(raw)
    if text is None:
        return None
    if text not in allowed:
        raise ManifestError(
            "第 %d 行 %s：取值 %r 不在允许范围内，可选：%s"
            % (line_number, column, text, "、".join(sorted(allowed)))
        )
    return text


def _parse_effective_date(raw: str | None, *, line_number: int) -> date | None:
    """解析生效日期。为空时返回 None。

    这里不强制月份与日期补零：手工填写时 2021-7-1 很常见，
    语义无歧义，按 date 对象存下来后也不影响排序比较。
    """
    text = _clean(raw)
    if text is None:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise ManifestError(
            "第 %d 行 effective_date：日期格式应为 %s，收到 %r"
            % (line_number, _DATE_FORMAT, text)
        ) from None


def _is_blank_row(raw: dict[str, object]) -> bool:
    """判断是否为空白行。表格末尾常有多余空行。"""
    values = [value for value in raw.values() if isinstance(value, str)]
    return not any(value.strip() for value in values)


def _build_row(
    raw: dict[str, object],
    *,
    line_number: int,
    seen_file_names: dict[str, int],
) -> ManifestRow | None:
    """把一行原始数据转成 ManifestRow。空白行返回 None。"""
    if _is_blank_row(raw):
        return None

    def cell(column: str) -> str | None:
        value = raw.get(column)
        return value if isinstance(value, str) else None

    file_name = _clean(cell("file_name"))
    if file_name is None:
        raise ManifestError("第 %d 行 file_name 为空" % line_number)
    if "/" in file_name or "\\" in file_name:
        raise ManifestError(
            "第 %d 行 file_name 不能含路径分隔符，只填文件名：%r"
            % (line_number, file_name)
        )
    if file_name in seen_file_names:
        raise ManifestError(
            "第 %d 行 file_name 与第 %d 行重复：%r"
            % (line_number, seen_file_names[file_name], file_name)
        )
    seen_file_names[file_name] = line_number

    title = _clean(cell("title"))
    if title is None:
        raise ManifestError("第 %d 行 title 为空" % line_number)

    platform = _check_whitelist(
        cell("platform"),
        column="platform",
        allowed=ALLOWED_PLATFORMS,
        line_number=line_number,
    )
    if platform is None:
        raise ManifestError("第 %d 行 platform 为空" % line_number)

    return ManifestRow(
        line_number=line_number,
        file_name=file_name,
        title=title,
        platform=platform,
        source_url=_clean(cell("source_url")),
        publisher=_check_whitelist(
            cell("publisher"),
            column="publisher",
            allowed=ALLOWED_PUBLISHERS,
            line_number=line_number,
        ),
        country=_check_whitelist(
            cell("country"),
            column="country",
            allowed=ALLOWED_COUNTRIES,
            line_number=line_number,
        ),
        doc_type=_check_whitelist(
            cell("doc_type"),
            column="doc_type",
            allowed=ALLOWED_DOC_TYPES,
            line_number=line_number,
        ),
        effective_date=_parse_effective_date(
            cell("effective_date"), line_number=line_number
        ),
        notes=_clean(cell("notes")),
    )


def load_manifest(path: Path) -> list[ManifestRow]:
    """读取并校验清单文件。

    任何校验失败都抛 ManifestError，消息里带行号。
    读取时用 utf-8-sig，与 data/manifest.csv 的编码一致——
    否则 BOM 会并进第一列的列名里，导致整张表认不出 file_name。
    """
    if not path.is_file():
        raise ManifestError("清单文件不存在：%s" % path)

    rows: list[ManifestRow] = []
    seen_file_names: dict[str, int] = {}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise ManifestError("清单缺少表头：%s" % path)

        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise ManifestError(
                "清单表头缺少列：%s。模板与字段说明见 data/README.md"
                % "、".join(missing)
            )

        # 表头占第 1 行，因此数据行从第 2 行开始计数
        for line_number, raw in enumerate(reader, start=2):
            row = _build_row(
                raw, line_number=line_number, seen_file_names=seen_file_names
            )
            if row is not None:
                rows.append(row)

    return rows
