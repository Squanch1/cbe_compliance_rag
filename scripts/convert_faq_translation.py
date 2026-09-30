"""把外部大模型翻好的 FAQ Markdown 落成 JSON。

用法：
    python scripts/convert_faq_translation.py data/faq_translated.md -o data/faq.json

格式约定见 docs/faq/translation-prompt.md。

**为什么中间用 Markdown 而不是直接要 JSON**：多段中文塞进 JSON 字符串，
引号、换行、反斜杠都得转义，模型很容易写坏，一坏就是整批重跑。分隔线格式
容错得多——多画两根横线、多敲几个空格都认。

**校验到什么程度**：只查「结构对不对、有没有明显缺斤少两」，不查翻译质量。
译文准不准、术语有没有照表，只有人能判断，脚本装作能判比不判更糟。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cbe_rag.observability import setup_console  # noqa: E402

# 条目分隔线。必须带 F-编号，这样正文里出现 `=== 注意 ===` 也不会被误认。
_ENTRY = re.compile(r"^={3,}\s*(F-\d+)\s*=*$")

# 字段分隔线。只认这四个词，别的一律当正文。
_FIELD = re.compile(r"^-{2,}\s*(问题|译文|原文|出处)\s*-*$")

_FIELDS: tuple[str, ...] = ("问题", "译文", "原文", "出处")

# 模型有时不听劝，硬套一层代码块围栏，那两行不该进正文
_FENCE = re.compile(r"^```")

_LATIN = re.compile(r"[A-Za-z]")


class TranslationError(Exception):
    """译文的 Markdown 有结构问题。消息里带行号，便于定位。"""


@dataclass(frozen=True)
class FaqTranslation:
    """一条译好的 FAQ。"""

    entry_id: str
    question_zh: str
    answer_zh: str
    answer_en: str
    source: str


def parse_translation(markdown: str) -> list[FaqTranslation]:
    """把译好的 Markdown 解析成结构化条目。

    任何一条缺字段、字段为空、编号重复，都抛 TranslationError 并带上行号——
    模型产出的长文本最怕「有一处不对但不知道在哪一行」，这也是 cases.py
    一行一道题报行号的同一个理由。

    **「原文」里必须有拉丁字母。** 这条是防模型偷懒不回填英文、或者把中文
    复制过去。原文本该是英文，一个字母都没有就说明回填出了问题。这不是
    翻译质量检查，是结构检查。
    """
    entries: list[FaqTranslation] = []
    seen: set[str] = set()

    entry_id: str | None = None
    entry_line = 0
    field: str | None = None
    buffers: dict[str, list[str]] = {}

    def flush() -> None:
        nonlocal entry_id, field, buffers
        if entry_id is None:
            return

        values: dict[str, str] = {}
        for name in _FIELDS:
            text = "\n".join(buffers.get(name, [])).strip()
            if not text:
                raise TranslationError(
                    "第 %d 行开始的条目 %s：「%s」是空的"
                    % (entry_line, entry_id, name)
                )
            values[name] = text

        if entry_id in seen:
            raise TranslationError("第 %d 行：条目 %s 出现了两次" % (entry_line, entry_id))

        if not _LATIN.search(values["原文"]):
            raise TranslationError(
                "第 %d 行开始的条目 %s：「原文」里一个拉丁字母都没有，"
                "多半是模型没把英文原样回填" % (entry_line, entry_id)
            )

        seen.add(entry_id)
        entries.append(
            FaqTranslation(
                entry_id=entry_id,
                question_zh=values["问题"],
                answer_zh=values["译文"],
                answer_en=values["原文"],
                source=values["出处"],
            )
        )
        entry_id = None
        field = None
        buffers = {}

    for number, raw in enumerate(markdown.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()

        if _FENCE.match(stripped):
            continue

        entry_match = _ENTRY.match(stripped)
        if entry_match:
            flush()
            entry_id = entry_match.group(1)
            entry_line = number
            field = None
            buffers = {}
            continue

        field_match = _FIELD.match(stripped)
        if field_match:
            if entry_id is None:
                raise TranslationError(
                    "第 %d 行出现了字段分隔线「%s」，但它不属于任何条目"
                    % (number, stripped)
                )
            field = field_match.group(1)
            buffers.setdefault(field, [])
            continue

        if field is not None:
            buffers[field].append(line)

    flush()

    if not entries:
        raise TranslationError(
            "没有解析出任何条目。确认格式是 `=== F-01 ===` 开头，"
            "字段分隔线写成 `--- 译文 ---` 这样（模板见 docs/faq/translation-prompt.md）"
        )

    return entries


def main() -> int:
    setup_console()

    parser = argparse.ArgumentParser(
        description="把译好的 FAQ Markdown 落成 JSON"
    )
    parser.add_argument("markdown", type=Path, help="外部大模型产出的 Markdown 文件")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="输出的 JSON 文件"
    )
    args = parser.parse_args()

    if not args.markdown.is_file():
        print("[FAIL] 找不到文件：%s" % args.markdown)
        return 1

    try:
        entries = parse_translation(args.markdown.read_text(encoding="utf-8"))
    except TranslationError as exc:
        print("[FAIL] %s" % exc)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(item) for item in entries], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("[OK] %d 条，写入 %s" % (len(entries), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
