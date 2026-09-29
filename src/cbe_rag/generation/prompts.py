"""提示词模板。

集中放在一处（见 CLAUDE.md 5.4）：散落在代码里的提示词没法通读，
改一句约束要翻好几个文件，也很容易在两处写出互相矛盾的要求。

材料按 `[1]` `[2]` 编号，回答里的引用也用同一套编号——编号是引用校验
的唯一接口，两边的生成规则必须一致。
"""

from __future__ import annotations

from cbe_rag.retrieval.models import RetrievedParent

# 系统提示词。六条规则分别对应项目里几条硬约束：
#   1、3  -> 不编造、依据不足时显式声明（CLAUDE.md 5.4）
#   2     -> 可追溯，每条结论挂引用（项目核心主张）
#   4     -> 多份规定冲突时以生效日期最新者为准（01-scope 3.4）
#   5     -> 未标注生效日期必须写明，不得推断（02-architecture 6.2.2）
#   6     -> 中文、简洁
SYSTEM_PROMPT = """\
你是一名跨境电商合规助手，回答卖家关于平台规则与税务合规的问题。

回答规则：

1. 只依据下面提供的材料回答。材料之外的信息，即使你知道，也不要说。
2. 每条结论后面标注依据的材料编号，形如 [1]；引用多份时写 [1][3]。
3. 材料不足以回答时，直接说明「依据不足，无法回答」，并说清楚缺什么。
   不要猜测，也不要用常识补充。
4. 多份材料对同一问题给出不同规定时，以生效日期最新的为准，并在回答里
   说明依据的是哪一版。
5. 引用的材料未标注生效日期时，必须写明「该材料未标注生效日期」，
   不得推断一个日期填上。
6. 用中文回答，简洁准确，不要复述材料原文。
"""

_NO_MATERIAL = "（没有检索到任何材料）"

# 材料的元数据行里，各字段之间的分隔符。用全角空格对中文更整齐，
# 但模型读起来和普通空格没差别，这里用普通空格更省 token。
_FIELD_SEPARATOR = "  "


def _describe(parent: RetrievedParent) -> str:
    """材料的一行元数据。

    生效日期如实写「未标注」而不是留空：留空会让模型以为这一栏被
    省略了，而它必须知道「这份材料没有日期」才能按规则 5 说明。
    """
    fields = [
        "发布机构：%s" % (parent.publisher or "未标注"),
        "适用国家：%s" % (parent.country or "未标注"),
        "文档类型：%s" % (parent.doc_type or "未标注"),
        "生效日期：%s" % (parent.effective_date or "未标注"),
    ]
    return _FIELD_SEPARATOR.join(fields)


def build_context(parents: list[RetrievedParent]) -> str:
    """把材料拼成带编号的上下文。

    编号从 1 开始且连续——回答里的 `[n]` 直接按下标映射回 parent_id，
    中间断号会让模型倾向于跳过缺失的编号，也让校验多一层没必要的分支。
    """
    if not parents:
        return _NO_MATERIAL

    blocks: list[str] = []
    for number, parent in enumerate(parents, start=1):
        blocks.append(
            "[%d] %s\n    %s\n    出处：%s\n    ---\n%s"
            % (
                number,
                parent.title,
                _describe(parent),
                parent.source_url or "未登记",
                parent.text,
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, parents: list[RetrievedParent]) -> str:
    """组装用户消息：问题在前，材料在后。

    问题放最前面：材料动辄上万 token，放在问题前面会把问题推到很远的
    位置，模型对它的注意力会弱一些。
    """
    return "问题：%s\n\n材料：\n%s" % (question, build_context(parents))


def has_unmarked_effective_date(parents: list[RetrievedParent]) -> bool:
    """材料里是否有未标注生效日期的。

    程序据此在回答里追加提示，而不是全靠模型自觉——见
    02-architecture 6.2.3：多份规定并存而部分未标日期时，必须显式
    告知用户无法判断孰新，不得默认选一份。
    """
    return any(parent.effective_date is None for parent in parents)
