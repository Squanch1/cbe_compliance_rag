"""跨境电商合规问答的演示界面。

**这里只做界面。** 业务逻辑一律通过 HTTP 调后端（见 CLAUDE.md 第 2 节），
界面上不做检索、不拼提示词、不判拒答——那些都在后端，这里只负责把结果
摆清楚。

页面本身保持成直接脚本，不包进函数（见 Streamlit 的 code-organization
约定）：包一层不会让它更清楚，反而多一层缩进要看。

启动：
    streamlit run apps/web/app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiError, BackendClient

# 下拉里「不限制」那一项。取值是代码，显示是中文——两者不能混。
UNRESTRICTED = "不限"

# 首屏的建议问题。挑的是语料里确实有答案、且能体现「引用到具体段落」的
# 几类问法——最后一条是语料里没有的，用来看拒答长什么样。
SUGGESTIONS = [
    "进口一站式服务的适用金额上限是多少",
    "非欧盟境内的卖家可以注册 IOSS 吗",
    "电子接口在什么情况下被视为供应商",
    "德国站的增值税税率是多少",
]

FILTER_HINT = "不选即不限制。限定范围会让召回变少，但更聚焦。"

st.set_page_config(
    page_title="跨境电商合规问答",
    page_icon=":material/gavel:",
    layout="wide",
)


@st.cache_resource
def backend() -> BackendClient:
    """后端客户端。

    用 cache_resource 跨 rerun 复用：Streamlit 每次交互都会重跑整个脚本，
    每次新建客户端等于每次重配一遍连接。
    """
    return BackendClient()


@st.cache_resource
def dimensions() -> dict[str, list[dict[str, str]]]:
    """维度取值：代码配中英文名。

    **拉不到时返回空字典而不是抛错**：后端没起时整个页面都在报错，下拉
    里再堆一条一样的错误只会更乱。空字典让下拉只剩「不限」，侧栏另给
    一句说明。
    """
    try:
        return backend().dimensions()
    except ApiError:
        return {}


def build_options(items: list[dict[str, str]]) -> dict[str, str]:
    """把维度项做成「代码（中文名）-> 代码」的映射。

    **代码在前、中文名在括号里。** 代码才是后端认的值，也会出现在日志、
    引用与筛选条件里；把代码摆出来，使用者对着界面上的字样就能在别处
    找到同一个东西。中文名只做解释，替掉代码反而让人对不上号。

    提交的仍然是代码。拿中文名去提交，后端拼出来的过滤条件匹配不上。

    拆成纯函数是为了能单测：options_for 依赖 st.cache_resource，而那东西
    在 pytest 的 bare mode 下不工作。
    """
    return {
        "%s（%s）" % (item["code"], item["name_zh"]): item["code"]
        for item in items
    }


def options_for(kind: str) -> dict[str, str]:
    """取某一类维度的下拉选项。"""
    return build_options(dimensions().get(kind, []))


def init_state() -> None:
    """把会话状态集中初始化在一处。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None


def render_retrieval(payload: dict[str, Any], *, live: bool) -> None:
    """展示这次检索的概况。

    分数与召回数是刻意暴露的：使用者看到「只召回 2 段、最高分 0.31」，
    就能理解回答为什么短、为什么说依据不足。

    live 用 st.status（能显示正在进行的步骤），回放历史用 st.expander
    ——两者外观一致，语义不同。
    """
    retrieval = payload["retrieval"]
    score = retrieval["top_score"]
    summary = "召回 %d 个父块，稠密路最高余弦 %s" % (
        retrieval["parent_count"],
        "无命中" if score is None else "%.4f" % score,
    )
    detail = (
        "检索质量的判据取稠密路的最高余弦相似度，不是融合分——"
        "融合分随权重变化（同一份语料在 0.7/0.3 下给 0.6970、0.5/0.5 下给 "
        "0.6407），拿它当阈值没有可比性。"
    )

    if live:
        with st.status(summary, type="compact", state="complete"):
            st.write(detail)
    else:
        with st.expander(summary, icon=":material/search:"):
            st.write(detail)


def render_full_context(item: dict[str, Any]) -> None:
    """按需拉取父块全文。

    **不能在 expander 里直接拉。** Streamlit 会计算折叠起来的内容——藏在
    折叠里的请求照样发出去，每次 rerun 都发一遍。改成按钮触发，结果存进
    session_state，点开才有那一次请求。
    """
    parent_key = "parent::%s" % item["parent_id"]
    if st.button("展开完整上下文", key="load::%s" % item["parent_id"]):
        try:
            st.session_state[parent_key] = backend().get_parent(item["parent_id"])
        except ApiError as exc:
            st.error(str(exc))
            return

    parent = st.session_state.get(parent_key)
    if parent is None:
        return

    st.caption("完整上下文（%d token）" % parent["token_count"])
    st.markdown(parent["text"])


def render_citation(index: int, item: dict[str, Any]) -> None:
    """展示一条引用：材料信息，以及其中被命中的段落。

    **引用锚定到段落，不是整份材料。** 模型读的是父块全文，支撑某句话的
    可能只是其中一段；只给材料级的引用，使用者翻半天找不到那句话。
    """
    children = item["matched_children"]
    label = "%d. %s（命中 %d 段）" % (index, item["title"], len(children))

    with st.expander(label):
        st.caption("出处：%s" % (item["source_url"] or "（未登记）"))
        st.caption(
            "生效日期：%s" % (item["effective_date"] or "未标注（回答里应已说明）")
        )
        st.caption(
            "维度：%s / %s / %s"
            % (item["country"], item["doc_type"], item["publisher"])
        )

        for child in children:
            with st.container(border=True):
                st.caption(
                    "字符 %s - %s　相关度 %.4f"
                    % (child["start_offset"], child["end_offset"], child["score"])
                )
                st.markdown(child["text"])

        render_full_context(item)


def render_answer(payload: dict[str, Any], *, live: bool) -> None:
    """渲染一条完整回答：检索概况、正文、引用、提示。"""
    render_retrieval(payload, live=live)

    if payload["refused"]:
        st.warning(payload["answer"], icon=":material/block:")
    else:
        st.markdown(payload["answer"])
        if payload["degraded"]:
            st.info(
                "这段回答没有可核对的出处，已按「依据不足」处理。",
                icon=":material/info:",
            )

    citations = payload["citations"]
    if citations:
        st.markdown("**引用来源**")
        for index, item in enumerate(citations, start=1):
            render_citation(index, item)

    for note in payload["notes"]:
        st.caption("提示：%s" % note)


def ask(question: str, filters: dict[str, str]) -> None:
    """问一个问题，把过程与结果渲染出来并记进会话。"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("检索并生成", type="compact", expanded=True):
            st.write("正在检索语料……")
            try:
                payload = backend().ask(
                    question,
                    session_id=st.session_state.session_id,
                    filters=filters,
                )
            except ApiError as exc:
                # 用户那条消息留着：出错了也要看得见自己问的是什么，
                # 否则界面上一片空白，像是点了个没反应的东西
                st.error(str(exc))
                return

        # 后端每次都回 session_id（不传时它生成一个），存下来，
        # 后面几轮才接得上同一段会话
        st.session_state.session_id = payload["session_id"]
        st.session_state.messages.append(
            {"role": "assistant", "content": payload["answer"], "payload": payload}
        )
        render_answer(payload, live=True)


init_state()

st.title("跨境电商合规问答")
st.caption(
    "回答全部依据已收录的平台政策与官方指南，每条结论都带可追溯的原文引用。"
)

with st.sidebar:
    st.subheader("检索范围")
    st.caption(FILTER_HINT)

    countries = options_for("countries")
    doc_types = options_for("doc_types")
    publishers = options_for("publishers")

    if not countries:
        st.warning(
            "没取到可选项，暂时只能全范围检索。"
            "确认后端已启动：python scripts/serve.py",
            icon=":material/warning:",
        )

    country = st.selectbox("国家或地区", [UNRESTRICTED, *countries])
    doc_type = st.selectbox("文档类型", [UNRESTRICTED, *doc_types])
    publisher = st.selectbox("发布机构", [UNRESTRICTED, *publishers])

    # 界面显示中文名，提交给后端的是代码——代码才是它拼进检索表达式的
    # 东西，中文名只是给人看的
    active_filters: dict[str, str] = {}
    if country != UNRESTRICTED:
        active_filters["country"] = countries[country]
    if doc_type != UNRESTRICTED:
        active_filters["doc_type"] = doc_types[doc_type]
    if publisher != UNRESTRICTED:
        active_filters["publisher"] = publishers[publisher]

    st.divider()
    if st.button("清空对话", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_answer(message["payload"], live=False)
        else:
            st.markdown(message["content"])

# 建议问题只在空对话时出现，点一下就当成输入。这里直接问、不 rerun：
# rerun 会让 pills 重新渲染并保持选中，又触发一次提问。
suggestion = None
if not st.session_state.messages:
    st.markdown("#### 试试问这些")
    suggestion = st.pills("建议问题", SUGGESTIONS, label_visibility="collapsed")

prompt = st.chat_input("问一个关于平台规则或税务合规的问题") or suggestion
if prompt:
    ask(prompt, active_filters)
