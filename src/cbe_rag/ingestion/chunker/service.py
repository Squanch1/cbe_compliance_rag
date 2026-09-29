"""把解析产物组装成父子两级块。

三步串起来：数 token → 切父块 → 每个父块内切子块 → 组装成 Chunk。

**关于 token_count 的一个说明**：切分决策用的是「各块 token 数之和」
（快），而记录的 token_count 是「拼接后重新计数」（准）。两者会有
几个 token 的出入——拼接处的分词边界与单独计数时不同。记录真实值
更有意义，因为读它的人关心的是「这块能不能喂进模型」。
"""

from __future__ import annotations

from cbe_rag.ingestion.chunker.child import CHILD_TARGET_TOKENS, chunk_parent
from cbe_rag.ingestion.chunker.counter import TokenCounter
from cbe_rag.ingestion.chunker.parent import (
    PARENT_LOWER_TOKENS,
    PARENT_UPPER_TOKENS,
    split_parents,
)
from cbe_rag.ingestion.parser.schema import (
    Chunk,
    ChunkLevel,
    ParsedDocument,
    make_child_chunk_id,
    make_parent_chunk_id,
)


def chunk_document(
    document: ParsedDocument,
    counter: TokenCounter,
    *,
    parent_lower: int = PARENT_LOWER_TOKENS,
    parent_upper: int = PARENT_UPPER_TOKENS,
    child_target: int = CHILD_TARGET_TOKENS,
) -> list[Chunk]:
    """把一份解析产物切成父子两级块。

    返回顺序是逐个父块输出，每个父块紧跟它自己的子块——
    便于阅读与调试，落库时顺序无关紧要。

    父块与子块各自独立编号：父块是 p0000、p0001……，子块是 c0000、
    c0001……。子块的 parent_id 指回所属父块。
    """
    blocks = document.blocks
    block_counts = counter.count_all([block.text for block in blocks])
    parent_groups = split_parents(
        blocks, block_counts, lower=parent_lower, upper=parent_upper
    )

    chunks: list[Chunk] = []
    parent_index = 0
    child_index = 0
    cursor = 0

    for group in parent_groups:
        # 父块内各块的 token 数，按位置从全局数组里切出来
        group_counts = block_counts[cursor : cursor + len(group)]
        cursor += len(group)

        parent_id = make_parent_chunk_id(document.doc_id, parent_index)
        chunking = chunk_parent(group, group_counts, target=child_target)

        chunks.append(
            Chunk(
                chunk_id=parent_id,
                doc_id=document.doc_id,
                level=ChunkLevel.PARENT,
                chunk_index=parent_index,
                text=chunking.text,
                token_count=counter.count(chunking.text),
            )
        )

        for span in chunking.children:
            chunks.append(
                Chunk(
                    chunk_id=make_child_chunk_id(document.doc_id, child_index),
                    doc_id=document.doc_id,
                    parent_id=parent_id,
                    level=ChunkLevel.CHILD,
                    chunk_index=child_index,
                    text=span.text,
                    token_count=counter.count(span.text),
                    start_offset=span.start,
                    end_offset=span.end,
                )
            )
            child_index += 1

        parent_index += 1

    return chunks
