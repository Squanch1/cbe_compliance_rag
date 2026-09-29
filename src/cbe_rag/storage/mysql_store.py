"""MySQL 适配器。

业务层通过本类访问 MySQL，不直接 import pymysql 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Callable, Protocol

import pymysql

from cbe_rag.config.settings import MysqlConfig
from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel
from cbe_rag.storage.ddl import (
    COUNTRY_SEED,
    DOC_TYPE_SEED,
    MYSQL_TABLES,
    PUBLISHER_SEED,
    DocumentStatus,
)
from cbe_rag.storage.health import HealthResult
from cbe_rag.storage.records import DocumentRecord


class MysqlCursor(Protocol):
    """游标中本适配器用到的部分。"""

    def execute(self, sql: str, args: tuple[Any, ...] | None = None) -> Any:
        ...

    @property
    def rowcount(self) -> int:
        ...

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> Any:
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        ...

    def fetchall(self) -> tuple[tuple[Any, ...], ...]:
        ...

    def close(self) -> None:
        ...


class MysqlConnection(Protocol):
    """连接中本适配器用到的部分。"""

    def cursor(self) -> MysqlCursor:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


def _close_quietly(resource: Any) -> None:
    """关闭资源，忽略关闭过程中的异常。

    关闭失败不应覆盖健康检查已经得出的结论。
    """
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        pass


# 读出文档记录时的列顺序，与 _row_to_record 一一对应。
#
# **写死顺序而不用 SELECT ***：表加字段时列顺序会变，解包随即错位，
# 而且不会报错——值会被安到别的字段上，错得很安静。
_RECORD_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "content_hash",
    "status",
    "title",
    "platform",
    "source_url",
    "publisher",
    "country",
    "doc_type",
    "effective_date",
    "collected_date",
    "raw_path",
    "missing_fields",
    "parse_attempts",
)

_RECORD_COLUMNS_SQL = ", ".join(_RECORD_COLUMNS)

_SELECT_BY_HASH = (
    "SELECT " + _RECORD_COLUMNS_SQL + " FROM documents WHERE content_hash = %s"
)

_SELECT_ACTIVE_BY_SOURCE_URL = (
    "SELECT " + _RECORD_COLUMNS_SQL + " FROM documents "
    "WHERE source_url = %s AND status = %s"
)

_INSERT_DOCUMENT = (
    "INSERT INTO documents ("
    "doc_id, content_hash, status, title, platform, source_url, publisher, "
    "country, doc_type, effective_date, collected_date, raw_path, "
    "missing_fields, parse_attempts, created_at, updated_at"
    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)

# 注意不含 status：元数据变了不代表文档要重新索引，状态由索引流程决定
_UPDATE_DOCUMENT_META = (
    "UPDATE documents SET "
    "title = %s, platform = %s, source_url = %s, publisher = %s, "
    "country = %s, doc_type = %s, effective_date = %s, "
    "missing_fields = %s, updated_at = %s "
    "WHERE doc_id = %s"
)

_UPDATE_DOCUMENT_STATUS = (
    "UPDATE documents SET status = %s, updated_at = %s WHERE doc_id = %s"
)

# 带解析尝试记录的那条。两者是同一件事的两面（「这次处理结果如何」），
# 一起写省一次往返。
_UPDATE_DOCUMENT_STATUS_WITH_ATTEMPTS = (
    "UPDATE documents SET status = %s, updated_at = %s, parse_attempts = %s "
    "WHERE doc_id = %s"
)

# 同 source_url 下只留一条活跃记录，其余的收掉。
#
# 先查出来再改：返回值不只是报告用，调用方还要拿它去删 Milvus 里的向量
# ——光改状态拦不住检索，理由见 supersede_siblings。
_SELECT_SIBLING_IDS = (
    "SELECT doc_id FROM documents "
    "WHERE source_url = %s AND doc_id != %s AND status = %s"
)

_SUPERSEDE_SIBLINGS = (
    "UPDATE documents SET status = %s, updated_at = %s "
    "WHERE source_url = %s AND doc_id != %s AND status = %s"
)

_SELECT_ACTIVE_DOCUMENTS = (
    "SELECT " + _RECORD_COLUMNS_SQL + " FROM documents "
    "WHERE status = %s ORDER BY collected_date, doc_id"
)

# 读出分块时的列顺序，与 _row_to_chunk 一一对应。理由同 _RECORD_COLUMNS：
# 不用 SELECT *，表加字段后列顺序变了会静默错位。
_CHUNK_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "parent_id",
    "level",
    "chunk_index",
    "text",
    "token_count",
    "start_offset",
    "end_offset",
)

_CHUNK_COLUMNS_SQL = ", ".join(_CHUNK_COLUMNS)

# 按主键批量取。占位符个数随 id 数量变化，在方法里现拼。
_SELECT_CHUNKS_BY_IDS = (
    "SELECT " + _CHUNK_COLUMNS_SQL + " FROM chunks WHERE chunk_id IN (%s)"
)
_SELECT_DOCUMENTS_BY_IDS = (
    "SELECT " + _RECORD_COLUMNS_SQL + " FROM documents WHERE doc_id IN (%s)"
)

_DELETE_CHUNKS_BY_DOC = "DELETE FROM chunks WHERE doc_id = %s"

_INSERT_CHUNK = (
    "INSERT INTO chunks ("
    "chunk_id, doc_id, parent_id, level, chunk_index, text, token_count, "
    "start_offset, end_offset, created_at"
    ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


def _placeholders(count: int) -> str:
    """拼出 count 个 %s 占位符。

    个数由代码控制（列表长度），值本身仍走参数绑定，不拼进 SQL。
    """
    return ", ".join(["%s"] * count)


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    """把查询结果的一行转成 Chunk。"""
    values = dict(zip(_CHUNK_COLUMNS, row))
    return Chunk(
        chunk_id=values["chunk_id"],
        doc_id=values["doc_id"],
        parent_id=values["parent_id"],
        level=ChunkLevel(values["level"]),
        chunk_index=values["chunk_index"],
        text=values["text"],
        token_count=values["token_count"],
        start_offset=values["start_offset"],
        end_offset=values["end_offset"],
    )


def _missing_fields_json(record: DocumentRecord) -> str | None:
    """把缺失字段清单转成 JSON 列的取值。

    空清单存 NULL 而不是 '[]'：两者读回来都是空元组，但 NULL 更贴近
    「没有缺失」这个语义。
    """
    if not record.missing_fields:
        return None
    return json.dumps(list(record.missing_fields), ensure_ascii=False)


def _parse_attempts_json(record: DocumentRecord) -> str | None:
    """把各解析层的尝试记录转成 JSON 列的取值。空记录存 NULL 同理。"""
    if not record.parse_attempts:
        return None
    return json.dumps(list(record.parse_attempts), ensure_ascii=False)


def _row_to_record(row: tuple[Any, ...]) -> DocumentRecord:
    """把查询结果的一行转成 DocumentRecord。

    missing_fields 是 JSON 列，pymysql 不替我们解析，拿到的是 JSON 文本。
    """
    values = dict(zip(_RECORD_COLUMNS, row))
    missing = values["missing_fields"]
    attempts = values["parse_attempts"]
    return DocumentRecord(
        doc_id=values["doc_id"],
        content_hash=values["content_hash"],
        status=DocumentStatus(values["status"]),
        title=values["title"],
        platform=values["platform"],
        source_url=values["source_url"],
        publisher=values["publisher"],
        country=values["country"],
        doc_type=values["doc_type"],
        effective_date=values["effective_date"],
        collected_date=values["collected_date"],
        raw_path=values["raw_path"],
        missing_fields=tuple(json.loads(missing)) if missing else (),
        parse_attempts=tuple(json.loads(attempts)) if attempts else (),
    )


class MysqlStore:
    """MySQL 访问入口。

    构造本对象不会产生网络调用，连接在每次操作时建立并关闭。
    连接池留到性能确有需要时再引入。
    """

    def __init__(
        self,
        config: MysqlConfig,
        connect: Callable[[], MysqlConnection] | None = None,
    ) -> None:
        """初始化。

        connect 仅用于测试注入；生产路径下由配置构造真实连接。
        """
        self._config = config
        self._connect: Callable[[], MysqlConnection] = (
            connect if connect is not None else self._connect_by_config
        )

    def _connect_by_config(self) -> MysqlConnection:
        """按配置建立真实连接。"""
        timeout = max(1, int(self._config.timeout_seconds))
        return pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password.get_secret_value(),
            database=self._config.database,
            charset=self._config.charset,
            connect_timeout=timeout,
            read_timeout=timeout,
            write_timeout=timeout,
            cursorclass=pymysql.cursors.Cursor,
        )

    def health_check(self) -> HealthResult:
        """探测 MySQL 连通性，并确认连到的是预期的库。

        一次往返同时取版本号与当前库名：连错库是部署时的常见错误，
        只报版本号看不出这个问题。

        失败时返回 ok=False 的结果而不是抛异常。
        """
        started = time.perf_counter()
        connection: MysqlConnection | None = None
        cursor: MysqlCursor | None = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION(), DATABASE()")
            row = cursor.fetchone()
            if row is None:
                ok = False
                detail = "查询执行成功但未返回结果行"
            else:
                ok = True
                detail = "version=%s database=%s" % (row[0], row[1])
        except Exception as exc:
            # 这里刻意捕获所有异常。健康检查的契约是报告问题而非抛异常，
            # 而失败来源不止驱动异常——例如密码含非 ASCII 字符时，
            # pymysql 会在编码阶段抛 UnicodeEncodeError，它既不是
            # MySQLError 也不是 OSError。漏掉一种就变成调用方崩溃。
            ok = False
            detail = "%s: %s" % (type(exc).__name__, exc)
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

        return HealthResult(
            service="mysql",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """空操作。

        本适配器每次操作自行建立并关闭连接，不持有需要释放的资源。
        方法存在是为了满足 StorageAdapter 协议，让调用方能统一遍历所有适配器。
        """
        return None

    def create_schema(self) -> list[str]:
        """建表，返回本次新建的表名。

        DDL 全部带 IF NOT EXISTS，因此可重复执行。返回的是**新建**的表名
        而不是全部表名，让调用方能区分「这次真的建了」和「本来就有」。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()

            # 先查现有表，才能分辨哪些是这次新建的
            cursor.execute("SHOW TABLES")
            existing = {row[0] for row in cursor.fetchall()}

            created: list[str] = []
            for name, statement in MYSQL_TABLES:
                cursor.execute(statement)
                if name not in existing:
                    created.append(name)

            connection.commit()
            return created
        except Exception:
            # 出错时回滚再抛出。这里捕获所有异常只是为了确保回滚，
            # 异常本身照常向上传播。
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def seed_dimensions(self) -> dict[str, int]:
        """灌入三张维度表的初始数据，返回每张表影响的行数。

        用 INSERT ... ON DUPLICATE KEY UPDATE，因此可重复执行。
        **以 ddl.py 里的定义为准**：直接改库里的值会被下次执行覆盖，
        要调整维度数据应当改 ddl.py（见该模块的注释）。
        """
        statements: list[tuple[str, str, list[tuple[Any, ...]]]] = [
            (
                "dim_country",
                "INSERT INTO dim_country (code, name_zh, name_en, is_active) "
                "VALUES (%s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, is_active = new.is_active",
                [tuple(row) for row in COUNTRY_SEED],
            ),
            (
                "dim_doc_type",
                "INSERT INTO dim_doc_type (code, name_zh, name_en, is_active) "
                "VALUES (%s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, is_active = new.is_active",
                [tuple(row) for row in DOC_TYPE_SEED],
            ),
            (
                "dim_publisher",
                "INSERT INTO dim_publisher (code, name_zh, name_en, official_url, is_active) "
                "VALUES (%s, %s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, official_url = new.official_url, "
                "is_active = new.is_active",
                [tuple(row) for row in PUBLISHER_SEED],
            ),
        ]

        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            affected: dict[str, int] = {}
            for name, statement, rows in statements:
                cursor.executemany(statement, rows)
                affected[name] = len(rows)
            connection.commit()
            return affected
        except Exception:
            # 出错时回滚再抛出。这里捕获所有异常只是为了确保回滚，
            # 异常本身照常向上传播。
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def _fetch_document(
        self, sql: str, args: tuple[Any, ...]
    ) -> DocumentRecord | None:
        """执行查询并转成一条记录，没有结果行时返回 None。

        sql 只接受本模块内的常量，不接受调用方拼接——它直接进数据库。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(sql, args)
            row = cursor.fetchone()
            return None if row is None else _row_to_record(row)
        finally:
            # 只读查询，不需要提交或回滚
            _close_quietly(cursor)
            _close_quietly(connection)

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """按内容哈希查一条文档记录。

        判重的第一步。**命中不等于可以跳过**——命中那条可能没走完
        （元数据不齐、解析各层不合格、中途出错，或被新版本取代过），
        要看它的 status 才知道该怎么办，规则见 indexing/decision.py。
        """
        return self._fetch_document(_SELECT_BY_HASH, (content_hash,))

    def find_active_by_source_url(self, source_url: str) -> DocumentRecord | None:
        """按 source_url 查**活跃**（indexed）的记录。

        用来发现「同一链接下的内容变了」。已下线的记录不必查：它们
        本就不参与检索，再下线一次没有意义。
        """
        return self._fetch_document(
            _SELECT_ACTIVE_BY_SOURCE_URL,
            (source_url, DocumentStatus.INDEXED.value),
        )

    def _execute_write(self, sql: str, args: tuple[Any, ...]) -> int:
        """执行一条写语句并提交，返回受影响的行数。

        sql 只接受本模块内的常量，不接受调用方拼接——它直接进数据库。
        出错时先回滚再抛出，异常本身照常向上传播。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(sql, args)
            affected = cursor.rowcount
            connection.commit()
            return affected
        except Exception:
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def insert_document(
        self, record: DocumentRecord, *, now: datetime | None = None
    ) -> None:
        """登记一份文档。

        content_hash 上有唯一约束，重复插入会抛错。但正常判重不该走到
        这里——调用方应先按哈希查过（见 indexing/service.py）。约束是
        并发场景下的兜底，不是常规判重手段。

        now 可注入，便于测试固定时间戳。
        """
        moment = now if now is not None else datetime.now()
        self._execute_write(
            _INSERT_DOCUMENT,
            (
                record.doc_id,
                record.content_hash,
                record.status.value,
                record.title,
                record.platform,
                record.source_url,
                record.publisher,
                record.country,
                record.doc_type,
                record.effective_date,
                record.collected_date,
                record.raw_path,
                _missing_fields_json(record),
                _parse_attempts_json(record),
                moment,
                moment,
            ),
        )

    def update_document_meta(
        self, record: DocumentRecord, *, now: datetime | None = None
    ) -> None:
        """更新元数据字段，**不动状态，也不动分块**。

        用在「内容没变、清单上的登记改了」这一种情况：切块与向量只
        依赖正文，元数据改了重跑一遍纯属白干。

        注意 Milvus 里的过滤字段（country / doc_type / publisher）不在
        这里同步——那是另一处，由索引流程调 MilvusStore 完成。
        """
        moment = now if now is not None else datetime.now()
        self._execute_write(
            _UPDATE_DOCUMENT_META,
            (
                record.title,
                record.platform,
                record.source_url,
                record.publisher,
                record.country,
                record.doc_type,
                record.effective_date,
                _missing_fields_json(record),
                moment,
                record.doc_id,
            ),
        )

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        *,
        parse_attempts: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> None:
        """改一份文档的状态，可选地一并写解析尝试记录。

        **状态由处理结果决定，不由人工设置**（见 docs/spec/02-architecture.md
        6.2.1）：索引成功写 indexed，解析各层都不合格写 needs_manual，
        过程出错写 failed，出了新版本写 superseded，元数据不齐写 pending。

        parse_attempts 与状态是同一件事的两面，一起写省一次往返。
        **不传时不动那一列**：传空列表与不传是两回事，前者是「解析过但
        没有记录可写」，后者是「这次不该碰它」。
        """
        moment = now if now is not None else datetime.now()

        if parse_attempts is None:
            self._execute_write(
                _UPDATE_DOCUMENT_STATUS, (status.value, moment, doc_id)
            )
            return

        self._execute_write(
            _UPDATE_DOCUMENT_STATUS_WITH_ATTEMPTS,
            (
                status.value,
                moment,
                json.dumps(parse_attempts, ensure_ascii=False),
                doc_id,
            ),
        )

    def supersede_siblings(
        self,
        source_url: str,
        keep_doc_id: str,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """把同一链接下其他活跃记录下线，返回被下线的 doc_id 列表。

        用在索引成功之后。**同 source_url 下只允许有一条活跃记录**，
        否则旧内容会被检索到并挂进引用里——答案看着有出处，出处却是
        已经失效的旧版。

        **返回值不只是报告用，调用方要拿它去删 Milvus 里的向量。**
        仅把状态改成 superseded 拦不住检索：Milvus 里没有 status 字段，
        检索按 country / doc_type / publisher 过滤，筛不到状态，旧记录的
        向量照样会被召回。

        返回空列表是正常情况：绝大多数链接下只有一条活跃记录。

        正常情况下判重已经处理过了，这里是兜底——被取代过的文件重新
        导进来（REINDEX）时，同一链接下可能还留着另一条活跃记录。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                _SELECT_SIBLING_IDS,
                (source_url, keep_doc_id, DocumentStatus.INDEXED.value),
            )
            doc_ids = [row[0] for row in cursor.fetchall()]

            if doc_ids:
                moment = now if now is not None else datetime.now()
                cursor.execute(
                    _SUPERSEDE_SIBLINGS,
                    (
                        DocumentStatus.SUPERSEDED.value,
                        moment,
                        source_url,
                        keep_doc_id,
                        DocumentStatus.INDEXED.value,
                    ),
                )

            connection.commit()
            return doc_ids
        except Exception:
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def list_active_documents(self) -> list[DocumentRecord]:
        """所有活跃（indexed）的文档，按采集日期排序。

        用于「库里有、清单里没有」的对账。调用方只报告不动手：删除
        不可逆，而且清单改错一个字就会让文档从检索里消失。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                _SELECT_ACTIVE_DOCUMENTS, (DocumentStatus.INDEXED.value,)
            )
            return [_row_to_record(row) for row in cursor.fetchall()]
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def _fetch_many(
        self, sql: str, args: tuple[Any, ...]
    ) -> list[tuple[Any, ...]]:
        """执行查询并返回所有结果行。

        与 _fetch_document 分开：那个期望 0 到 1 行，这个期望 0 到 N 行，
        混用会让两边的语义都变模糊。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(sql, args)
            return list(cursor.fetchall())
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """按 chunk_id 批量取分块，返回顺序不保证。

        检索折叠后要拿父块全文，这里一次查一批。逐条查的话，保留 5 个
        父块就是 5 次往返，而它们本来就是同一批。
        """
        if not chunk_ids:
            return []
        rows = self._fetch_many(
            _SELECT_CHUNKS_BY_IDS % _placeholders(len(chunk_ids)),
            tuple(chunk_ids),
        )
        return [_row_to_chunk(row) for row in rows]

    def get_documents(self, doc_ids: list[str]) -> list[DocumentRecord]:
        """按 doc_id 批量取文档记录，返回顺序不保证。

        取引用元数据（标题、出处、生效日期）用，一次查一批同理。
        """
        if not doc_ids:
            return []
        rows = self._fetch_many(
            _SELECT_DOCUMENTS_BY_IDS % _placeholders(len(doc_ids)),
            tuple(doc_ids),
        )
        return [_row_to_record(row) for row in rows]

    def replace_chunks(
        self, doc_id: str, chunks: list[Chunk], *, now: datetime | None = None
    ) -> None:
        """替换一份文档的全部分块：先删旧的再插新的。

        **删和插在同一事务里**。中途失败时旧的还在，不会变成「块被删
        光了」或「一半新一半旧」——后者尤其难查，检索看不出异常，只是
        内容对不上。

        重跑同一份文档（REINDEX）必须走这里而不是直接插：主键是
        chunk_id，直接插会撞键；而块数可能变少，光靠 upsert 清不掉
        多出来的那些。
        """
        moment = now if now is not None else datetime.now()
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.parent_id,
                chunk.level.value,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.start_offset,
                chunk.end_offset,
                moment,
            )
            for chunk in chunks
        ]

        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(_DELETE_CHUNKS_BY_DOC, (doc_id,))
            if rows:
                # 空列表时 executemany 不发语句，省一次往返
                cursor.executemany(_INSERT_CHUNK, rows)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)
