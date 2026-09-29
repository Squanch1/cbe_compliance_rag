# 04 接口契约

本文约束 HTTP 接口的请求响应形状。**接口层只做参数校验与格式转换，
业务逻辑一律不写在这里**（见 `02-architecture.md` 第 7 节）。

---

## 1. 通用约定

### 1.1 路径与版本

所有业务接口挂在 `/api/v1/` 下。版本号进路径而不是请求头：这条项目里
只有一个消费方（自建前端），路径里的版本号一眼可见，也不必在调试工具
里额外配头。

### 1.2 trace_id

每个请求生成一个 `trace_id`（UUID），在响应里回传，同时进日志。

**为什么进响应**：使用者报「刚才那个回答有问题」时，能直接把 `trace_id`
给出来。没有它，只能靠时间戳在日志里翻。

调用方也可以自带 `trace_id`（跨服务串联时用），但格式要合规，否则服务端
重新生成——不合规的值直接拼进日志会破坏日志结构。

### 1.3 错误结构

所有非 2xx 响应共用同一个形状：

```json
{
  "trace_id": "3f2a...",
  "error": {
    "code": "uncalibrated_threshold",
    "message": "拒答阈值尚未标定，无法判断检索质量。"
  }
}
```

`code` 是稳定的机器可读标识，`message` 是给人看的说明。**调用方按 `code`
分支，不要按 `message` 匹配**——文案会改，`code` 不会。

| code | HTTP | 含义 |
|---|---|---|
| `invalid_request` | 400 | 请求体不合法（缺字段、取值越界） |
| `unknown_parent` | 404 | 按 id 取父块时找不到 |
| `uncalibrated_threshold` | 503 | 拒答阈值未标定，检索质量判不了 |
| `upstream_unavailable` | 503 | 依赖的外部服务不可用 |
| `internal_error` | 500 | 其余未预期的错误 |

`uncalibrated_threshold` 单独成一类而不是并入 `internal_error`：它不是
缺陷，是**刻意不让服务在未校准的状态下放行结果**（见 `02-architecture.md`
6.3）。运维看到这个码该去标定阈值，而不是去查 bug。

---

## 2. 问答接口

```
POST /api/v1/ask
```

### 2.1 请求

```json
{
  "question": "进口一站式服务的适用金额上限是多少",
  "session_id": "可选，多轮对话用，同一会话传同一个值",
  "filters": {
    "country": "EU",
    "doc_type": "guideline",
    "publisher": "eu_commission"
  }
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `question` | 是 | 问题正文，1 至 2000 字符 |
| `session_id` | 否 | 不传就是单轮，服务端不保留上下文 |
| `filters` | 否 | 三个维度都可选，不传即不筛 |

**过滤条件由调用方显式传，不从问题里猜。** 猜错了会直接筛掉正确文档，
而且错得不报错（见 `retrieval/search.py`）。

### 2.2 响应

```json
{
  "trace_id": "3f2a...",
  "session_id": "7c1b...",
  "answer": "进口一站式服务的适用金额上限为单票货件不超过 150 欧元 [1]。",
  "refused": false,
  "degraded": false,
  "citations": [
    {
      "parent_id": "8f3c..._p0007",
      "title": "Explanatory Notes on VAT e-commerce rules",
      "source_url": "https://...",
      "effective_date": "2021-07-01",
      "country": "EU",
      "doc_type": "guideline",
      "publisher": "eu_commission",
      "score": 0.5867,
      "matched_children": [
        {
          "chunk_id": "8f3c..._c0031",
          "text": "……consignments of an intrinsic value not exceeding EUR 150……",
          "start_offset": 1204,
          "end_offset": 1588,
          "score": 0.5867
        }
      ]
    }
  ],
  "notes": [],
  "retrieval": {
    "top_score": 0.5867,
    "parent_count": 5,
    "elapsed_ms": 412
  }
}
```

**`refused` 与 `degraded` 是两件事，不要合：**

| | 含义 | 排查方向 |
|---|---|---|
| `refused = true` | 检索质量不够，**没有调用模型** | 检索 |
| `degraded = true` | 调了模型，但回答没有可核对的出处，已替换成「依据不足」 | 提示词或模型 |

两者都为 `true` 时以 `refused` 为准（根本没走到生成那一步）。

**`answer` 永远是最终要展示的文本。** 拒答或降级时它已被替换，调用方
直接展示即可，不必再判断——少一处判断，就少一个「忘了判断」的机会。

### 2.3 引用的形状

**引用锚定到子块，不锚定到父块。** 每个 `parent_id` 下带 `matched_children`，
列出这条父块里全部命中的子块，含原文与字符偏移。

两条理由（详见 `02-architecture.md` 6.5）：

1. 模型读的是父块全文，但真正支撑某句结论的可能只是其中一段。只给父块
   级的引用，用户点开看到的是几千字，找不到那句话。
2. 分数最高的子块不一定是模型依据的那条。**只锚定最高分那条，会让用户
   点开发现原文支撑不了那句话，而且不报错。**

`start_offset` / `end_offset` 指向**父块正文里的位置**，不是全文。界面拿
到的就是父块正文，按这个偏移高亮即可。

**父块全文不随本接口返回。** 父块约 1500 token、折合三千至五千汉字，
5 条引用都带上会让响应体到两万字量级，而多数情况下用户不会展开查看。
按需拉取见下一节。

### 2.4 多轮

传了 `session_id` 时，服务端把这一问一答追加进会话上下文，供下一轮使用。
上下文有**条数上限与过期时间**（见 `config` 的 Redis 段），超出后丢弃
最早的轮次。

**会话上下文只影响提问的理解，不影响引用的来源。** 引用的每一段都必须
来自本轮的检索结果——上一轮的材料不能用在本轮的回答里，否则引用会指向
没有被检索过的内容。

---

## 3. 父块懒加载接口

```
GET /api/v1/parents/{parent_id}
```

按 `parent_id` 取父块全文，供界面「展开上下文」时调用。

### 3.1 响应

```json
{
  "parent_id": "8f3c..._p0007",
  "doc_id": "8f3c...",
  "text": "（父块全文）",
  "token_count": 1979,
  "chunk_index": 7,
  "title": "Explanatory Notes on VAT e-commerce rules",
  "source_url": "https://...",
  "effective_date": "2021-07-01",
  "collected_date": "2026-09-29",
  "country": "EU",
  "doc_type": "guideline",
  "publisher": "eu_commission"
}
```

`token_count` 一并返回：界面上「这段有多长」是使用者关心的信息，而父块
长度不固定（实测 633 至 2170）。

父块不存在时返回 `404` 与 `unknown_parent`。

**这个接口不返回子块。** 子块由问答接口的 `matched_children` 给出——
界面只有在已经知道「哪几段被命中」的前提下才会来拉父块，先有子块再有
父块，顺序不会反。

---

## 4. 健康检查接口

```
GET /api/v1/health
```

返回各适配器的连通性，形状与 `scripts/check_services.py` 的一致：

```json
{
  "ok": false,
  "services": [
    {"service": "milvus", "ok": true, "detail": "version=2.6.6", "elapsed_ms": 12.3},
    {"service": "mysql", "ok": true, "detail": "version=8.0.36 database=cbe_compliance", "elapsed_ms": 8.1},
    {"service": "redis", "ok": false, "detail": "OperationalError: ...", "elapsed_ms": 5012.0},
    {"service": "bge-m3", "ok": true, "detail": "路径=... 精度=fp16 模型=未加载（首次 encode 时加载）", "elapsed_ms": 0.2},
    {"service": "bailian", "ok": true, "detail": "base_url=... model=qwen-plus", "elapsed_ms": 0.1}
  ]
}
```

`ok` 是所有服务都通的简写。**任一服务不通时 HTTP 仍返回 200**——这个接口
的语义是「报告状态」，不是「服务本身能不能响应」。返回 503 会让监控把
「某个依赖挂了」和「这个服务挂了」混为一谈。

`bge-m3` 与 `bailian` 的健康检查都**不发起真实调用**（不加载模型、不烧
额度），因此它们的 `detail` 只说明配置齐备，不保证当时可用。

---

## 5. 不在这份契约里的东西

| | 为什么 |
|---|---|
| 流式响应（SSE） | `02-architecture.md` 2.1 提到过，但首期先做完整响应。引用的校验要在全文生成后才能做，流式下「先吐一半再发现没有引用」的体验比等几秒更糟 |
| 认证与鉴权 | 首期只有自建前端一个消费方，部署在内网 |
| 索引触发接口 | 导入走 `scripts/import_corpus.py`，是运维动作，不暴露成 HTTP |
| 评测接口 | 评测走脚本与 `eval_cases` 表 |
