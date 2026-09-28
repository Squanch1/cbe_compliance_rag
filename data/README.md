# 语料目录约定

本文件说明原始文档怎么放、元数据怎么登记。`scripts/` 下的导入工具按本约定读取。

## 目录结构

```
data/
├── README.md        本文件
├── manifest.csv     元数据清单，一行一个文档，手工维护
└── raw/             原始文档本体，不入库（版权与体积原因）
    ├── ioss-threshold.html
    ├── eu-vat-guide.pdf
    └── ...
```

**只有 `manifest.csv` 进版本库，`raw/` 下的文件不进。** 文档本体体积大且有版权问题，清单记录的是「收了哪些资料」，本身有价值。

## 工作流程

1. 把下载到的原始文件放进 `data/raw/`
2. 在 `manifest.csv` 里加一行，填上该文档的元数据
3. 运行导入工具，它会按清单读取文件、解析、入库

**可以先导入、后补齐。** 清单里的 `source_url`、`publisher`、`country`、`doc_type` 任一项为空时，该文档会停在 `pending` 状态，不会进入向量库。补齐后重新导入即可。

## 清单字段

列名不可改动，导入工具按列名读取。

| 列 | 必填 | 格式 | 说明 |
|---|---|---|---|
| `file_name` | 是 | 文件名含扩展名 | 相对 `data/raw/` 的文件名，不能带路径 |
| `title` | 是 | 文本 | 文档标题，用原文语言的标题 |
| `source_url` | 是 | 完整 URL | 原文出处。**缺失则无法进入向量库** |
| `publisher` | 是 | 取值见下表 | 发布机构 |
| `country` | 是 | 取值见下表 | 适用国家或地区 |
| `doc_type` | 是 | 取值见下表 | 文档类型 |
| `effective_date` | 否 | `YYYY-MM-DD` | 生效日期。填不出就留空，回答时会标注「未标注生效日期」 |
| `platform` | 是 | `amazon` | 首期固定填 `amazon` |
| `notes` | 否 | 文本 | 备注，导入工具不读，写给自己看 |

**采集日期（`collected_date`）不在清单里**——导入时程序自动填当天日期。这个字段只用于运维排查（判断语料有多旧），不影响检索与回答，所以不占用你的手工维护成本。

### 取值

`country`：

| 值 | 含义 |
|---|---|
| `EU` | 欧盟整体（适用于欧盟层面的法规与指南） |
| `DE` / `FR` / `IT` / `ES` / `NL` / `PL` | 具体成员国 |

`doc_type`：

| 值 | 含义 |
|---|---|
| `guideline` | 官方指南、操作说明 |
| `regulation` | 法规条文 |
| `policy` | 平台政策 |
| `faq` | 常见问题 |

`publisher`：

| 值 | 含义 |
|---|---|
| `amazon` | Amazon 官方 |
| `eu_commission` | 欧盟委员会 |

**不在以上取值内的写法会被导入工具拒绝。** 需要新增取值时改 `docs/spec/03-data-model.md` 的维度表定义并说明理由，不要直接往清单里写新值。

## 示例

一行填好的记录长这样：

```csv
file_name,title,source_url,publisher,country,doc_type,effective_date,platform,notes
ioss-threshold.html,IOSS - Import One-Stop Shop,https://europa.eu/youreurope/business/taxation/vat/one-stop-shop/index_en.htm,eu_commission,EU,guideline,2021-07-01,amazon,货值 150 欧元阈值
```

注意 `effective_date` 留空是允许的：

```csv
eu-vat-guide.pdf,VAT Guide for Small Businesses,https://.../vat-guide.pdf,eu_commission,EU,guideline,,amazon,PDF 双栏排版
```

## 注意事项

- **`file_name` 必须唯一**，不能两行指向同一个文件
- **一个文件一行**，不要在一个单元格里塞多个文件名
- **文件放进 `raw/` 之后不要再改名**，清单里记的是文件名；确实要改名时两处一起改
- **不要用 Excel 另存为其他格式**，保持 CSV；另存为 `.xlsx` 后导入工具读不了
- 用 Excel 打开时若中文乱码，说明文件被转存成非 UTF-8 了

## 为什么用清单而不是让程序自己扫目录

元数据里有两类信息程序推不出来：**`source_url`（你从哪下载的）和 `effective_date`（文档什么时候生效）**。这两项必须人工提供。

剩下的字段虽然部分能从文件名或内容里猜，但猜错比缺失更麻烦——猜错的值会静默进入检索过滤条件，用户按国家筛选时得到错误结果却看不出原因。所以统一人工登记。
