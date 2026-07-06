# QA Assistant Tool 接口文档

> 版本: v1.1  
> 最后更新: 2026-07-06  
> 角色: 平台 Router 按按钮，Tool 干活

---

## 一、Tool 标识

| 字段 | 值 |
|------|-----|
| `tool_id` | `qa_assistant` |
| `name` | 问答助手 / QA Assistant |
| `version` | 1.0.0 |
| `description` | 基于 cqaip.cn 知识库的智能问答工具，支持 QA 精确匹配与 RAG 生成式回答两种模式 |
| `capabilities` | 政策咨询、平台操作指引、模型/数据集查询、行业资讯检索、赛事活动查询、供需市场查询 |

---

## 二、调用方式

### HTTP (REST API)

```
POST /api/v1/qa/ask
Content-Type: application/json
Authorization: Bearer <token>    # 可选，平台 Casdoor SSO 透传
```

### gRPC (备选，待定)

```
service QAAssistant {
    rpc Ask(AskRequest) returns (AskResponse);
}
```

---

## 三、输入参数 (Request)

```json
{
    "query": "string (required)",
    "mode": "string (optional, default: 'auto')",
    "threshold": "float (optional, default: 0.85)",
    "top_k": "int (optional, default: 5)",
    "enable_rag": "bool (optional, default: true)",
    "user_id": "string (optional)",
    "source_filter": ["string (optional)"]
}
```

### 参数字段说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | ✅ | — | 用户提问原文，最长 500 字符 |
| `mode` | string | ❌ | `"auto"` | 回答模式：`"auto"` 自动选择, `"qa_only"` 仅 QA 匹配, `"rag_only"` 仅 RAG 生成 |
| `threshold` | float | ❌ | `0.85` | QA 匹配相似度阈值，范围 0.0~1.0 |
| `top_k` | int | ❌ | `5` | RAG 检索 chunk 数量，范围 1~15 |
| `enable_rag` | bool | ❌ | `true` | 是否允许 RAG 兜底。`false` 时 QA 未命中即返回空 |
| `user_id` | string | ❌ | `""` | 用户标识（用于日志/审计） |
| `source_filter` | string[] | ❌ | `[]` | 限定检索来源，如 `["policies", "competitions"]`。空数组表示不限 |

### mode 字段说明

| 模式 | 行为 |
|------|------|
| `auto` | 意图分析 → QA 搜索 → 相似度 ≥ threshold → 返回 QA 答案；否则走 RAG |
| `qa_only` | 只做 QA 匹配，不调用 LLM。主要用于低成本快速应答场景 |
| `rag_only` | 跳过 QA 匹配，直接检索文档 chunks → LLM 生成。用于实时性不高的深度问答 |

---

## 四、输出参数 (Response)

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "answer": "string",
        "mode": "string",
        "qa_hit": false,
        "qa_similarity": 0.0,
        "matched_question": "string",
        "source_urls": ["string"],
        "rag_chunks": [
            {
                "title": "string",
                "source": "string",
                "similarity": 0.0,
                "content_preview": "string"
            }
        ],
        "intent": {
            "type": "string",
            "confidence": 0.0,
            "label": "string"
        },
        "latency_ms": 0
    }
}
```

### 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 0=成功, 1001=参数错误, 2001=内部错误, 2002=LLM不可用 |
| `message` | string | 状态描述 |
| `data.answer` | string | 最终回答文本。QA 模式返回已有答案，RAG 模式返回 LLM 生成内容 |
| `data.mode` | string | 实际使用的模式：`"QA"` 或 `"RAG"` |
| `data.qa_hit` | bool | 是否命中 FAQ 知识库 |
| `data.qa_similarity` | float | QA 最高相似度分数（含 rerank 和 BM25 归一化） |
| `data.matched_question` | string | QA 匹配到的问题原文（仅在 qa_hit=true 时有值） |
| `data.source_urls` | string[] | 信息来源链接，最多 3 条 |
| `data.rag_chunks` | object[] | RAG 检索到的相关片段详情（仅在 RAG 模式时有值） |
| `data.intent.type` | string | 识别到的用户意图类型 |
| `data.intent.confidence` | float | 意图置信度 |
| `data.intent.label` | string | 意图中文标签 |
| `data.latency_ms` | int | 端到端耗时（毫秒） |

### 意图类型

| 类型 | 中文标签 | 触发场景 |
|------|----------|----------|
| `content_query` | 内容查询 | "平台有哪些模型"、"有什么数据" |
| `policy_faq` | 政策咨询 | "补贴条件是什么"、"怎么申请扶持" |
| `how_to` | 操作指导 | "怎么注册"、"如何充值" |
| `document_lookup` | 文件查阅 | "原文在哪"、"PDF下载" |
| `platform_info` | 平台介绍 | "这个平台是做什么的" |
| `general` | 通用 | 其他未分类问题 |

---

## 五、处理流程

```
Router 收到用户问题
    │
    ├─ 判断意图 → 是否需要 Tool?
    │      否 → 平台自己处理（导航、登录等）
    │      是 → 调用 QA Assistant Tool
    │
    ▼
QA Assistant 处理:
    │
    ├─ 1. Intent Classification（关键词 + 嵌入双通道）
    │
    ├─ 2. QA Search（稠密向量 + BM25 + RRF 融合 + 重排序）
    │      │
    │      ├─ 相似度 ≥ threshold? → 返回 QA 答案（~300ms）
    │      │
    │      └─ 相似度 < threshold? → 走 RAG
    │
    └─ 3. RAG Fallback
           │
           ├─ Chunk 检索（稠密向量 + BM25 + RRF 融合）
           ├─ LLM 生成（DeepSeek-v4-Flash, temp=0.3）
           └─ 返回生成答案 + 来源链接（~6s）
```

---

## 六、示例

### 请求示例 — QA 命中

```http
POST /api/v1/qa/ask
Content-Type: application/json

{
    "query": "重庆市算力券的补贴比例和上限是多少",
    "threshold": 0.85
}
```

**响应：**

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "answer": "算力券补贴比例为20%，上限100万元...",
        "mode": "QA",
        "qa_hit": true,
        "qa_similarity": 0.9234,
        "matched_question": "重庆市算力券的补贴比例和上限是多少",
        "source_urls": ["https://cqaip.cn/policies/..."],
        "rag_chunks": [],
        "intent": {
            "type": "policy_faq",
            "confidence": 0.85,
            "label": "政策咨询"
        },
        "latency_ms": 324
    }
}
```

### 请求示例 — RAG 生成

```http
POST /api/v1/qa/ask
Content-Type: application/json

{
    "query": "重庆618电商大促AI工具使用情况怎么样"
}
```

**响应：**

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "answer": "根据最新报道，重庆618期间...",
        "mode": "RAG",
        "qa_hit": false,
        "qa_similarity": 0.6213,
        "source_urls": [
            "https://cqaip.cn/industry-news/..."
        ],
        "rag_chunks": [
            {
                "title": "618电商大促AI工具...",
                "source": "https://cqaip.cn/industry-news/...",
                "similarity": 0.8542,
                "content_preview": "今年618期间，重庆各类AI工具使用量同比增长..."
            }
        ],
        "intent": {
            "type": "content_query",
            "confidence": 0.85,
            "label": "内容查询"
        },
        "latency_ms": 5937
    }
}
```

### 请求示例 — 限制来源

```http
POST /api/v1/qa/ask
Content-Type: application/json

{
    "query": "最近有什么比赛可以参加",
    "source_filter": ["competitions"]
}
```

---

## 七、错误处理

| code | message | 触发条件 | 处理建议 |
|------|---------|----------|----------|
| `1001` | `invalid_query` | query 为空或超长 | 提示用户输入有效问题 |
| `1002` | `invalid_mode` | mode 参数值非法 | 检查 mode 参数 |
| `1003` | `invalid_threshold` | threshold 超出 0~1 范围 | 校准参数 |
| `2001` | `internal_error` | 系统内部异常 | 重试或联系管理员 |
| `2002` | `llm_unavailable` | LLM 服务不可用 | 降级为 qa_only 模式返回结果 |
| `2003` | `qa_empty_rag_disabled` | QA 未命中且 RAG 被禁用 | 提示知识库未覆盖，建议开启 RAG |
| `2004` | `embedding_timeout` | 向量嵌入超时 | 重试 |
| `2005` | `milvus_unavailable` | 向量数据库不可用 | 检查 Milvus 连接 |

---

## 八、性能特征

| 指标 | QA 模式 | RAG 模式 |
|------|---------|----------|
| P50 延迟 | ~320ms | ~4,000ms |
| P95 延迟 | ~350ms | ~12,000ms |
| 关键瓶颈 | 向量检索 | LLM 生成 |
| 可用性 | 离线可用（依赖 Milvus） | 需 LLM API 在线 |
| 关键词覆盖率 | 80%+（FAQ 覆盖范围内） | 依文档质量而定 |
| QA 知识库规模 | 441 条 FAQ 对 | — |
| 文档知识库规模 | 3,810 个检索片段 | — |

### 资源消耗

| 组件 | 内存 | 说明 |
|------|------|------|
| BGE-M3 嵌入模型 | ~2.2GB | 加载后常驻内存 |
| BGE-Reranker-v2-m3 | ~1.1GB | 可选，QA 模式重排序用 |
| Milvus Lite | ~200MB | 向量数据库 |
| jieba 分词 + BM25 索引 | ~100MB | 关键词检索 |

---

## 九、依赖与部署

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥3.11 | 运行时 |
| pymilvus + milvus-lite | ~3.0 | 向量数据库（嵌入文件） |
| BGE-M3 | v2 | 本地嵌入模型（1024维） |
| jieba | 0.42+ | 中文分词（BM25） |
| rank-bm25 | 0.2+ | 关键词检索 |
| OpenAI SDK | 1.x | LLM API 调用（兼容 DeepSeek） |

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MILVUS_URI` | Milvus 连接路径 | `lite://./data/milvus.db` |
| `EMBEDDING_MODEL` | 嵌入模型路径 | `./models/bge-m3` |
| `CHAT_API_BASE_URL` | LLM API 地址 | `https://api.deepseek.com` |
| `CHAT_MODEL` | LLM 模型名 | `deepseek-v4-flash` |
| `QA_MATCH_THRESHOLD` | QA 匹配阈值 | `0.85` |

---

## 十、更新记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-07-06 | v1.0 | 初始版本 |
| 2026-07-06 | v1.1 | source_url 修复：所有来源使用真实前端页面 URL（policy→`/policies/{id}`, 竞赛→`/industry/competitions/{id}`, 数据集→`/datasets/{id}`, 供需→`/marketplace/{id}`, 模型→`/models/{id}`）；QA 知识库重建至 441 条；RRF 融合 BM25 归一化修复；意图分类阈值调优 |
