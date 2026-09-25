# 架构与实现说明

本文档记录 v2.0 的关键设计决策、请求链路与数据模型，作为 README 的补充。

---

## 一、分层与依赖方向

```
客户端层   app.py (Streamlit) / main.py (CLI) / client.py (HTTP 封装)
              │  只依赖 HTTP 契约，不 import 任何后端模块
              ▼
接口层     backend/api.py          路由、鉴权、并发闸门、指标中间件
              ▼
编排层     backend/services.py      KBService / ToolService / ResumeService / InterviewService
              ▼
能力层     ai_client.py      异步大模型调用（并发闸门 + 重试 + Mock）
           anti_hallucination.py  上下文编排 / 引用校验 / 可信度
           memory.py        滑动窗口 + 滚动摘要
           tools/           Function Calling 工具集
              ▼
存储层     rag/             Chroma + 进程内热索引 + BM25
           backend/db/      异步 ORM（MySQL / SQLite）
```

依赖是单向的：客户端不知道后端内部结构，能力层不知道 HTTP 的存在，
存储层不知道业务语义。这样任何一层都可以单独替换或测试。

---

## 二、关键请求链路

### 1. RAG 问答 `POST /api/v1/rag/ask`

```
1. 并发闸门 acquire（>256 则排队，15s 超时 → 503）
2. KBService.retrieve
     a. 加载（带缓存）该集合的 IDF 统计量与 BM25 索引
     b. 查询向量化（字符 n-gram 哈希 TF-IDF）
     c. 向量召回 top-20（进程内热索引，numpy 内积）
     d. BM25 召回 top-20（返回 score + matched_idf）
     e. RRF 融合 → 相关性双判据过滤 → top-5
3. build_context_block：给片段编号，生成「参考资料」区块 + SourceRef 列表
4. 无片段且 refuse_without_context → 直接拒答（confidence=0），不进模型
5. AsyncAIClient.chat（低 temperature，走 LLM 并发闸门）
6. guard_answer：正则提取引用编号 → 校验范围 → 剔除无效编号 → 算可信度
7. 返回 {content, sources, citation_report, confidence, retrieval, ...}
```

### 2. Function Calling `POST /api/v1/tools/ask`

```
1. 系统提示注入防幻觉规则 + 工具使用规则
2. 带 tools 调用模型
3. 若返回 tool_calls：
     a. 把 assistant(tool_calls) 追加进消息历史
     b. asyncio.gather 并行执行全部工具（每个带独立超时）
     c. 每个结果以 role=tool + tool_call_id 配对回传
     d. 回到第 2 步（最多 MAX_TOOL_ROUNDS 轮）
4. 输出最终回答，并原样附带工具调用链（名称/参数/结果/耗时/来源）
```

工具内部的数据源优先级：**外部实时 API → 本地数据库**。
外部 API 未配置或调用失败都会静默回退，并在返回值里标注 `data_origin`。

### 3. 多轮对话 `POST /api/v1/sessions/{id}/messages`

```
1. 从 MySQL 读取会话（含 summary / summary_upto_seq）+ 全部历史消息
2. 拼接本轮用户消息到历史
3. ConversationMemory.build：
     a. 从最新一条往回选窗口（<=8 条 && <=4000 字）
     b. 窗口外未摘要的消息累积 >=1500 字 → 调模型更新滚动摘要
     c. 组装 messages = [system] + [摘要(system)] + 窗口原文
4. 按会话模式分流：
     mock_interview → 面试官人格 system prompt（含题库参考题）
     rag_chat       → 先检索知识库拼 system prompt，可选叠加工具
5. 调用模型 → 防幻觉校验
6. 事务写入：用户消息 + 助手消息（含引用/工具调用/引用编号），
   必要时更新会话摘要与 summary_upto_seq，更新 message_count
```

---

## 三、长上下文策略细节

| 参数 | 默认 | 作用 |
|------|------|------|
| `MEMORY_WINDOW_MESSAGES` | 8 | 窗口内保留原文的消息条数上限 |
| `MEMORY_WINDOW_MAX_CHARS` | 4000 | 窗口原文的字数上限（至少保留 1 条） |
| `MEMORY_SUMMARY_TRIGGER_CHARS` | 1500 | 未摘要历史累积到多少字触发一次摘要 |
| `MEMORY_SUMMARY_MAX_CHARS` | 600 | 摘要目标长度上限 |
| `MEMORY_ENABLE_SUMMARY` | true | 关闭后只做滑动窗口不做摘要 |

**为什么两者都要？** 只做滑窗会「失忆」（早期达成的共识丢失），
只做摘要会丢细节（用户刚说的数字、项目名被压缩掉）。
窗口保精度、摘要保跨度，二者组合才能在有限上下文里维持长程一致性。

**摘要是有状态的**：`summary_upto_seq` 记录已压缩到哪一条，
下次只需要把「旧摘要 + 新增对话」喂给模型，不会每轮重算全量（成本可控）。

**降级策略**：模型不可用或 Mock 模式下，使用**抽取式摘要**（按条截断拼接），
不生成任何新内容，因此不会引入幻觉。

---

## 四、防幻觉的四层防线

| 层 | 位置 | 机制 |
|----|------|------|
| L1 检索闸门 | `rag/kb.py` | 向量相似度与关键词 IDF 信息量双判据都不过线 → 视为「知识库无内容」 |
| L2 Prompt 约束 | `prompts/system.py` | 7 条事实性硬约束；资料不足必须说明；禁止编造数字/公司/时间/薪资 |
| L3 采样参数 | `config.py` | `temperature=0.2`（摘要 0.1），业务层不擅自调高 |
| L4 输出校验 | `backend/anti_hallucination.py` | 强制引用编号 + 编号存在性校验 + 剔除无效引用 + 可信度打分 |

**可信度分数**由三部分加权：

```
confidence = 0.45 × min(1, 最高相似度 / 0.45)   # 检索质量
           + 0.35 × 引用覆盖率                   # 是否标注了引用
           + 0.20 × min(1, 来源数量 / 3)          # 证据广度
```

拒答场景 `confidence = 0`；无来源但有内容时上限压到 0.2。

> 阈值 `RAG_MIN_SCORE=0.10` / `RAG_MIN_MATCHED_IDF=11.0` 是在当前语料上校准的经验值：
> 相关提问的向量相似度通常 ≥0.13，无关闲聊 ≤0.09；
> 长问句的相似度会被稀释，因此补充 IDF 信息量判据（相关 ≥13，无关 ≤9）。
> **换语料后需要用 `scripts/check_rag.py` 重新观察分布并调整。**

---

## 五、并发与性能设计

### 为什么快

1. **全异步链路**：没有 `requests`、同步 ORM 之类的阻塞调用。
2. **两道闸门**：请求级（256）+ 模型级（64），流量尖峰时快速失败而不是拖垮全局。
3. **Chroma 之上加进程内热索引**：Chroma 的 `query` 内部串行，实测吞吐封顶 ~120 QPS。
   热索引用 numpy 做归一化向量内积（几十微秒），把混合检索吞吐提到 160~200 QPS；
   数据仍然写在 Chroma 里，热索引只是读缓存，写入时失效重建。
4. **状态文件 mtime 缓存 + TTL**：IDF 与 BM25 状态文件按 `RAG_STATE_TTL`（默认 3 秒）
   检查一次更新时间，避免了每次检索 8 次 `stat` 系统调用；外部脚本重建知识库后最多 3 秒热生效。
5. **计数缓存**：集合块数缓存，写入/删除时失效，省掉每次检索的 Chroma 往返。
6. **日志批量落库**：请求日志进 `asyncio.Queue`，后台任务每秒批量写一次，不阻塞请求。
7. **放大线程池**：`ThreadPoolExecutor(96)`，绕开 `asyncio.to_thread` 默认 20 线程的上限。

### 实测（本机单进程，真实 HTTP 协议栈）

```
并发 120 用户 × 5 轮 = 600 请求，成功率 100%，吞吐 99 req/s
延迟 P50 494ms / P90 1676ms / P95 2047ms / P99 2856ms，0 个 503
```

### 已知边界

* 纯 Python 的向量化计算受 GIL 限制，单进程吞吐有上限；多核扩展靠 `--workers N`
  （Windows 下多进程依赖命名管道，受限环境可能不可用）。
* 真实场景瓶颈是大模型上游配额，先调 `MAX_CONCURRENT_LLM`。

---

## 六、数据模型

| 表 | 用途 | 关键字段 |
|----|------|---------|
| `resumes` | 简历原文 | `content`(LONGTEXT)、`content_length` |
| `chat_sessions` | 会话 | `mode`、`summary`、`summary_upto_seq`、`message_count` |
| `chat_messages` | 消息 | `seq`、`role`、`sources_json`、`tool_calls_json`、`citations_json` |
| `job_postings` | 在招岗位（岗位搜索工具数据源） | `title`、`city`、`salary_min/max`、`skills_json` |
| `salary_records` | 薪资分位（薪资工具数据源） | `role`、`city`、`level`、`p25/p50/p75/p90`、`sample_size` |
| `kb_documents` | 知识库文档登记 | `collection`、`chunk_count`、`indexed_at` |
| `request_logs` | 请求日志（可观测性） | `path`、`status_code`、`latency_ms`、`in_flight` |

设计取舍：

* **JSON 用 Text 存储**（`*_json` 字段）而不是 MySQL JSON 类型，
  这样同一套模型能同时跑 MySQL 与 SQLite 回退，不需要两份代码。
* **Text 在 MySQL 上映射为 LONGTEXT**（`with_variant`），避免默认 TEXT 的 64KB 截断。
* **时间字段用 Python 端默认值**，不依赖数据库 `NOW()`，保证两种后端行为一致。

---

## 七、RAG 存储布局

```
data/chroma/                     Chroma 持久化目录（向量 + 文档 + 元数据，事实来源）
data/chroma/kb_state/
    ├── <collection>.idf.json    IDF 统计量（哈希桶 -> 文档频率）
    ├── <collection>.bm25.json   BM25 索引（每篇文档的词频 + 长度 + df）
    └── manifest.json            每个集合的建库信息（块数、切分参数、更新时间）
```

`<collection>.idf.json` 与 `<collection>.bm25.json` 与 Chroma 中的向量必须同源同版本，
因此**重建知识库必须走 `KnowledgeBase.build()`**（或 `scripts/build_kb.py`），
它会同时刷新三者并让内存缓存失效。手动删 Chroma 目录而不删 `kb_state` 会导致向量空间不一致。

---

## 八、扩展点

| 想做什么 | 改哪里 |
|---------|--------|
| 换成真正的语义 embedding | `rag/embeddings.py`，实现 Chroma 的 `EmbeddingFunction` 协议即可 |
| 加新工具 | `backend/tools/` 增加模块 + `register(registry)`，在 `build_default_registry` 挂载 |
| 加新岗位类型 | `prompts/` 新增模块，在 `prompts/__init__.py` 注册 |
| 换向量库 | 替换 `rag/store.py`，保持 `upsert/query/count/memory_get` 契约 |
| 加鉴权/多租户 | `backend/api.py` 的 `verify_api_key` 依赖 + `X-User-Id` 透传到仓储层 |
| 接入真实招聘/薪酬 API | `.env` 配 `JOB_API_BASE` / `SALARY_API_BASE`，工具会自动优先调用 |
