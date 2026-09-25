# legacy —— v2 / v1 归档

这里放的是**历史版本**的代码，保留仅供查阅与对照，**不参与 v3 构建、不被测试收集**。

v3 把项目从「Streamlit + 独立后端 + MySQL + 多用户」改成了
「单进程 + 静态单页前端 + SQLite + 单机」，所以下面这些都被替换掉了。

| 文件 | 原来是什么 | v3 里对应什么 |
|------|-----------|--------------|
| `cli_v2.py` | 命令行客户端（菜单式） | 已移除；单机版只有图形界面 |
| `client_v2.py` | 后端 HTTP 客户端（CLI 与 Streamlit 共用） | 前端改为浏览器直接调 REST，不需要客户端库 |
| `streamlit_app_v2.py` | Streamlit 网页前端 | `web/`（原生 HTML/CSS/JS 单页） |
| `test_streamlit_app_v2.py` | Streamlit 前端的 AppTest 回归测试 | `scripts/check_frontend.py`（静态一致性检查） |
| `test_api_v2.py` | v2 接口测试 | `tests/test_v3_api.py` |
| `check_backend_v2.py` | v2 后端端到端自检（含并发压测） | `scripts/check_app.py`（v3 完整流程自检） |
| `load_test_v2.py` | 并发压测脚本（120 并发） | 单机版不再需要；并发能力验证留在 v2 记录里 |
| `v2_backend/api.py` | v2 的 FastAPI 路由 | `backend/api_v3.py` |
| `v2_backend/schemas.py` | v2 的请求/响应模型 | 由 `api_v3.py` 内联处理 |
| `v2_backend/runtime_settings.py` | 在线改 Key → 写回 `.env` | `backend/llm_config.py`（写 `data/settings.json`） |
| `main.py代码详解_v1.md` | v1 命令行版逐行讲解 | 思路仍可参考，函数名已不存在 |

## 为什么当初要换掉

* **Streamlit 打包进 exe 要 250MB、启动 8~15 秒**，而且 PyInstaller 打 Streamlit 坑多；
  换成原生单页后 exe 是 125MB、启动几秒。
* **要求用户装 MySQL 不合理**，单机应用应该零依赖，改 SQLite。
* **配置文件要重启才生效**，用户无法接受；改为界面上填写、`settings.json` 持久化、热更新。
* **多用户/鉴权对单机场景是多余的复杂度**，砍掉。

如果你要从旧版找回某个功能的实现思路，这里的代码就是当时的原貌。
