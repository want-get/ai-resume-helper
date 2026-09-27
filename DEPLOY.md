# 部署与分发指南（v3 单机版）

v3 是**单机应用**：一个 exe、一个进程、SQLite 本地文件。
没有服务器、没有数据库服务、没有登录，也没有前后端分离。

---

## 一、给普通用户：发 exe 文件

**普通用户不需要看这一节，也不需要装 Python** —— 他们只需要一个 exe 文件，双击就能用。

### 1. 怎么把 exe 给别人

| 方式 | 做法 | 适合 |
|------|------|------|
| **GitHub Releases（推荐）** | 仓库 → Releases → Draft a new release → 打 tag → **把 exe 拖进附件** → Publish | 公开分发，任何人点一下就能下载 |
| 直接传文件 | 把 `dist/AI求职助手.exe` 通过网盘/IM 发给对方 | 只给少数几个人 |

> ⚠️ **不要把 exe 提交进 git 仓库。** GitHub 单文件上限 **100 MB**，
> 而这个 exe 是 **124.9 MB**，push 会被直接拒绝；就算能提交，它也会永久留在
> git 历史里，导致每次 clone 都要多下一百多兆。**Releases 附件才是放二进制产物的地方**
> （上限 2 GB，且不占 git 历史）。

> ⚠️ **附件名请用纯 ASCII**（本仓库用的是 `AI-Resume-Helper.exe`，不是本地的
> `AI求职助手.exe`）。实测 GitHub 的附件上传接口对中文文件名处理有问题：
> 用 `gh release upload` 或 `??name=中文` 上传都会**被静默截断成 `AI.exe`**，
> 而上传本身还返回成功，很容易误以为没问题。上传完**务必回仓库页面确认附件名**。

#### 推荐的上传命令

```bash
# gh：本地上传（注意先把文件复制成 ASCII 名，避免 gh 处理中文路径出错）
copy "dist\AI求职助手.exe" "%TEMP%\up_ai.exe"
gh release upload v3.0.0 "%TEMP%\up_ai.exe" --clobber
# gh 会用文件名作为附件名，所以这样得到的是 up_ai.exe
# 想要指定名字，用下面的 API 方式：
```

```bash
# API：可以显式指定附件名（推荐，名字可控）
curl -X POST \
  -H "Authorization: Bearer <你的token>" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@%TEMP%\up_ai.exe" \
  "https://uploads.github.com/repos/<owner>/<repo>/releases/<release_id>/assets?name=AI-Resume-Helper.exe"
```

> `release_id` 从 `GET /repos/<owner>/<repo>/releases/tags/<tag>` 的 `id` 字段取，
> 或直接用响应里的 `upload_url`（去掉 `{?name,label}` 后缀）。

**上传后必须验证**（只看命令返回成功是不够的）：

```bash
gh release view v3.0.0 --repo <owner>/<repo> --json assets
# 确认 name 是你要的、size 与本机文件一致、state 为 uploaded
```

### 2. exe 从哪来（这一步是开发者做的，不是普通用户做的）

```bash
pip install -r requirements.txt pyinstaller
python build_exe.py --clean
```

产物：`dist/AI求职助手.exe`（实测 **124.9 MB**，约 2 分钟）。

首次运行会在 exe 同级目录自动创建 `data/`（数据库、向量库、日志、用户配置）。

> 如果装在 `C:\Program Files` 这类不可写目录，程序会自动把数据改放到
> `%LOCALAPPDATA%\AIResumeHelper\data\`，不会因为权限问题启动失败。

**为什么只有 125MB**（同类应用常见的 300MB+）：
- 不打包 Chromium —— 用 `channel="msedge"` 驱动系统自带 Edge（Win10/11 都预装），省 150MB+
- 不打包 Streamlit —— 前端是原生 HTML/CSS/JS，由 FastAPI 直接托管
- 向量化离线（字符 n-gram 哈希 TF-IDF）—— 不需要联网下载 embedding 模型，
  这一点对打包尤其关键，否则首次运行下载失败就直接跑不起来

### 3. 分发前检查

- [ ] 在**干净的机器**（没装过 Python）上双击试过
- [ ] 确认对方机器有 Edge 或 Chrome（自定义官网渲染爬虫需要；只用内置来源则无所谓）
- [ ] 告诉对方「配置在界面右上角填，不用改任何文件」
- [ ] 告诉对方「首次使用要先在模型设置里填 API Key」
- [ ] 如果要连内网，`python main.py --host 0.0.0.0` 并自行加反向代理与鉴权

---

## 二、给开发者：源码运行

```bash
pip install -r requirements.txt
python main.py                     # 自动找空闲端口 + 打开浏览器
python main.py --port 8900 --no-browser
python server.py --reload          # 只跑后端（改代码自动重启）
```

需要改动的地方：

| 想改什么 | 改哪里 |
|---------|--------|
| 岗位来源 | 界面「岗位来源」，或数据目录的 `sources.json` |
| 抓取行为（并发/超页/详情条数） | `.env` 里的 `CRAWLER_*` |
| 大模型 | 界面「模型设置」（存 `data/settings.json`） |
| 切分与检索阈值 | `.env` 里的 `RAG_*` |
| 反问措辞/防幻觉规则 | `prompts/system.py` |

---

## 三、容器部署（可选，非单机场景）

```bash
cp .env.example .env
docker compose up -d --build
# 打开 http://localhost:8000
```

数据落在 `kb_data` 卷里。注意：**容器基础镜像不含 Edge/Chrome**，
所以容器里只能使用公开 JSON 接口类来源；要用「自定义官网渲染爬虫」，
请换成带浏览器的基础镜像（例如基于 `mcr.microsoft.com/playwright/python`）。

---

## 四、换成 MySQL（可选）

单机版默认 SQLite 是为了免除依赖。确实需要 MySQL 时：

```bash
pip install aiomysql
```

```ini
DATABASE_URL=mysql+aiomysql://root:密码@127.0.0.1:3306/ai_resume_helper?charset=utf8mb4
```

库不存在会自动创建；连不上会自动回退 SQLite，`/health` 与界面会如实标注。

---

## 五、出问题怎么排查

| 现象 | 看哪里 |
|------|--------|
| 双击 exe 没反应 | 用命令行运行 `AI求职助手.exe`，控制台会打印原因 |
| 服务起不来 | 数据目录下 `logs/app.log` |
| 抓不到岗位 | 界面「岗位来源」里逐个试抓，会显示每个来源的失败原因 |
| 大模型报错 | 界面「模型设置」→「仅验证」，会显示具体错误 |
| 界面打不开 | 控制台会打印实际端口（默认 8000 被占用时会顺延） |

调试用接口：
`GET /health`（数据库/知识库/模型状态）、`GET /api/v1/system`（数据目录与运行形态）、
`GET /docs`（完整交互式 API 文档）。
