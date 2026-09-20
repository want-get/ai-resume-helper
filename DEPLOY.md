# 上线部署指南（Streamlit Community Cloud）

本文档说明如何把网页版（`app.py`）部署到 Streamlit Community Cloud，让朋友通过公网链接访问。

## 一、部署前准备

### 1. 安装依赖

`requirements.txt` 已包含 `openai`、`pdfplumber`、`streamlit`：

```bash
pip install -r requirements.txt
```

### 2. 初始化 Git 仓库并推送到 GitHub

Streamlit Community Cloud 通过 GitHub 仓库部署，需要先把项目推上去：

```bash
git init
git add .
git commit -m "init"
# 在 GitHub 网页上新建一个仓库，然后：
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

> 注意：`.gitignore` 已配置好，`requirements.txt` 会被正常提交，而 `.env`、`*.txt` 结果文件、`.streamlit/secrets.toml` 不会被提交。

### 3. 配置 API Key（Secrets）

API Key **不要写进代码，也不要提交到 Git**。

- **本地开发**：复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml`，填入你的密钥。
- **云端部署**：在 Streamlit Cloud 的 Secrets 面板填入：

```toml
DEEPSEEK_API_KEY = "sk-你的密钥"
```

## 二、部署到 Streamlit Community Cloud

1. 打开 <https://share.streamlit.io> 并用 GitHub 账号登录。
2. 点击「New app」。
3. 选择你的仓库、分支（`main`）和主文件路径 `app.py`。
4. 在「Advanced settings → Secrets」中填入上面的 `DEEPSEEK_API_KEY`。
5. 点击「Deploy」，等待几分钟即可得到公网链接。

之后每次 `git push` 都会自动重新部署。

## 三、注意事项

- **链接是公开的**：免费版没有登录/密码，拿到链接的人都能访问。只把链接发给你的朋友，不要发到公开场合。
- **会休眠**：App 一段时间无人访问会进入睡眠，下一个人首次打开需要等几秒到几十秒的冷启动，属正常现象。
- **会消耗 API 额度**：朋友每次使用都会调用 DeepSeek API，费用从你的账户余额扣除，请留意余额。
- **历史记录仅当前会话有效**：网页版历史记录保存在会话内，刷新页面或换浏览器会清空（这是有意为之，避免云端多用户共写文件导致冲突）。

## 四、本地运行网页版（调试）

```bash
streamlit run app.py
```

命令行版仍然可用：

```bash
python main.py
```
