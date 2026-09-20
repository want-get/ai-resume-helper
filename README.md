# AI 简历优化助手

一个基于 DeepSeek 大模型的 AI 工具，提供简历优化、面试题生成、模拟面试、简历评分四大功能，支持技术岗/非技术岗两种岗位类型，并提供命令行版（`main.py`）和网页版（`app.py`）两种使用方式。

## 功能

所有功能均支持按岗位类型（技术岗 / 非技术岗）自动调整优化策略、面试问题和评分侧重点。

### 1. 简历优化
- AI 帮你改进简历措辞，使用更专业的表达
- 量化成果，突出关键数据和业绩
- 按 STAR 法则重新组织项目经历
- 给出具体的改进建议

### 2. 生成面试题
- 根据简历内容自动生成针对性面试问题
- 涵盖专业能力、项目经验、综合素质三个维度
- 每个问题都有考察点说明

### 3. 模拟面试
- AI 扮演面试官，进行多轮对话式面试
- 每次回答后给出评价和改进建议
- 问题循序渐进，从简单到深入
- 可随时退出

### 4. 简历评分
- 从 5 个维度给简历打分（满分 100 分）
- 基本信息完整性、教育背景、工作经历、项目经验、技能证书
- 详细的评分说明和改进建议

## 技术栈

- Python 3.7+
- DeepSeek 大模型 API
- OpenAI SDK（兼容模式）
- Streamlit（网页版界面）
- pdfplumber（PDF 文本提取）

## 项目结构

```
ai_resume_helper/
├── main.py           # 命令行版主程序（菜单 + 用户交互）
├── app.py            # 网页版（Streamlit，含历史记录）
├── ai_client.py       # 大模型调用封装
├── pdf_reader.py      # PDF 文本提取模块
├── prompts/          # Prompt 模板包（按岗位类型分类）
│   ├── __init__.py   # 包入口，提供 get_prompts() 接口
│   ├── common.py      # 公用规则（面试规则、评分权重、优化要求）
│   ├── tech.py        # 技术岗模板
│   └── non_tech.py    # 非技术岗模板
├── first_streamlit_app.py  # Streamlit 入门练习 demo
├── requirements.txt   # Python 依赖
├── .gitignore         # Git 忽略文件
└── README.md          # 项目说明
```

## 安装

### 1. 克隆或下载项目

```bash
cd ai_resume_helper
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

**方式一：设置环境变量（推荐）**

Windows:
```bash
set DEEPSEEK_API_KEY=你的API密钥
```

Linux / macOS:
```bash
export DEEPSEEK_API_KEY=你的API密钥
```

**方式二：使用 .env 文件（更方便）**

1. 安装 python-dotenv：
```bash
pip install python-dotenv
```

2. 在项目根目录创建 `.env` 文件：
```
DEEPSEEK_API_KEY=你的API密钥
```

3. 在 `ai_client.py` 顶部添加：
```python
from dotenv import load_dotenv
load_dotenv()
```

> 获取 API Key：访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并获取。

## 使用

### 命令行版

```bash
python main.py
```

运行后先选择岗位类型（技术岗 / 非技术岗），再按菜单输入数字选择功能。菜单中可随时输入 `5` 切换岗位。

### 网页版

```bash
streamlit run app.py
```

在浏览器中打开后，先在左侧选择岗位类型和功能，上传 PDF 简历即可使用。

## 各功能使用说明（命令行版）

> 网页版步骤类似：在左侧选择岗位类型和功能后，上传 PDF 简历并点击对应按钮即可。

### 简历优化
1. 选择功能 1
2. 输入 PDF 简历文件的完整路径
3. 等待 AI 优化
4. 选择是否保存结果到文件

### 生成面试题
1. 选择功能 2
2. 输入 PDF 简历文件的完整路径
3. 等待生成
4. 查看面试题和考察点

### 模拟面试
1. 选择功能 3
2. 输入 PDF 简历文件的完整路径
3. AI 提出第一个问题
4. 输入你的回答
5. AI 给出评价和下一个问题
6. 输入 `quit` 退出面试

### 简历评分
1. 选择功能 4
2. 输入 PDF 简历文件的完整路径
3. 查看评分结果和改进建议

## 常见问题

### Q: 提示 "API Key 不能为空"
A: 请检查是否设置了 `DEEPSEEK_API_KEY` 环境变量。

### Q: 调用失败，显示网络错误
A: 请检查网络连接，或稍后重试。

### Q: AI 的回答不满意怎么办
A: 可以修改 `prompts/` 包中对应岗位（`tech.py` / `non_tech.py`）的 Prompt 模板，或调整 `common.py` 中的公用规则。

### Q: 可以换别的大模型吗？
A: 可以。修改 `ai_client.py` 中的 `base_url` 和 `model` 参数即可。只要是兼容 OpenAI API 格式的都可以。

## 开发说明

### 添加新功能
1. 在 `prompts/tech.py` 和 `prompts/non_tech.py` 中添加对应的 Prompt 模板
2. 在 `main.py` 和 `app.py` 中添加功能函数
3. 在 `show_menu()` 和 `main()`（以及 `app.py` 的侧边栏）中添加菜单选项

### 添加新岗位类型
1. 新建 `prompts/<新岗位>.py`，定义四个 Prompt 模板（可复用 `common.py` 的公用规则）
2. 在 `prompts/__init__.py` 的 `ROLE_LABELS` 和 `_PROMPT_MODULES` 中注册
3. 界面的岗位选择下拉框会自动出现新岗位

### 修改 Prompt
所有 Prompt 模板都在 `prompts/` 包中，按岗位类型分类，公用规则集中在 `common.py`，统一管理，方便修改。

## 许可证

MIT License
