# main.py 代码详解

> 本文档逐行解析 `main.py` 的每一段代码，帮助理解整个命令行版 AI 简历优化助手的实现逻辑。

---

## 一、整体结构概览

`main.py` 是整个项目的**主程序入口**，负责：
- 读取用户的 PDF 简历
- 调用 `ai_client.py` 封装的大模型 API
- 使用 `prompts.py` 中的 Prompt 模板
- 提供菜单式交互，整合 4 个功能

| 函数 | 作用 |
|---|---|
| `get_resume_from_pdf()` | 让用户输入 PDF 路径，读取并返回简历文本（所有功能共用） |
| `optimize_resume()` | 功能1：简历优化 |
| `generate_questions()` | 功能2：生成面试题 |
| `mock_interview()` | 功能3：模拟面试（多轮对话） |
| `score_resume()` | 功能4：简历评分 |
| `show_menu()` | 显示主菜单 |
| `main()` | 主循环，根据用户选择调用对应功能 |

---

## 二、导入模块（第 1-7 行）

```python
import os

from pdf_reader import extract_text_from_pdf, is_valid_resume_text
from ai_client import AIClient
from prompts import RESUME_OPTIMIZE_PROMPT, INTERVIEW_QUESTIONS_PROMPT
from prompts import MOCK_INTERVIEW_SYSTEM_PROMPT, MOCK_INTERVIEW_FOLLOWUP_PROMPT
from prompts import RESUME_SCORE_PROMPT
```

**逐行解释：**

| 行号 | 代码 | 说明 |
|---|---|---|
| 1 | `import os` | 导入 Python 标准库 `os`，用于检查文件是否存在（`os.path.exists()`） |
| 3 | `from pdf_reader import extract_text_from_pdf, is_valid_resume_text` | 从同目录的 `pdf_reader.py` 导入两个函数：`extract_text_from_pdf`（提取 PDF 文本）和 `is_valid_resume_text`（校验文本有效性） |
| 4 | `from ai_client import AIClient` | 导入 `AIClient` 类，用于调用 DeepSeek 大模型 |
| 5-7 | `from prompts import ...` | 从 `prompts.py` 导入 5 个 Prompt 模板常量，分别对应 4 个功能（其中模拟面试用了 2 个：System Prompt + Followup Prompt） |

> 💡 注意：`MOCK_INTERVIEW_FOLLOWUP_PROMPT` 实际在 `mock_interview()` 中**没有被使用**，因为模拟面试靠 System Prompt + 完整 messages 历史实现多轮对话，不需要额外拼接 followup prompt。这行导入可以删除，但不影响运行。

---

## 三、get_resume_from_pdf() 函数（第 10-50 行）

这是所有功能的**共用输入函数**，负责让用户输入 PDF 路径并提取文本。

```python
def get_resume_from_pdf():
    print("提示：请输入你的 PDF 简历文件完整路径")
    print("示例：C:\\Users\\xxx\\Desktop\\resume.pdf")

    while True:
        pdf_path = input("请输入 PDF 文件路径（输入 quit 返回菜单）：").strip()

        if pdf_path.lower() == "quit":
            return None

        if not pdf_path:
            print("文件路径不能为空！\n")
            continue

        pdf_path = pdf_path.strip('"').strip("'")

        if not os.path.exists(pdf_path):
            print(f"❌ 文件不存在：{pdf_path}\n")
            continue

        if not pdf_path.lower().endswith('.pdf'):
            print("❌ 文件必须是 PDF 格式（.pdf 后缀）！\n")
            continue

        print("\n正在读取 PDF 内容...")
        text = extract_text_from_pdf(pdf_path)

        if not is_valid_resume_text(text):
            print(f"❌ {text if text else 'PDF 内容为空'}")
            print("请确认文件是文本型 PDF（非扫描图片），且包含简历内容\n")
            continue

        print(f"✅ 成功读取简历（共 {len(text)} 字）\n")
        return text
```

**逐段解释：**

### 1. 函数开头（第 10-18 行）
```python
def get_resume_from_pdf():
    """让用户输入 PDF 简历的文件路径，读取并返回简历文本"""
    print("提示：请输入你的 PDF 简历文件完整路径")
    print("示例：C:\\Users\\xxx\\Desktop\\resume.pdf")
```
- 定义函数，无参数
- 打印提示信息，告诉用户需要输入 PDF 完整路径

### 2. 循环读取用户输入（第 20-50 行）
用 `while True` 无限循环，**直到用户输入有效路径或 quit 才退出**。

| 行号 | 代码 | 说明 |
|---|---|---|
| 21 | `pdf_path = input("...").strip()` | 读取用户输入，`.strip()` 去除首尾空格 |
| 23-24 | `if pdf_path.lower() == "quit": return None` | 输入 `quit`（不区分大小写）返回 `None`，上层函数收到 `None` 就返回菜单 |
| 26-28 | `if not pdf_path: ... continue` | 空输入，提示后重新循环 |
| 31 | `pdf_path = pdf_path.strip('"').strip("'")` | 去除路径首尾的引号（拖拽文件到终端时，Windows 会自动加上引号） |
| 33-35 | `if not os.path.exists(pdf_path): ... continue` | 用 `os.path.exists()` 检查文件是否存在 |
| 37-39 | `if not pdf_path.lower().endswith('.pdf'): ... continue` | 用 `.endswith('.pdf')` 校验后缀，`.lower()` 是为了兼容 `.PDF` 大写后缀 |
| 42 | `text = extract_text_from_pdf(pdf_path)` | 调用 `pdf_reader.py` 的函数提取文本 |
| 44-47 | `if not is_valid_resume_text(text): ... continue` | 校验文本是否有效（非空、非错误信息、≥30字）；无效则提示重新输入 |
| 49-50 | `print(...); return text` | 校验通过，打印成功信息并返回文本 |

**关键点：**
- `return None` vs `return text`：返回 `None` 表示用户想退出，返回字符串表示成功
- 所有校验失败都用 `continue` 回到循环开头，重新输入

---

## 四、optimize_resume() 函数（第 53-99 行）

**功能1：简历优化**。这是单轮对话功能的典型代表，`generate_questions()` 和 `score_resume()` 的结构几乎完全一样。

```python
def optimize_resume():
    print("\n" + "=" * 50)
    print("          AI 简历优化")
    print("=" * 50)

    resume_text = get_resume_from_pdf()

    if resume_text is None:
        return

    print("\n" + "-" * 50)
    print("AI 正在分析和优化你的简历，请稍候...")
    print("-" * 50 + "\n")

    try:
        client = AIClient()
        prompt = RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)
        result = client.chat(prompt)

        if result.startswith("AI调用失败"):
            print(f"❌ {result}")
            return

        print("=" * 50)
        print("          优化结果")
        print("=" * 50)
        print()
        print(result)
        print()

        save = input("是否保存优化结果到文件？(y/n): ").strip().lower()
        if save == "y":
            filename = input("请输入文件名（默认：optimized_resume.txt）: ").strip()
            if not filename:
                filename = "optimized_resume.txt"
            if not filename.endswith(".txt"):
                filename += ".txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"✅ 已保存到 {filename}")
            except Exception as e:
                print(f"❌ 保存失败：{e}")

    except ValueError as e:
        print(f"❌ {e}")
        print("提示：请设置 DEEPSEEK_API_KEY 环境变量")
```

**逐段解释：**

### 1. 打印标题（第 54-56 行）
```python
print("\n" + "=" * 50)
print("          AI 简历优化")
print("=" * 50)
```
- `"=" * 50`：把 `=` 重复 50 次，形成分隔线
- `\n`：换行

### 2. 获取简历文本（第 58-61 行）
```python
resume_text = get_resume_from_pdf()
if resume_text is None:
    return
```
- 调用共用函数读取简历
- 如果返回 `None`（用户输入了 quit），直接返回菜单

### 3. 调用 AI（第 67-74 行）
```python
try:
    client = AIClient()
    prompt = RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)
    result = client.chat(prompt)

    if result.startswith("AI调用失败"):
        print(f"❌ {result}")
        return
```

| 代码 | 说明 |
|---|---|
| `client = AIClient()` | 创建 AI 客户端实例（内部会读取 `DEEPSEEK_API_KEY` 环境变量） |
| `RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)` | 用 `.format()` 把简历文本填入 Prompt 模板的 `{resume_text}` 占位符 |
| `result = client.chat(prompt)` | 调用单轮对话方法，传入填充后的 Prompt |
| `if result.startswith("AI调用失败")` | `ai_client.py` 中调用失败会返回以 "AI调用失败" 开头的字符串，这里做判断 |

### 4. 打印结果（第 76-81 行）
```python
print("=" * 50)
print("          优化结果")
print("=" * 50)
print()
print(result)
print()
```
- 打印分隔线 + 标题 + AI 返回的结果

### 5. 保存到文件（第 83-95 行）
```python
save = input("是否保存优化结果到文件？(y/n): ").strip().lower()
if save == "y":
    filename = input("请输入文件名（默认：optimized_resume.txt）: ").strip()
    if not filename:
        filename = "optimized_resume.txt"
    if not filename.endswith(".txt"):
        filename += ".txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"✅ 已保存到 {filename}")
    except Exception as e:
        print(f"❌ 保存失败：{e}")
```

| 代码 | 说明 |
|---|---|
| `.strip().lower()` | 去除空格并转小写，兼容用户输入 `Y`、`Yes` 等 |
| `if not filename: filename = "optimized_resume.txt"` | 用户直接回车，使用默认文件名 |
| `if not filename.endswith(".txt"): filename += ".txt"` | 自动补 `.txt` 后缀 |
| `with open(filename, "w", encoding="utf-8") as f:` | 以写入模式打开文件，指定 UTF-8 编码（避免中文乱码） |
| `f.write(result)` | 写入内容 |

### 6. 异常处理（第 97-99 行）
```python
except ValueError as e:
    print(f"❌ {e}")
    print("提示：请设置 DEEPSEEK_API_KEY 环境变量")
```
- `AIClient()` 初始化时如果没设置 `DEEPSEEK_API_KEY`，会抛出 `ValueError`
- 捕获后提示用户设置环境变量

---

## 五、generate_questions() 函数（第 102-148 行）

**功能2：生成面试题**。结构和 `optimize_resume()` **几乎完全一样**，只有 3 处不同：

| 不同点 | optimize_resume | generate_questions |
|---|---|---|
| 标题 | "AI 简历优化" | "AI 面试题生成器" |
| Prompt 模板 | `RESUME_OPTIMIZE_PROMPT` | `INTERVIEW_QUESTIONS_PROMPT` |
| 默认文件名 | `optimized_resume.txt` | `interview_questions.txt` |

其余逻辑（读取简历、调用 AI、打印结果、保存文件、异常处理）完全相同。

> 💡 这种"结构相同、细节不同"的设计是模块化的体现：把共用的输入逻辑抽成 `get_resume_from_pdf()`，每个功能只关注自己的 Prompt 和展示。

---

## 六、mock_interview() 函数（第 151-220 行）

**功能3：模拟面试**。这是**最复杂**的功能，因为它是**多轮对话**，需要维护对话历史。

```python
def mock_interview():
    print("\n" + "=" * 50)
    print("       AI 模拟面试")
    print("=" * 50)
    print("输入 quit 可随时退出面试\n")

    resume_text = get_resume_from_pdf()

    if resume_text is None:
        return

    print("\n" + "-" * 50)
    print("面试官正在准备问题，请稍候...")
    print("-" * 50 + "\n")

    try:
        client = AIClient()

        system_prompt = MOCK_INTERVIEW_SYSTEM_PROMPT.format(resume_text=resume_text)

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        first_question = client.chat_with_history(messages)

        if first_question.startswith("AI调用失败"):
            print(f"❌ {first_question}")
            return

        messages.append({"role": "assistant", "content": first_question})

        question_count = 1
        print(f"【面试官】（第{question_count}题）")
        print(first_question)
        print()

        while True:
            user_answer = input("【你】：").strip()

            if user_answer.lower() == "quit":
                print("\n面试结束，祝你求职顺利！")
                break

            if not user_answer:
                print("请输入你的回答！\n")
                continue

            print("\n" + "-" * 50)
            print("面试官正在思考...")
            print("-" * 50 + "\n")

            messages.append({"role": "user", "content": user_answer})

            response = client.chat_with_history(messages)

            if response.startswith("AI调用失败"):
                print(f"❌ {response}")
                break

            messages.append({"role": "assistant", "content": response})

            question_count += 1
            print(f"【面试官】（第{question_count}题）")
            print(response)
            print()

    except ValueError as e:
        print(f"❌ {e}")
        print("提示：请设置 DEEPSEEK_API_KEY 环境变量")
```

### 核心概念：messages 列表

多轮对话的关键是 `messages` 列表，它记录了**完整的对话历史**。每条消息是一个字典：

```python
{"role": "system", "content": "..."}      # 系统设定（面试规则）
{"role": "user", "content": "..."}        # 用户（候选人）的回答
{"role": "assistant", "content": "..."}   # AI（面试官）的问题
```

**对话流程：**

```
第1轮：messages = [system]
       → 调用 chat_with_history → AI 提出第1个问题
       → messages = [system, assistant(问题1)]

第2轮：用户输入回答
       → messages = [system, assistant(问题1), user(回答1)]
       → 调用 chat_with_history → AI 评价并提出第2个问题
       → messages = [system, assistant(问题1), user(回答1), assistant(问题2)]

第3轮：...（以此类推）
```

### 逐段解释：

#### 1. 准备 System Prompt（第 169-173 行）
```python
system_prompt = MOCK_INTERVIEW_SYSTEM_PROMPT.format(resume_text=resume_text)
messages = [
    {"role": "system", "content": system_prompt}
]
```
- 把简历内容填入模拟面试的 System Prompt（里面设定了面试规则）
- 初始化 `messages` 列表，第一条是 system 消息

#### 2. 让 AI 提出第一个问题（第 175-186 行）
```python
first_question = client.chat_with_history(messages)

if first_question.startswith("AI调用失败"):
    print(f"❌ {first_question}")
    return

messages.append({"role": "assistant", "content": first_question})

question_count = 1
print(f"【面试官】（第{question_count}题）")
print(first_question)
print()
```
- 调用 `chat_with_history()` 并传入 `messages`，AI 根据 system prompt 提出第一个问题
- 把 AI 的回复追加到 `messages`（`role` 为 `assistant`）
- 打印第一个问题

#### 3. 对话循环（第 188-216 行）
```python
while True:
    user_answer = input("【你】：").strip()

    if user_answer.lower() == "quit":
        print("\n面试结束，祝你求职顺利！")
        break

    if not user_answer:
        print("请输入你的回答！\n")
        continue

    messages.append({"role": "user", "content": user_answer})

    response = client.chat_with_history(messages)

    if response.startswith("AI调用失败"):
        print(f"❌ {response}")
        break

    messages.append({"role": "assistant", "content": response})

    question_count += 1
    print(f"【面试官】（第{question_count}题）")
    print(response)
    print()
```

| 步骤 | 代码 | 说明 |
|---|---|---|
| 1 | `user_answer = input("【你】：").strip()` | 读取用户回答 |
| 2 | `if user_answer.lower() == "quit": break` | 输入 quit 退出面试 |
| 3 | `messages.append({"role": "user", "content": user_answer})` | 把用户回答加入历史 |
| 4 | `response = client.chat_with_history(messages)` | 带着**完整历史**调用 AI，AI 能记住之前的对话 |
| 5 | `messages.append({"role": "assistant", "content": response})` | 把 AI 回复加入历史 |
| 6 | 打印 AI 的问题 | 题号 +1 |

**关键点：**
- 每次调用 `chat_with_history()` 都传入**完整的 messages 列表**，这样 AI 才能"记住"之前的对话
- 每轮对话后都要把 AI 回复 `append` 进 `messages`，否则下一轮 AI 会"失忆"
- `break` 用于退出 while 循环（quit 或调用失败时）

---

## 七、score_resume() 函数（第 223-255 行）

**功能4：简历评分**。结构和 `optimize_resume()`、`generate_questions()` 完全一样，只有 3 处不同：

| 不同点 | 值 |
|---|---|
| 标题 | "AI 简历评分" |
| Prompt 模板 | `RESUME_SCORE_PROMPT` |
| 是否有保存功能 | ❌ 没有（评分结果不提供保存） |

其余逻辑完全相同。

---

## 八、show_menu() 函数（第 258-268 行）

```python
def show_menu():
    print("\n" + "=" * 50)
    print("    AI 简历优化助手 v1.0")
    print("    （基于 DeepSeek 大模型）")
    print("=" * 50)
    print("  1. 简历优化")
    print("  2. 生成面试题")
    print("  3. 模拟面试")
    print("  4. 简历评分")
    print("  0. 退出")
    print("=" * 50)
```

简单的菜单显示函数，每次循环都会调用。

---

## 九、main() 函数（第 271-292 行）

**程序主循环**，负责菜单交互和功能分发。

```python
def main():
    print("欢迎使用 AI 简历优化助手！")

    while True:
        show_menu()
        choice = input("请选择功能（0-4）：").strip()

        if choice == "1":
            optimize_resume()
        elif choice == "2":
            generate_questions()
        elif choice == "3":
            mock_interview()
        elif choice == "4":
            score_resume()
        elif choice == "0":
            print("\n再见！祝你求职顺利！")
            break
        else:
            print("\n无效选择，请输入 0-4 之间的数字")

        input("\n按回车键返回菜单...")
```

**逐段解释：**

| 行号 | 代码 | 说明 |
|---|---|---|
| 272 | `print("欢迎使用 AI 简历优化助手！")` | 程序启动时的欢迎语 |
| 274 | `while True:` | 无限循环，直到用户选择 0 退出 |
| 275 | `show_menu()` | 显示菜单 |
| 276 | `choice = input("请选择功能（0-4）：").strip()` | 读取用户选择 |
| 278-285 | `if/elif` 分支 | 根据选择调用对应功能函数 |
| 286-288 | `elif choice == "0": ... break` | 选择 0 时打印再见并 `break` 退出循环 |
| 289-290 | `else: print(...)` | 输入其他内容时提示无效 |
| 292 | `input("\n按回车键返回菜单...")` | 功能执行完后暂停，等用户按回车再回到菜单顶部 |

**关键点：**
- `input("\n按回车键返回菜单...")` 的作用是**暂停**，让用户看完结果再返回菜单，否则菜单会立刻刷出来
- 这行在 `choice == "0"` 时不会执行（因为 break 了）

---

## 十、程序入口（第 295-296 行）

```python
if __name__ == "__main__":
    main()
```

- `__name__` 是 Python 的特殊变量
- 当直接运行 `python main.py` 时，`__name__` 的值是 `"__main__"`
- 当 `main.py` 被其他文件 `import` 时，`__name__` 的值是 `"main"`（模块名）
- 所以这段代码的意思是：**只有直接运行 main.py 时才执行 main()**，被导入时不执行

---

## 十一、总结：四个功能的异同

| 功能 | 对话类型 | Prompt 模板 | 保存功能 | 特殊点 |
|---|---|---|---|---|
| 简历优化 | 单轮 | `RESUME_OPTIMIZE_PROMPT` | ✅ | 基础模板 |
| 生成面试题 | 单轮 | `INTERVIEW_QUESTIONS_PROMPT` | ✅ | 同上，换 Prompt |
| 模拟面试 | 多轮 | `MOCK_INTERVIEW_SYSTEM_PROMPT` | ❌ | 用 `messages` 维护历史，`chat_with_history()` |
| 简历评分 | 单轮 | `RESUME_SCORE_PROMPT` | ❌ | 同上，换 Prompt |

**单轮 vs 多轮的区别：**
- **单轮**：每次调用 `client.chat(prompt)`，只传一个 Prompt，AI 不需要记住上下文
- **多轮**：每次调用 `client.chat_with_history(messages)`，传入完整对话历史，AI 能记住之前说了什么
