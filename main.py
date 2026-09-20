import os

from pdf_reader import extract_text_from_pdf, is_valid_resume_text
from ai_client import AIClient
from prompts import get_prompts, ROLE_LABELS, TECH, NON_TECH


def choose_role_type():
    """让用户选择岗位类型，返回 TECH 或 NON_TECH"""
    print("请选择岗位类型：")
    print("  1. 技术岗")
    print("  2. 非技术岗")
    while True:
        choice = input("请输入（1-2）: ").strip()
        if choice == "1":
            return TECH
        elif choice == "2":
            return NON_TECH
        print("无效选择，请输入 1 或 2")


def get_resume_from_pdf():
    """
    让用户输入 PDF 简历的文件路径，读取并返回简历文本

    会循环提示直到输入有效路径且成功提取出文本
    返回：提取出的简历文本字符串
    """
    print("提示：请输入你的 PDF 简历文件完整路径")
    print("示例：C:\\Users\\xxx\\Desktop\\resume.pdf")

    while True:
        pdf_path = input("请输入 PDF 文件路径（输入 quit 返回菜单）：").strip()

        if pdf_path.lower() == "quit":
            return None

        if not pdf_path:
            print("文件路径不能为空！\n")
            continue

        # 去除用户拖拽或粘贴路径时可能带的引号
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


def optimize_resume(role_type):
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
        prompt = get_prompts(role_type).RESUME_OPTIMIZE_PROMPT.format(resume_text=resume_text)
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


def generate_questions(role_type):
    print("\n" + "=" * 50)
    print("       AI 面试题生成器")
    print("=" * 50)

    resume_text = get_resume_from_pdf()

    if resume_text is None:
        return

    print("\n" + "-" * 50)
    print("AI 正在生成面试题，请稍候...")
    print("-" * 50 + "\n")

    try:
        client = AIClient()
        prompt = get_prompts(role_type).INTERVIEW_QUESTIONS_PROMPT.format(resume_text=resume_text)
        result = client.chat(prompt)

        if result.startswith("AI调用失败"):
            print(f"❌ {result}")
            return

        print("=" * 50)
        print("          面试题")
        print("=" * 50)
        print()
        print(result)
        print()

        save = input("是否保存面试题到文件？(y/n): ").strip().lower()
        if save == "y":
            filename = input("请输入文件名（默认：interview_questions.txt）: ").strip()
            if not filename:
                filename = "interview_questions.txt"
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


def mock_interview(role_type):
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

        system_prompt = get_prompts(role_type).MOCK_INTERVIEW_SYSTEM_PROMPT.format(resume_text=resume_text)

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


def score_resume(role_type):
    print("\n" + "=" * 50)
    print("       AI 简历评分")
    print("=" * 50)

    resume_text = get_resume_from_pdf()

    if resume_text is None:
        return

    print("\n" + "-" * 50)
    print("AI 正在评分，请稍候...")
    print("-" * 50 + "\n")

    try:
        client = AIClient()
        prompt = get_prompts(role_type).RESUME_SCORE_PROMPT.format(resume_text=resume_text)
        result = client.chat(prompt)

        if result.startswith("AI调用失败"):
            print(f"❌ {result}")
            return

        print("=" * 50)
        print("          评分结果")
        print("=" * 50)
        print()
        print(result)
        print()

    except ValueError as e:
        print(f"❌ {e}")
        print("提示：请设置 DEEPSEEK_API_KEY 环境变量")


def show_menu(role_type):
    print("\n" + "=" * 50)
    print("    AI 简历优化助手 v1.0")
    print("    （基于 DeepSeek 大模型）")
    print(f"    当前岗位：{ROLE_LABELS[role_type]}")
    print("=" * 50)
    print("  1. 简历优化")
    print("  2. 生成面试题")
    print("  3. 模拟面试")
    print("  4. 简历评分")
    print("  5. 切换岗位")
    print("  0. 退出")
    print("=" * 50)


def main():
    print("欢迎使用 AI 简历优化助手！")
    role_type = choose_role_type()

    while True:
        show_menu(role_type)
        choice = input("请选择功能（0-5）：").strip()

        if choice == "1":
            optimize_resume(role_type)
        elif choice == "2":
            generate_questions(role_type)
        elif choice == "3":
            mock_interview(role_type)
        elif choice == "4":
            score_resume(role_type)
        elif choice == "5":
            role_type = choose_role_type()
            continue
        elif choice == "0":
            print("\n再见！祝你求职顺利！")
            break
        else:
            print("\n无效选择，请输入 0-5 之间的数字")

        input("\n按回车键返回菜单...")


if __name__ == "__main__":
    main()
