# PDF 文本提取模块
# 用于从用户上传的 PDF 简历中提取文本内容
# 支持：文件路径（字符串）和文件对象（如 Streamlit 上传的 UploadedFile）

import pdfplumber


def extract_text_from_pdf(pdf_source):
    """
    从 PDF 提取文本内容

    参数：
        pdf_source: PDF 文件路径（字符串）或文件对象（如 Streamlit 上传的文件）

    返回：
        成功时返回提取的文本字符串（多页用换行符连接）
        失败时返回 "PDF读取失败：xxx" 便于上层判断
    """
    try:
        with pdfplumber.open(pdf_source) as pdf:
            text_parts = []
            for page in pdf.pages:
                # extract_text() 在空白页可能返回 None，需兜底
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts).strip()
    except Exception as e:
        return f"PDF读取失败：{e}"


def is_valid_resume_text(text):
    """
    检查 PDF 提取出的文本是否为有效简历内容

    判断标准：
    1. 不为空
    2. 不是错误信息
    3. 长度足够（至少 30 字，避免空白或纯图片 PDF）
    """
    if not text:
        return False
    if text.startswith("PDF读取失败"):
        return False
    if len(text.strip()) < 30:
        return False
    return True


if __name__ == "__main__":
    # 模块自测：直接运行时可测试某个 PDF 文件
    import sys

    # 支持两种方式：命令行参数 或 交互式输入
    if len(sys.argv) >= 2:
        pdf_path = sys.argv[1]
    else:
        print("用法：python pdf_reader.py <PDF文件路径>")
        print(r"示例：python pdf_reader.py C:\Users\xxx\resume.pdf")
        print("（也可以直接回车后手动输入路径）\n")
        pdf_path = input("请输入 PDF 文件路径：").strip()
        # 去除拖拽/粘贴时可能带的引号
        pdf_path = pdf_path.strip('"').strip("'")
        if not pdf_path:
            print("路径不能为空！")
            sys.exit(1)

    print(f"正在读取：{pdf_path}\n")
    result = extract_text_from_pdf(pdf_path)

    if is_valid_resume_text(result):
        print(f"✅ 提取成功，共 {len(result)} 字\n")
        print("=" * 50)
        print(result[:500])  # 只打印前 500 字预览
        if len(result) > 500:
            print(f"\n...（共 {len(result)} 字，仅显示前 500 字）")
    else:
        print(f"❌ {result}")
