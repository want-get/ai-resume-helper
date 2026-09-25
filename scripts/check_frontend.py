"""前端静态一致性自检：核对 app.js 引用的元素 id / 接口路径是否都存在。

这类拼写错误（``$("btnSource")`` vs ``id="btnSources"``）在浏览器里只会表现为
「点了没反应」，很难排查，因此用脚本提前拦住。

    python scripts/check_frontend.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import WEB_DIR  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ->  {detail}" if detail else ""))


def main() -> int:
    html_path = WEB_DIR / "index.html"
    js_path = WEB_DIR / "app.js"
    css_path = WEB_DIR / "style.css"

    print("=" * 74)
    print("[1] 静态文件是否齐全")
    print("=" * 74)
    for path in (html_path, js_path, css_path):
        check(f"{path.name} 存在且非空", path.exists() and path.stat().st_size > 200,
              f"{path.stat().st_size if path.exists() else 0} 字节")

    html = html_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")

    # ---------------- 2. JS 引用的 id 是否都在 HTML 里 ----------------
    print()
    print("=" * 74)
    print("[2] app.js 引用的元素 id 都能在 index.html 找到")
    print("=" * 74)
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    used_ids = set(re.findall(r'\$\("([^"]+)"\)', js))
    missing = sorted(used_ids - html_ids)
    check("所有 $(id) 都有对应元素", not missing,
          f"缺失 {missing}" if missing else f"共检查 {len(used_ids)} 个 id")
    unused = sorted(html_ids - used_ids)
    if unused:
        print(f"       （HTML 中未被 JS 直接引用，通常是容器元素：{len(unused)} 个）")

    # ---------------- 3. JS 调用的接口是否都真实存在 ----------------
    print()
    print("=" * 74)
    print("[3] app.js 调用的接口都能在后端找到")
    print("=" * 74)
    called = set(re.findall(r'api\("([^"?]+)', js))
    called |= set(re.findall(r'api\(\s*"([^"?]+)\?', js))

    import backend.api_v3 as api_v3_module  # noqa: E402

    declared: set[str] = set()
    for route in api_v3_module.router.routes:
        path = getattr(route, "path", "")
        declared.add(path.replace("/api/v1", "", 1))

    unknown = []
    for path in sorted(called):
        normalized = re.sub(r"\$\{[^}]+\}", "{x}", path)
        if not any(
            re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", d), normalized) for d in declared
        ):
            unknown.append(path)
    check("所有前端调用的接口都已实现", not unknown,
          f"未实现：{unknown}" if unknown else f"共检查 {len(called)} 个接口")
    print(f"       后端共暴露 {len(declared)} 个接口")

    # ---------------- 4. 关键交互约定 ----------------
    print()
    print("=" * 74)
    print("[4] 关键交互约定")
    print("=" * 74)
    check("提交回答后清空输入框", "box.value = \"\"" in js.replace("  ", " ")
          or 'box.value = ""' in js)
    check("失败时把内容还给用户", "box.value = answer" in js)
    check("428 门禁有专门提示", "428" in js and "missing_fields" in js)
    check("模型设置不回显明文 Key", 'type="password"' in html)
    check("步骤门禁：未就绪的步骤会被锁", "stepAvailable" in js and "locked" in js)
    check("厂商下拉可切换并自动填 Base URL", "applyProvider" in js and "default_model" in js)

    # ---------------- 5. 样式类名对得上 ----------------
    print()
    print("=" * 74)
    print("[5] JS 用到的关键样式类在 CSS 里有定义")
    print("=" * 74)
    css_classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))
    # 先把模板表达式 ${...} 剥掉，否则里面的内容会被误当成类名
    js_without_templates = re.sub(r"\$\{[^}]*\}", "", js)
    js_classes = set()
    for match in re.findall(r'class="([^"]+)"', js_without_templates):
        for name in match.split():
            if name:
                js_classes.add(name)
    # 这几个是运行时动态切换的状态类，定义在 CSS 里但可能没被 class=" 直接写出
    missing_css = sorted(js_classes - css_classes)
    check("JS 用到的样式类都有定义", not missing_css,
          f"未定义：{missing_css}" if missing_css else f"共检查 {len(js_classes)} 个类名")

    # ---------------- 6. 复选框不会被全局 input 规则拉满 ----------------
    print()
    print("=" * 74)
    print("[6] 表单样式细节")
    print("=" * 74)
    check("复选框已重置宽度（否则会被拉满整行）",
          "input[type=checkbox]" in css and "width: auto" in css)
    check("复选框在 HTML 中出现过", 'type="checkbox"' in html or "checkbox" in js)

    print()
    print("=" * 74)
    print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for item in FAILED:
            print(f"    - {item}")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
