# ===== AI 求职助手 · 后端镜像（单机版）=====
# 说明：这是「单机版」应用，正常情况下不需要 Docker——
# 直接双击打包好的 exe 即可。这个镜像供服务器/容器环境部署时使用。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AI_HELPER_DATA_DIR=/data

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 只读资源与可写数据分离：/data 建议挂载出来，否则容器重建会丢数据
RUN mkdir -p /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code==200 else 1)"

# 容器里没有浏览器，也不该自动打开；--no-browser 关掉自动跳转
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
