# 使用Python 3.11 slim镜像作为基础 - 使用DaoCloud镜像源加速
FROM docker.m.daocloud.io/python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置国内镜像源环境变量
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖 - 使用清华源
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY llama/ ./llama/
COPY start_server.sh ./

# 创建日志目录并赋予脚本执行权限
RUN chmod +x start_server.sh && mkdir -p /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV LOG_DIR=/app/logs

# 暴露端口8001
EXPOSE 8001

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动应用
CMD ["./start_server.sh"]