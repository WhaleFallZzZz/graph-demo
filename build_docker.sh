#!/bin/bash

# Docker构建脚本 - 适配国内网络环境

echo "🐳 开始构建知识图谱API Docker镜像..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker是否运行
if ! docker info &> /dev/null; then
    echo "❌ Docker未运行，请启动Docker服务"
    exit 1
fi

# 设置国内镜像加速器（如果已配置）
echo "🔄 检查Docker镜像加速器配置..."
DOCKER_DAEMON_CONFIG="/etc/docker/daemon.json"
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    echo "✅ 已检测到Docker配置文件: $DOCKER_DAEMON_CONFIG"
    echo "当前配置:"
    cat "$DOCKER_DAEMON_CONFIG"
else
    echo "⚠️  未检测到Docker配置文件，建议配置国内镜像加速器："
    echo "{"
    echo "  \"registry-mirrors\": ["
    echo "    \"https://registry.docker-cn.com\","
    echo "    \"https://docker.mirrors.ustc.edu.cn\","
    echo "    \"https://hub-mirror.c.163.com\","
    echo "    \"https://mirror.ccs.tencentyun.com\""
    echo "  ]"
    echo "}"
fi

# 构建镜像
echo "🏗️  开始构建镜像..."
docker build -t knowledge-graph-api .

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功！"
    echo ""
    echo "🚀 运行容器示例:"
    echo "docker run -d \\"
    echo "  --name knowledge-graph \\"
    echo "  -p 8001:8001 \\"
    echo "  -e SILICONFLOW_API_KEY=\"your-api-key\" \\"
    echo "  -e NEO4J_PASSWORD=\"neo4j-password\" \\"
    echo "  -e COS_SECRET_ID=\"cos-secret-id\" \\"
    echo "  -e COS_SECRET_KEY=\"cos-secret-key\" \\"
    echo "  knowledge-graph-api"
    echo ""
    echo "📊 查看日志:"
    echo "docker logs -f knowledge-graph"
    echo ""
    echo "🌐 测试接口:"
    echo "curl http://localhost:8001/health"
else
    echo "❌ Docker镜像构建失败！"
    echo "请检查网络连接和Docker配置"
    exit 1
fi