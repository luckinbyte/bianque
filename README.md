# Bianque — 团队源码分析 Agent

给代码看病的工具:你提一个关于某份源码的问题,Bianque 启动一个 agent,**结合源码探索**;问题不清就**向你追问**,澄清后**结合源码给出带证据(文件:函数:行号)的答案**。整个过程**流式输出**、可**实时中断**、**绝不修改源码**(工具集严格只读)。

形态:**局域网网页 app**(FastAPI + SSE)。`/api/*` 本身也是外部 agent 可调用的 HTTP 接口,浏览器只是其首个客户端。

## 特性
- 流式输出 agent 每一步(推理增量 / 工具调用 / 工具结果)。
- 多轮澄清:agent 不清楚就停下问你,你回答后继续。
- 实时中断:Stop 按钮取消进行中的 LLM 调用。
- 强制证据:每条结论附 `文件:符号:Lxx-Lyy` + 片段。
- 多 provider(默认 OpenAI 兼容,可切 Anthropic / 本地)。
- 只读 + 路径沙箱:工具只能读 `ALLOWED_ROOTS` 内的文件,无 write/shell。
- 多用户并发会话,共享同一组代码目录。

## 快速开始
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]

cp .env.example .env   # 至少改 APP_PASSWORD 和 ALLOWED_ROOTS
set -a; . ./.env; set +a
python -m uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
```
浏览器打开 `http://<本机IP>:8000/`,在"设置"里填 **APP_PASSWORD** 和你的 **模型 provider/apikey**(存浏览器 localStorage,不进 URL),即可使用。

## 安全须知
- **HTTPS**:局域网传明文有风险(apikey/密码过网)。生产请前置 caddy/nginx 做 TLS,或配置 `TLS_CERT`/`TLS_KEY`。
- **APP_PASSWORD** 是唯一访问门槛,务必用强密码。
- **apikey** 仅在服务端内存按会话持有,绝不落盘/打日志;浏览器侧存 localStorage(可在设置里清除)。
- 工具严格只读,路径强制限制在 `ALLOWED_ROOTS`。

> 详见 `docs`/计划文件。本项目不依赖 MCP;若未来需要 MCP 接入,可在同一引擎上薄包一层。
