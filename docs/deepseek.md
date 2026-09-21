# DeepSeek 接入说明

## 实际功能

DeepSeek 根据当前切片的结构化报告调用只读工具，再解释采样范围、质控指标和区域坐标。当前集成只传输文本，不上传原始图像；也不会凭 API Key 对远程基础模型执行训练或微调。

API 使用 `https://api.deepseek.com/chat/completions`。默认模型为 `deepseek-flash`，可在 `.env` 中改成账号可用的其他模型。模型名称应通过官方账户或 `/models` 核实。

## 本地配置与启动

1. 复制 `.env.example` 为 `.env`，填写 `LLM_API_KEY`。
2. 安装更新后的依赖：`pip install -e '.[dev]'`。
3. 重新构建前端：`npm --prefix frontend run build`。
4. 重启 `python -m uvicorn pathology.main:app --host 127.0.0.1 --port 8000`。
5. 页面上方显示「DeepSeek 已配置」，回答标注「DeepSeek · 模型名」。

Docker 默认不读取宿主机的 `.env` 文件内容到容器。要在容器中启用，可在本机 compose 服务配置中添加 `env_file: .env`，再重建服务；不要把真实密钥写进 Dockerfile、Compose 公开配置或前端代码。

环境变量优先于 `.env`。如果模型仍未切换，请先检查启动终端是否设置了旧的 `LLM_BASE_URL`、`LLM_MODEL` 或 `LLM_API_KEY`。

## 请求与回退

- 第一轮 `tool_choice=required`，必须获取当前报告证据。
- DeepSeek 请求设置 `thinking.type=disabled`；如返回服务商要求的中间字段，仅透传给该服务商。
- 后续轮次允许生成最终文本，最多四轮，每轮至多三个工具。
- 只允许 `get_quality`、`get_regions`、`get_report`，拒绝额外参数。
- 401 显示认证失败，402 显示余额不足，429 显示限流；超时和网络错误分别提示。
- 回退回答标注「本地证据助手」，不会冒充 DeepSeek 回答。
- `/api/health` 只返回服务商、模型名称及配置模式，不返回密钥。

## 本次验证

使用账号可用模型列表核实默认模型，使用合成统计指标完成真实 HTTP 工具循环，返回 `source=llm`、`provider=DeepSeek`。医院数据及训练权重仍等待另行接入。测试脚本不会把真实密钥或报告写进 GitHub。

参考：[官方工具调用](https://api-docs.deepseek.com/guides/tool_calls/)、[官方思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。
