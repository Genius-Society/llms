import gradio as gr

ZH2EN = {  # main ↓
    "大模型部署实例合集": "LLM Deployment Instances",
    "API 部署聚合": "API Aggregation",
    "真实 DeepSeek R1 Qwen 7B 模型": "Real DeepSeek R1 Qwen 7B",  # apis ↓
    "请先在设置中配置有效 API 密钥": "Please set valid api keys in settings first.",
    "⚙️ 设置": "⚙️ Settings",
    "模型选择": "Select a model",
    "API 密钥": "API key",
    "系统提示词": "System prompt",
    "最大 token 数": "Max new tokens",
    "温度参数": "Temperature",
    "Top-P 采样": "Top P sampling",  # deepseek ↓
    "有算力的可自行克隆至本地或复刻至购买了 GPU 环境的账号测试": "If you have computing power, you can test by cloning to local or forking to an account with purchased GPU environment",
    "⚙️ 参数设置": "⚙️ Parameters",
    "系统提示词": "System prompt",
    "最大 token 数": "Max new tokens",
    "温度参数": "Temperature",
    "Top-K 采样": "Top K sampling",
    "Top-P 采样": "Top P sampling",
    "重复性惩罚": "Repetition penalty",
    "当前仅提供模型的 ModelScope 版部署实例，有算力的可自行克隆至本地或复刻至购买了 GPU 环境的账号测试": "This is a HuggingFace deployment instance of model, if you have computing power, you can test by cloning to local or forking to an account with purchased GPU environment",
}

I18N = gr.I18n(
    zh={key: key for key in ZH2EN},
    en=ZH2EN,
)
