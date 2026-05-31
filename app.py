import gradio as gr
from modules.apis import LLM_APIs
from modules.deepseek import DeepSeek_R1_Qwen_7B
from utils import EN_US

ZH2EN = {
    "大模型部署实例合集": "LLM Deployment Instances",
    "API 部署聚合": "API Aggregation",
    "真实 DeepSeek R1 Qwen 7B 模型": "Real DeepSeek R1 Qwen 7B",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


gr.TabbedInterface(
    interface_list=[LLM_APIs(), DeepSeek_R1_Qwen_7B()],
    tab_names=[_L("API 部署聚合"), _L("真实 DeepSeek R1 Qwen 7B 模型")],
    title=_L("大模型部署实例合集"),
).launch(
    theme=gr.themes.Soft(),
    css="#gradio-share-link-button-0 { display: none; }",
    ssr_mode=False,
)
