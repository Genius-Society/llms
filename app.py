import gradio as gr
from modules.apis import LLM_APIs
from modules.deepseek import DeepSeek_R1_Qwen_7B
from utils import I18N

gr.TabbedInterface(
    interface_list=[LLM_APIs(), DeepSeek_R1_Qwen_7B()],
    tab_names=[I18N("API 部署聚合"), I18N("真实 DeepSeek R1 Qwen 7B 模型")],
    title=I18N("大模型部署实例合集"),
).launch(
    theme=gr.themes.Soft(),
    css="#gradio-share-link-button-0 { display: none; }",
    ssr_mode=False,
    i18n=I18N,
)
