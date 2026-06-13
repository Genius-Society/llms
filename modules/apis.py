import os
import gradio as gr
from openai import OpenAI
from utils import I18N


def predict(msg, history, system_prompt, model, api_url, api_key, max_tk, temp, top_p):
    try:
        if not api_key:
            raise ValueError(I18N.translations["en"]["请先在设置中配置有效 API 密钥"])

        msgs = [{"role": "system", "content": system_prompt}]
        for user, assistant in history:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "system", "content": assistant})

        msgs.append({"role": "user", "content": msg})
        client = OpenAI(api_key=api_key, base_url=api_url)
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=max_tk,
            temperature=temp,
            top_p=top_p,
            stream=False,
        ).to_dict()["choices"][0]["message"]["content"]

    except Exception as e:
        response = f"{e}"

    return response


def deepseek(message, history, model, api_key, system_prompt, max_tk, temp, top_p):
    response = predict(
        message,
        history,
        system_prompt,
        model,
        "https://api.deepseek.com",
        api_key,
        max_tk,
        temp,
        top_p,
    )
    outputs = []
    for new_token in response:
        outputs.append(new_token)
        yield "".join(outputs)


def kimi(message, history, model, api_key, system_prompt, max_tk, temp, top_p):
    response = predict(
        message,
        history,
        system_prompt,
        model,
        "https://api.moonshot.cn/v1",
        api_key,
        max_tk,
        temp,
        top_p,
    )
    outputs = []
    for new_token in response:
        outputs.append(new_token)
        yield "".join(outputs)


def LLM_APIs():
    with gr.Blocks() as apis:
        with gr.Tab("DeepSeek"):
            with gr.Accordion(label=I18N("⚙️ 设置"), open=False) as ds_acc:
                ds_model = gr.Dropdown(
                    choices=["deepseek-chat", "deepseek-reasoner"],
                    value="deepseek-chat",
                    label=I18N("模型选择"),
                )
                ds_key = gr.Textbox(
                    os.getenv("ds_api_key"),
                    type="password",
                    label=I18N("API 密钥"),
                )
                ds_sys = gr.Textbox(
                    "You are a useful assistant. first recognize user request and then reply carfuly and thinking",
                    label=I18N("系统提示词"),
                )
                ds_maxtk = gr.Slider(0, 32000, 10000, label=I18N("最大 token 数"))
                ds_temp = gr.Slider(0, 1, 0.3, label=I18N("温度参数"))
                ds_topp = gr.Slider(0, 1, 0.95, label=I18N("Top-P 采样"))

            gr.ChatInterface(
                deepseek,
                additional_inputs=[
                    ds_model,
                    ds_key,
                    ds_sys,
                    ds_maxtk,
                    ds_temp,
                    ds_topp,
                ],
            )

        with gr.Tab("Kimi"):
            with gr.Accordion(label=I18N("⚙️ 设置"), open=False):
                kimi_model = gr.Dropdown(
                    choices=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
                    value="moonshot-v1-32k",
                    label=I18N("模型选择"),
                )
                kimi_key = gr.Textbox(
                    os.getenv("kimi_api_key"),
                    type="password",
                    label=I18N("API 密钥"),
                )
                kimi_sys = gr.Textbox(
                    "You are a useful assistant. first recognize user request and then reply carfuly and thinking",
                    label=I18N("系统提示词"),
                )
                kimi_maxtk = gr.Slider(0, 32000, 10000, label=I18N("最大 token 数"))
                kimi_temp = gr.Slider(0, 1, 0.3, label=I18N("温度参数"))
                kimi_topp = gr.Slider(0, 1, 0.95, label=I18N("Top-P 采样"))

            gr.ChatInterface(
                kimi,
                additional_inputs=[
                    kimi_model,
                    kimi_key,
                    kimi_sys,
                    kimi_maxtk,
                    kimi_temp,
                    kimi_topp,
                ],
            )

    return apis.queue()
