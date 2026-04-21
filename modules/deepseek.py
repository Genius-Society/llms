import torch
import gradio as gr
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from utils import EN_US

ZH2EN = {
    "有算力的可自行克隆至本地或复刻至购买了 GPU 环境的账号测试": "If you have computing power, you can test by cloning to local or forking to an account with purchased GPU environment",
    "⚙️ 参数设置": "⚙️ Parameters",
    "系统提示词": "System prompt",
    "最大 token 数": "Max new tokens",
    "温度参数": "Temperature",
    "Top-K 采样": "Top K sampling",
    "Top-P 采样": "Top P sampling",
    "重复性惩罚": "Repetition penalty",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = MODEL_ID.split("/")[-1]
CONTEXT_LENGTH = 16000
DESCRIPTION = (
    f"This is a HuggingFace deployment instance of {MODEL_NAME} model, if you have computing power, you can test by cloning to local or forking to an account with purchased GPU environment"
    if EN_US
    else f"当前仅提供 {MODEL_NAME} 模型的 ModelScope 版部署实例，有算力的可自行克隆至本地或复刻至购买了 GPU 环境的账号测试"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda"):
    if EN_US:
        import huggingface_hub

        MODEL_DIR = huggingface_hub.snapshot_download(
            MODEL_ID, cache_dir="./__pycache__"
        )

    else:
        import modelscope

        MODEL_DIR = modelscope.snapshot_download(MODEL_ID, cache_dir="./__pycache__")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")


def predict(msg, history, prompt, temper, max_tokens, top_k, repeat_penalty, top_p):
    # Format history with a given chat template
    stop_tokens = ["<|endoftext|>", "<|im_end|>", "|im_end|"]
    instruction = "<|im_start|>system\n" + prompt + "\n<|im_end|>\n"
    for user, assistant in history:
        instruction += f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n{assistant}\n<|im_end|>\n"

    instruction += f"<|im_start|>user\n{msg}\n<|im_end|>\n<|im_start|>assistant\n"
    try:
        if device == torch.device("cpu"):
            raise EnvironmentError(
                _L("有算力的可自行克隆至本地或复刻至购买了 GPU 环境的账号测试")
            )

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        input_ids, attention_mask = enc.input_ids, enc.attention_mask
        if input_ids.shape[1] > CONTEXT_LENGTH:
            input_ids = input_ids[:, -CONTEXT_LENGTH:]
            attention_mask = attention_mask[:, -CONTEXT_LENGTH:]

        generate_kwargs = dict(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            streamer=streamer,
            do_sample=True,
            temperature=temper,
            max_new_tokens=max_tokens,
            top_k=top_k,
            repetition_penalty=repeat_penalty,
            top_p=top_p,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

    except Exception as e:
        streamer = f"{e}"

    outputs = []
    for new_token in streamer:
        outputs.append(new_token)
        if new_token in stop_tokens:
            break

        yield "".join(outputs)


def DeepSeek_R1_Qwen_7B():
    with gr.Accordion(label=_L("⚙️ 参数设置"), open=False) as ds_acc:
        prompt = gr.Textbox(
            "You are a useful assistant. first recognize user request and then reply carfuly and thinking",
            label=_L("系统提示词"),
        )
        temper = gr.Slider(0, 1, 0.6, label=_L("温度参数"))
        maxtoken = gr.Slider(0, 32000, 10000, label=_L("最大 token 数"))
        topk = gr.Slider(1, 80, 40, label=_L("Top-K 采样"))
        repet = gr.Slider(0, 2, 1.1, label=_L("重复性惩罚"))
        topp = gr.Slider(0, 1, 0.95, label=_L("Top-P 采样"))

    return gr.ChatInterface(
        predict,
        description=DESCRIPTION,
        additional_inputs_accordion=ds_acc,
        additional_inputs=[prompt, temper, maxtoken, topk, repet, topp],
    ).queue()
