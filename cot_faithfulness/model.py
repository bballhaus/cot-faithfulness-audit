import os
import torch
from transformer_lens import HookedTransformer


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Instruct models supported by TransformerLens that expose residual streams,
# usable for the cross-family generalization check. They use different chat
# templates, so prompt construction must go through prompts.format_chat
# (tokenizer.apply_chat_template) rather than the hand-built Llama-3 template.
ALT_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
}


def hf_login(token=None):
    token = token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name=DEFAULT_MODEL, device=None, dtype=None, hf_token=None):
    hf_login(hf_token)
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float16 if device != "cpu" else torch.float32
    model = HookedTransformer.from_pretrained(
        name,
        device=device,
        dtype=dtype,
        default_padding_side="left",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
