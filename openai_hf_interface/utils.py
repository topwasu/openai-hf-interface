import os

from .formatter import DoNothingFormatter, LLaMaChatFormatter, OpenAIChatFormatter
from .openai import OpenAI_LLM, prompt_openai_single, prompt_openai_chat_single


def get_formatter(model_name, **formatter_kwargs):
    if model_name.startswith('meta-llama') and model_name.endswith('chat-hf'):
        return LLaMaChatFormatter(**formatter_kwargs)
    elif model_name.startswith('gpt-3.5') or model_name.startswith('gpt-4') or model_name.startswith('o1'):
        return OpenAIChatFormatter(**formatter_kwargs)
    else:
        return DoNothingFormatter(**formatter_kwargs)


def create_llm(model_name, **formatter_kwargs):
    formatter = get_formatter(model_name, **formatter_kwargs)
    if 'davinci' in model_name:
        return OpenAI_LLM(model_name, prompt_openai_single, formatter)
    elif model_name.startswith('gpt-3.5') or model_name.startswith('gpt-4') or model_name.startswith('o1'):
        return OpenAI_LLM(model_name, prompt_openai_chat_single, formatter)
    elif model_name.startswith('meta-llama'):
        from .hf import HF_LLM, LLaMA_model
        return HF_LLM(model_name, LLaMA_model(model_name), formatter)
    else:
        raise NotImplementedError