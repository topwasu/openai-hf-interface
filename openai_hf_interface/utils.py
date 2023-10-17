from .formatter import DoNothingFormatter, LLaMaChatFormatter, OpenAIChatFormatter
from .hf import HF_LLM, LLaMA_model
from .openai import OpenAI_LLM, prompt_openai_single, prompt_openai_chat_single


def get_formatter(model_name):
    if model_name.startswith('meta-llama') and model_name.endswith('chat-hf'):
        return LLaMaChatFormatter()
    elif model_name.startswith('gpt-3.5') or model_name.startswith('gpt-4'):
        return OpenAIChatFormatter()
    else:
        return DoNothingFormatter()


def create_llm(model_name):
    formatter = get_formatter(model_name)
    if 'davinci' in model_name:
        return OpenAI_LLM(model_name, prompt_openai_single, formatter)
    elif model_name.startswith('gpt-3.5') or model_name.startswith('gpt-4'):
        return OpenAI_LLM(model_name, prompt_openai_chat_single, formatter)
    elif model_name.startswith('meta-llama'):
        return HF_LLM(LLaMA_model(model_name), formatter)
    else:
        raise NotImplementedError