import asyncio
import logging
import openai
import time

from .base import LLMBase


async def prompts_async_handler(model, prompts, func, **kwargs):
    all_res = []
    for ind in range(0, len(prompts), 1000): # Batch 1000 requests
        res = await asyncio.gather(*[func(model, prompt, **kwargs) for prompt in prompts[ind:ind+1000]])
        all_res += res
    return all_res


async def prompt_openai_single(model, prompt, **kwargs):
    ct = 0
    n_retries = 30
    while ct <= n_retries:
        try:
            response = openai.Completion.create(model=model, prompt=prompt, **kwargs)
            return response['choices'][0]['text']
        except Exception as e:
            ct += 1
            print(f'Exception occured: {e}')
            print(f'Waiting for {10 * ct} seconds')
            time.sleep(5 * ct)


async def prompt_openai_chat_single(model, messages, **kwargs):
    ct = 0
    n_retries = 10
    while ct <= n_retries:
        try:
            response = await openai.ChatCompletion.acreate(model=model, messages=messages, **kwargs)
            return response['choices'][0]['message']['content']
        except Exception as e: 
            ct += 1
            print(f'Exception occured: {e}')
            print(f'Waiting for {10 * ct} seconds')
            await asyncio.sleep(10 * ct)


class OpenAI_LLM(LLMBase):
    def __init__(self, model, formatter, prompt_single_func):
        self.model = model
        self.formatter = formatter
        self.prompt_single_func = prompt_single_func

    def handle_kwargs(self, kwargs):
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 1000
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 180 if self.model.startswith('gpt-4') else 30
        if 'request_timeout' not in kwargs:
            kwargs['request_timeout'] = 180 if self.model.startswith('gpt-4') else 30
        return kwargs

    def prompt(self, prompts, **kwargs):
        kwargs = self.handle_kwargs(kwargs)

        prompts = [self.formatter.format_prompt(prompt) for prompt in prompts]
        outputs = asyncio.run(prompts_async_handler(self.model, prompts, self.prompt_single_func, **kwargs))

        return [self.formatter.format_output(output) for output in outputs]
    
    async def aprompt(self, prompts, **kwargs):
        kwargs = self.handle_kwargs(kwargs)

        prompts = [self.formatter.format_prompt(prompt) for prompt in prompts]
        outputs = await prompts_async_handler(self.model, prompts, self.prompt_single_func, **kwargs)

        return [self.formatter.format_output(output) for output in outputs]

    def score(self, prompts):
        return asyncio.run(prompts_async_handler(prompts, self._score_async))
    
    def override_formatter(self, formatter):
        self.formatter = formatter
