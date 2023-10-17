import asyncio
import openai
import time

from .base import LLMBase


async def prompt_openai_single(model, prompt, **kwargs):
    ct = 0
    n_retries = 30
    while ct <= n_retries:
        try:
            response = await openai.Completion.acreate(model=model, prompt=prompt, **kwargs)
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
    def __init__(self, model, prompt_single_func, formatter):
        self.model = model
        self.prompt_single_func = prompt_single_func
        super().__init__(model, formatter)

    def handle_kwargs(self, kwargs):
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0
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
        outputs = asyncio.run(self._prompt_batcher(prompts, **kwargs))

        return [self.formatter.format_output(output) for output in outputs]
    
    async def aprompt(self, prompts, **kwargs):
        kwargs = self.handle_kwargs(kwargs)

        prompts = [self.formatter.format_prompt(prompt) for prompt in prompts]
        outputs = await self._prompt_batcher(prompts, **kwargs)

        return [self.formatter.format_output(output) for output in outputs]
    
    def override_formatter(self, formatter):
        self.formatter = formatter

    async def _prompt_batcher(self, prompts, **kwargs):
        all_res = []
        for ind in range(0, len(prompts), 1000): # Batch 1000 requests
            res = await asyncio.gather(*[self._get_prompt_res(prompt, **kwargs) for prompt in prompts[ind:ind+1000]])
            all_res += res
        return all_res
    
    async def _get_prompt_res(self, prompt, **kwargs):
        cache_res = self.lookup_cache(prompt, **kwargs)
        if cache_res is not None:
            return cache_res
        
        res = await self.prompt_single_func(self.model, prompt, **kwargs)
        self.update_cache(prompt, res, **kwargs)
        return res