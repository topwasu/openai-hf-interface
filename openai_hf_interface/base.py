from abc import ABC, abstractmethod

from .cache import InMemoryCache, SQLiteCache


class LLMBase(ABC):
    def __init__(self, model_name, formatter):
        self.model_name = model_name
        self.formatter = formatter
        self.cache_type = None
        self.cache = None

    @abstractmethod
    def prompt(self, prompts):
        """
        Prompt the LLM

        :param prompts: list of (string or list of strings), the latter is for multi-turn conversation
        """
        pass

    @abstractmethod
    def score(self, prompts):
        """
        Prompt the LLM

        :param prompts: list of strings only
        """
        pass

    def override_formatter(self, formatter):
        """
        Prompt the LLM

        :param prompts: list of strings only
        """
        self.formatter = formatter

    def setup_cache(self, cache_type):
        self.cache_type = cache_type
        if self.cache_type == 'in_memory':
            self.cache = InMemoryCache()
        elif self.cache_type == 'disk':
            self.cache = SQLiteCache()
        else:
            raise NotImplementedError

    def lookup_cache(self, prompt, **kwargs):
        if self.cache is None:
            return None
        stop = kwargs['stop'] if 'stop' in kwargs else []
        prompt_str = self.formatter.prompt_to_string(prompt)
        res = self.cache.lookup(prompt_str, self.model_name, kwargs['temperature'], kwargs['max_tokens'], stop)
        print(res)
        return res
    
    def update_cache(self, prompt, ret_val, **kwargs):
        if self.cache is None:
            return
        stop = kwargs['stop'] if 'stop' in kwargs else []
        prompt_str = self.formatter.prompt_to_string(prompt)
        self.cache.update(prompt_str, self.model_name, [ret_val], kwargs['temperature'], kwargs['max_tokens'], stop)
