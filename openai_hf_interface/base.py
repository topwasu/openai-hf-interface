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
        prompt_str = self.formatter.prompt_to_string(prompt)
        temp = kwargs['temperature'] if 'temperature' in kwargs else 0
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs else kwargs['max_length'] if 'max_tokens' in kwargs else -1
        stop = kwargs['stop'] if 'stop' in kwargs else []
        return self.cache.lookup(prompt_str, self.model_name, temp, max_tokens, stop)
    
    def update_cache(self, prompt, ret_val, **kwargs):
        if self.cache is None:
            return
        prompt_str = self.formatter.prompt_to_string(prompt)
        temp = kwargs['temperature'] if 'temperature' in kwargs else 0
        max_tokens = kwargs['max_tokens'] if 'max_tokens' in kwargs else kwargs['max_length'] if 'max_tokens' in kwargs else -1
        stop = kwargs['stop'] if 'stop' in kwargs else []
        self.cache.update(prompt_str, self.model_name, [ret_val], temp, max_tokens, stop)
