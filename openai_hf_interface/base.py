from abc import ABC, abstractmethod

class LLMBase(ABC):
    def __init__(self, formatter):
        self.formatter = formatter

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