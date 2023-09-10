from abc import ABC, abstractmethod


class PromptFormatter(ABC):
    @abstractmethod
    def format_prompt(self):
        pass

    @abstractmethod
    def format_output(self):
        pass


class DoNothingFormatter(PromptFormatter):
    def format_prompt(self, prompt):
        return prompt
    
    def format_output(self, output):
        return output
    

class LLaMaChatFormatter(PromptFormatter):
    def __init__(self, instruction=None): 
        self.instruction = instruction

    def format_prompt(self, prompt):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        BOS, EOS = "<s>", "</s>"
        DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

        if isinstance(prompt, str): 
            prompt = [prompt]

        if self.instruction is None:
            prompt = [DEFAULT_SYSTEM_PROMPT] + prompt
        prompt = [B_SYS + prompt[0] + E_SYS + prompt[1]] + prompt[2:]

        formatted = [
            f"{BOS}{B_INST} {(question).strip()} {E_INST} {(answer).strip()} {EOS}"
            for question, answer in zip(prompt[::2], prompt[1::2])
        ]
        formatted.append(f"{BOS}{B_INST} {(prompt[-1]).strip()} {E_INST}")

        return "".join(formatted)
    
    def format_output(self, output):
        return output


class OpenAIChatFormatter(PromptFormatter):
    def __init__(self, instruction=None): 
        self.instruction = instruction

    def format_prompt(self, prompt):
        if isinstance(prompt, str): 
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = []

            for user_msg, assistant_msg in zip(prompt[::2], prompt[1::2]):
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": prompt[-1]})

        if self.instruction is not None:
            messages = [{"role": "system", "content": self.instruction}] + messages
        return messages

    def format_output(self, output):
        return output