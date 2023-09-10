# OpenAI and Hugging Face Interface Alignment

A simple interface implementation that aligns Huggingface's Transformers interface with OpenAI's API. The interface allows switching LLM in your code base from OpenAI's GPT to Meta's LLaMA on Huggingface with just a one-line change.

## Installation

To `import openai_hf_interface` in your project, go to the project's top-level directory and run the following script:

```
git clone https://github.com/topwasu/openai-hf-interface.git
cd openai-hf-interface
pip install -e .
```

The above command only installs `openai` package for you. If you want to use Huggingface's model, you need to install `transformers` by following the instructions on their [installation page](https://huggingface.co/docs/transformers/installation). Once you've done that, please also run ```pip install sentencepiece accelerate``` so that you can use `LlamaTokenizer` and load models onto multiple gpus.

## Usage

```
import os
os.environ['OPENAI_API_KEY'] = 'PUT-YOUR-KEY-HERE'
from openai_hf_interface import create_llm

llm = create_llm('gpt-3.5-turbo')
print(llm.prompt(['Hello!', 'Bonjour!'], temperature=0, max_tokens=500))
```
Now, if you want to use LLaMA instead of ChatGPT, you simply need to change from
```
llm = create_llm('gpt-3.5-turbo')
```
to 
```
llm = create_llm('meta-llama/Llama-2-13b-hf')
```

Note: the `prompt` method takes in either a list of strings or a list of list of strings (you can think of it as a list of conversations) and return a list of strings. We have a `PromptFormatter` class to format these inputs before feeding them to the model. The codebase provides default `PromptFormatter` subclasses for OpenAI's models and Huggingface's LLama 2. Feel free to write your own custom `PromptFormatter` and override the default `PromptFormatter` by calling the method `override_formatter`. Please look at the [formatter file](openai_hf_interface/formatter.py) for more information.

## Feedback

Feel free to open a Github issue for any questions/feedback/issues. Pull request is also welcome!

## License

MIT
