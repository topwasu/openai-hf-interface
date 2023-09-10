# OpenAI and Hugging Face Interface Alignment

A simple interface implementation that aligns Huggingface's Transformers interface with OpenAI's API. The interface allows switching LLM in your code base from OpenAI's GPT to Meta's LLaMA on Huggingface with just a one-line change.

## Installation

To `import openai_hf_interface` in your project, go to the project's top-level directory and run the following script:

```
git clone https://github.com/topwasu/openai-hf-interface.git
cd openai-hf-interface
pip install -e .
```

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

## Feedback

Feel free to open a Github issue for any questions/feedback/issues. Pull request is also welcome!