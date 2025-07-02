import os
from openai_hf_interface import create_llm, choose_provider

choose_provider('ai_studio') # or choose_provider('openrouter')
llm = create_llm('gemini-1.5-flash')
llm.setup_cache('disk')
prompt1 = 'Hey adf what should we did to day?'
# prompt1 = "who are you?"
prompt2 = ('what is this picture? explain in a single sentence', './example_img/fake_pikachu.jpg')
prompt3 = [('what is this picture? explain in a single setnece', './example_img/fake_pikachu.jpg'),
           'This image depicts a stylized, electrified version of Pikachu with glowing eyes and lightning bolts in the background.',
           'It does not look like Pikachu to me. What is your second guess?']
print(llm.prompt([prompt1], temperature=0, seed=2))

# Outputs you will get vvv
# Output 1 ==> 'I am an AI language model created by OpenAI, designed to assist with a wide range of questions and tasks by providing information and generating text based on the input I receive. How can I assist you today?', 
# Output 2 ==> 'This image depicts a stylized, electrified version of Pikachu with glowing eyes and lightning bolts in the background.', 
# Output 3 ==> 'This image features a cartoonish, electrified character with large, glowing eyes and lightning bolts, resembling a playful, energetic creature.'
# Note: These three requests were sent to Openai API asynchronously!
