import os 
from apikey import apikey


import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
os.environ['OPENAI_API_KEY']  = apikey


# We want the user to give the title of the desired
# YT video and we'll generate the prompts based on that 
# "prompt templates" will help us do that 

# CHAIN - will run the topic through the prompt template 
# then go  


# App Framework 
st.title('Youtube Scripter GPT')
prompt = st.text_input('Plug in your prompt here')

# Prompt Template 
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a youtube video titled {topic}'
)

#LLMs
llm = OpenAI(temperature = 0.9) # creating instance of our OpenAI server
title_chain = LLMChain(llm = llm, prompt = title_template, verbose=True)

# shows the response to the screen
if prompt:
    response = title_chain.run(topic = prompt)
    st.write(response)