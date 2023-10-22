# ------------ IDEA -----------------
# We want the user to give the title of the desired
# YT video and we'll generate the prompts based on that 
# "prompt templates" will help us do that 

# we'll pass through the topic to the Title Chain, it'll generate 
# a title then we'll take that title and the wiki research and 
# we'll pass it to the Script Chain 


import os 
from apikey import apikey

import streamlit as st 
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper  


os.environ['OPENAI_API_KEY'] = apikey


# App Framework 
st.title('Youtube Scripter GPT')
prompt = st.text_input('Plug in your prompt here')

# Prompt Template 
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template = 'Write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research}'
)


# Memory
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')


#LLMs
# llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length": 800})
llm = OpenAI(temperature=0.9) 

title_chain = LLMChain(
    llm = llm,
    prompt = title_template,
    verbose=True, 
    output_key='title',
    memory = title_memory)

script_chain = LLMChain(
    llm = llm, 
    prompt = script_template, 
    verbose=True, 
    output_key='script',
    memory = script_memory)

wiki = WikipediaAPIWrapper()


# Shows the response to the screen
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research )
    
    st.write(title)
    st.write(script)

    with st.expander('Title history'):
        st.info(title_memory.buffer)
    
    with st.expander('Wikipedia Research history'):
        st.info(wiki_research)
    
    with st.expander('Script history'):
        st.info(script_memory.buffer)
