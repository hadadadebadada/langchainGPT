# import os
# from apikey import apikey

# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI

# chat = ChatOpenAI(temperature=0)

# os.environ['OPENAI_API_KEY'] = apikey

# st.title('GOGOGOGOGOG')
# prompt = st.text_input("PROMPT HERE!")

# title_template = PromptTemplate(
#     input_variables=['topic'],
#     template='write me a story about {topic}.'
# )

# # OPENAI LLM
# llm = OpenAI(temperature=0.9)
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

# # Check if the previous_responses list exists in session state, if not, create it
# if 'previous_responses' not in st.session_state:
#     st.session_state.previous_responses = []

# if prompt:
#     response = title_chain.run(topic=prompt)
#     st.write(response)

#     # Add the response to the list of previous responses
#     st.session_state.previous_responses.append(response)

#     # Display the list of previous responses
#     st.write("Previous responses:")
#     for i, resp in enumerate(st.session_state.previous_responses[:-1]):  # Exclude the current response
#         st.write(f"{i+1}. {resp}")

###########################################################################################################################################################

import os
from apikey import apikey

import streamlit as st
from langchain import OpenAI, ConversationChain

os.environ['OPENAI_API_KEY'] = apikey

st.title('GOGOGOGOGOG')
prompt = st.text_input("PROMPT HERE!")

llm = OpenAI(temperature=0.9)
conversation = ConversationChain(llm=llm, verbose=True)

# Check if the conversation chain exists in session state, if not, create it
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = conversation

# Initialize the list of previous responses in session state
if 'previous_responses' not in st.session_state:
    st.session_state.previous_responses = []

if prompt:
    response = st.session_state.conversation_chain.predict(input=prompt)
    st.write(response)

    # Add the response to the list of previous responses
    st.session_state.previous_responses.append(response)

    # Display the list of previous responses in reverse order
    st.write("Previous responses:")
    for i, resp in reversed(list(enumerate(st.session_state.previous_responses[:-1]))):  # Exclude the current response
        st.write(f"{i+1}. {resp}")


#########################################################################################################################################################

# import osimport os
# from apikey import apikey

# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )

# os.environ["OPENAI_API_KEY"] = apikey

# chat = ChatOpenAI(temperature=0)

# st.title("Chatbot App")

# user_input = st.text_input("Enter your message here:")

# # Define the chat prompt template
# system_message_template = "You are a nasty assistant."
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
# human_message_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)

# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# # Set up the LLM chain with the chat model
# chain = LLMChain(llm=chat, prompt=chat_prompt)

# if user_input:
#     response = chain.run(text=user_input)
#     st.write(f"Chatbot: {response}")

# from langchain.chains import LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,Michael Jackson passed away in 2009. He is no longer living. His body was laid to rest in a mausoleum at the Forest Lawn Memorial Park in Glendale, California.
# )

# os.environ["OPENAI_API_KEY"] = apikey

# chat = ChatOpenAI(temperature=0)AttributeError: 'ConversationChain' object has no attribute '_history'


# st.title("Chatbot App")

# user_input = st.text_input("Enter your message here:")

# # Define the chat prompt template
# system_message_template = "You are a nasty assistant."
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)
# human_message_template = "{text}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)

# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# # Set up the LLM chain with the chat model
# chain = LLMChain(llm=chat, prompt=chat_prompt)

# if user_input:
#     response = chain.run(text=user_input)
#     st.write(f"Chatbot: {response}")
