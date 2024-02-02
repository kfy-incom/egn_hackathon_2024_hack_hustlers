import gradio as gr
from src.dronning_chat import dronning_chat

DESCRIPTION = """
![img](https://studerendeonline.dk/images/dynamic/company/logo/109481)

This is hack hustler's submission for the LLM Hackathon.

"""

gr.ChatInterface(dronning_chat, description=DESCRIPTION, title="twoday Kapacity EGN OpenAI hackathon").launch()