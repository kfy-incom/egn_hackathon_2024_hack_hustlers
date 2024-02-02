import gradio as gr
from src.app_interface import combine_results

DESCRIPTION = """
![img](https://studerendeonline.dk/images/dynamic/company/logo/109481)

This is hack hustler's submission for the LLM Hackathon.

"""

gr.ChatInterface(combine_results, description=DESCRIPTION, title="twoday Kapacity EGN OpenAI hackathon").launch()