import gradio as gr
from src.app_interface import combine_results

DESCRIPTION = """
![img](file/kickasslogo.jpg)

This is hack hustler's submission for the LLM Hackathon.

"""
with open('kickasslogo_base64.txt', 'r') as fp:
    data_string = fp.read()
DESCRIPTION = (
            "<div >"
            + f'<img src="{data_string}" height="300px" width="300px">'
            + "</div>"
    )

gr.ChatInterface(
    combine_results,
    description=DESCRIPTION,
    title="Hack Hustlers at twoday Kapacity EGN OpenAI hackathon",
).launch()
