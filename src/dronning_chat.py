from annoy import AnnoyIndex
from src.azure_client import initialize_client
import numpy as np
import json

client = initialize_client()
EMBEDDING_DTYPE = np.float32

f = 1536
u = AnnoyIndex(f, 'angular')
u.load('src/queen_speeches.ann')
with open('src/processed_speeches.json') as fp:
    text_dict = json.load(fp)

def dronning_chat(question, history):
    """
    This is where your logic goes!
    For an example see: https://www.gradio.app/guides/creating-a-chatbot-fast
    
    """
    embedding = client.embeddings.create(input=question, model="text-embedding-ada-002").data[0].embedding
    embedding = np.array(embedding, dtype=EMBEDDING_DTYPE)
    vectors = u.get_nns_by_vector(embedding, 5, search_k=-1, include_distances=True)
    texts = [text_dict[str(i)] for i in vectors[0]]
    selected_conversation_hist = [
        {"role": "system",
        "content": f"""
            You are a polite and helpful assistant having a conversation with a human. 
            You are answering questions to the best of your ability. 
            You are not trying to be funny or clever. You are trying to be helpful. 
            You are not trying to show off.
            You answer the question in the same language as the question is asked.
            """},
    ]
    modified_question = "Brugeren spurgte: \n" + \
                        question + \
                        '\n Du har nu følgende fra dronningens nytårstaler fra 2001-2023 at svare ud fra: \n' + \
                        '\n '.join(texts) + \
                        '\n Besvar spørgsmålet.'
    # Add question to conversation
    messages = selected_conversation_hist + [{"role": "user", "content": modified_question}]

    raw_answer = client.chat.completions.create(model="gpt-4-1106-preview", messages=messages)
    answer = raw_answer.choices[0].message.content
    return answer