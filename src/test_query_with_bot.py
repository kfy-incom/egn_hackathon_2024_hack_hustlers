from annoy import AnnoyIndex
import numpy as np
import json
from src.azure_client import initialize_client
from src.dronning_chat import dronning_chat

client = initialize_client()
EMBEDDING_DTYPE = np.float32

f = 1536
u = AnnoyIndex(f, 'angular')
u.load('src/queen_speeches.ann')
with open('src/processed_speeches.json') as fp:
    text_dict = json.load(fp)
question = "Hvad var den største overraskelse i talen i 2023?"
answer = dronning_chat(question, [])
print(answer)