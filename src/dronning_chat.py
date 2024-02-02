from annoy import AnnoyIndex
from src.azure_client import initialize_client
import numpy as np
import json

client = initialize_client()
EMBEDDING_DTYPE = np.float32

f = 1536
u = AnnoyIndex(f, "angular")
z = AnnoyIndex(f, "angular")
u.load("src/queen_speeches.ann")
z.load("src/queen_speeches_summaries.ann")
with open("src/processed_speeches.json") as fp:
    text_dict = json.load(fp)
with open("src/processed_speeches_summaries.json") as fp:
    summary_text_dict = json.load(fp)


def dronning_chat(question, history):
    """
    This is where your logic goes!
    For an example see: https://www.gradio.app/guides/creating-a-chatbot-fast

    """
    embedding = (
        client.embeddings.create(input=question, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )
    embedding = np.array(embedding, dtype=EMBEDDING_DTYPE)
    vectors = u.get_nns_by_vector(embedding, 20, search_k=-1, include_distances=True)
    texts = [text_dict[str(i)] for i in vectors[0]]
    summary_vectors = z.get_nns_by_vector(
        embedding, 20, search_k=-1, include_distances=True
    )
    summary_texts = [summary_text_dict[str(j)] for j in summary_vectors[0]]
    selected_conversation_hist = [
        {
            "role": "system",
            "content": f"""
            You are a polite and helpful assistant having a conversation with a human. 
            You are answering questions to the best of your ability. 
            You are not trying to be funny or clever. You are trying to be helpful. 
            You are not trying to show off.
            You answer the question in the same language as the question is asked.
            You are an expert on the Danish royal family. You have intimate
            knowledge of the family members, their relations and events
            that have occurred in their family.
            
            Be aware that Queen Margrethe II abdicated on January 14th 2024, so
            from that date the regent of Denmark is King Frederik X, formerly
            Crown-prince Frederik, Queen Margrete's son. Likewise, Crown-princess
            Mary is Queen Mary as of January 14th 2024. In your responses,
            refer to the current monarch as King Frederik X and the current
            queen as Queen Mary.
            """,
        },
    ]
    combined_texts = "\n".join(texts)
    combined_summary_texts = "\n".join(summary_texts)
    modified_question = f"""
        Brugeren spurgte:
        {question}
        Du har nu følgende fra dronningens nytårstaler fra 2001-2023:
        {combined_texts}
        Samt følgende opsummeringer:
        {combined_summary_texts}:
        Besvar spørgsmålet ud fra nytårstalerne, samt din generelle viden om den royale familie.
        Svar i følgende json format:
        {{
            'answer': ANSWER_AS_STRING, 
            'confidence': CONFIDENCE_AS_FLOAT_BETWEEN_0_AND_1
        }}
        Hvis confidence er under 0.1, så angiv i svaret at du er usikker, men giv dit bedste bud.
        """
    # Add question to conversation
    messages = selected_conversation_hist + [
        {"role": "user", "content": modified_question}
    ]

    raw_answer = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    answer = raw_answer.choices[0].message.content
    return answer
