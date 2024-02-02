from annoy import AnnoyIndex
import numpy as np
import json
EMBEDDING_DTYPE = np.float32
import src.azure_client as azure_client
client = azure_client.initialize_client()
f = 1536
t = AnnoyIndex(f, 'angular')
z = AnnoyIndex(f, 'angular')
idx = 0
idx_z = 0
text_dict = {}
text_dict_summary = {}
for i in range(2001, 2024):
    with open(f'queens_speeches/speeches/{i}.txt', encoding='UTF-8') as f:
        total_text = f"År {i}"
        for line_idx, line in enumerate(f.readlines()):
            total_text += '\n' + line
            line = line.strip()
            if len(line)>0:
                line = f'År: {i}\n Linjenummer: {line_idx} \n' +  line
                embedding = client.embeddings.create(input=line, model="text-embedding-ada-002").data[0].embedding
                embedding = np.array(embedding, dtype=EMBEDDING_DTYPE)
                t.add_item(idx, embedding)
                text_dict[idx] = line
                idx += 1
    selected_conversation_hist = [
        {"role": "system",
         "content": f"""
            Your job is to summarize the Danish Queens new years speeches.
            Please include information such as but not limited to 
            themes, people, countries, facts, events and minorities.
            Please do so in Danish.
             """},
    ]
    modified_question = "Opsummer følgende tale: \n" + \
                        total_text
    # Add question to conversation
    messages = selected_conversation_hist + [{"role": "user", "content": modified_question}]

    raw_answer = client.chat.completions.create(model="gpt-35-turbo", messages=messages)
    answer = raw_answer.choices[0].message.content
    if answer is None:
        raise NotImplementedError()
    for summary_idx, summary_text in enumerate(answer.split('.')):
        summary_text = f'År: {i}\n Linjenummer: {summary_idx} \n' +  summary_text
        embedding = client.embeddings.create(input=summary_text, model="text-embedding-ada-002").data[0].embedding
        embedding = np.array(embedding, dtype=EMBEDDING_DTYPE)
        z.add_item(idx_z, embedding)
        text_dict_summary[idx_z] = summary_text
        idx_z += 1
    print(i)
t.build(10) # 10 trees
t.save('src/queen_speeches.ann')
z.build(10)
z.save('src/queen_speeches_summaries.ann')

with open('src/processed_speeches.json', 'w') as fp:
    json.dump(text_dict, fp)
with open('src/processed_speeches_summaries.json', 'w') as fp:
    json.dump(text_dict_summary, fp)