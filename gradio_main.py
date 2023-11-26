import gradio as gr
import pandas as pd
import numpy as np
from scipy import spatial
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

dataframe = pd.read_csv('./embeddings.csv', index_col=0)
dataframe['embeddings'] = dataframe['embeddings'].apply(eval).apply(np.array)

dataframe.head()

def create_context(
        question, dataframe, max_len=1800, size="ada"
):
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    dataframe['distances'] = [[] for _ in range(len(dataframe))]
    # Get the distances from the embeddings
    for i in range(0, len(dataframe['embeddings'])):
        dataframe['distances', i] = 1 - spatial.distance.cosine(q_embeddings, dataframe['embeddings'][i])

    returns = []
    cur_len = 0

    for i, row in dataframe.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        question,
        model="gpt-4",
        max_len=1800,
        size="ada",
        max_tokens=150,
        stop_sequence=None
):
    context = create_context(
        question,
        dataframe,
        max_len=max_len,
        size=size,
    )
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


demo = gr.Interface(fn=answer_question, inputs="text", outputs="text")
demo.launch()
