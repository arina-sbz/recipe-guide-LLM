import numpy as np
import pandas as pd
from torch import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai

# Initialize the GenerativeAI client
assistant = genai.GenerativeModel("gemini-1.5-flash")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Load the embeddings dataframe
embeddings_df = pd.read_csv("embeddings_df.csv")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Function to get embeddings from the model for a given text
def get_embeddings(text):
    # Tokenize the input text and move it to the GPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings (usually the last hidden state)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    print(f"Hi embedding is{embeddings}")
    return embeddings


# Function to retrieve the top N most relevant documents based on cosine similarity between the user query and document embeddings
def get_relevant_docs(user_query, embeddings_df, top_n=3):
    # Convert the user query into embeddings
    query_embeddings = np.array(get_embeddings(user_query))
    print(f"Query embeddings are as an array{query_embeddings}")

    # print(query_embeddings)

    def cosine_similarity(embedding):
        return float(
            np.dot(query_embeddings, embedding)
            / (np.linalg.norm(query_embeddings) * np.linalg.norm(embedding))
        )

    # print(embeddings_df["embeddings"])
    embeddings_df["similarity"] = embeddings_df["embeddings"].apply(
        lambda x: cosine_similarity(np.array(x)[0])
    )

    # Get the top n most relevant documents
    relevant_docs = embeddings_df.nlargest(top_n, "similarity")["input"].tolist()
    # print(relevant_docs)
    # sorted_embeddings_df = embeddings_df.sort_values(by="similarity", ascending=False)

    return relevant_docs


def make_rag_prompt(query, relevant_passage):
    # Ensure all elements in relevant_passage are strings before joining
    relevant_passage = " ".join([str(passage) for passage in relevant_passage])
    prompt = (
        f"You are a helpful and informative recipe chatbot that answers questions using text from the reference passage included below.\n\n "
        f"Add some extra information to make your response more helpful and engaging. \n\n"
        f"only anwer the questions with the topic of the recipes,ingredients, directions and cooking methods.\n\n "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"Give the answer in a markdown format.\n\n"
        f"If the answer contains Ingrediens, give them in a unordered list with a title format.\n\n"
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


def generate_response(assistant, user_prompt):
    answer = assistant.generate_content(
        user_prompt,
        stream=True,
    )
    return answer.text


def generate_answer(query):
    relevant_text = get_relevant_docs(query, embeddings_df)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer
