import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import openai
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
import os
import torch

# Define API keys and model paths
openai.api_key = os.environ.get("open_api_key")
model_dir = "model/"

# Load the fine-tuned model and tokenizer
tokenizer_finetuned = RobertaTokenizerFast.from_pretrained(model_dir)
model_finetuned = RobertaForQuestionAnswering.from_pretrained(model_dir)
tokenizer_finetuned = RobertaTokenizerFast.from_pretrained('deepset/roberta-large-squad2')
model_finetuned = RobertaForQuestionAnswering.from_pretrained('deepset/roberta-large-squad2')
model_finetuned.to('cuda' if torch.cuda.is_available() else 'cpu')

from FAISS_INDEX import load_faiss_index

def generate_answer(question, relevant_chunk, model_choice="openai"):
    """Generate answer using either OpenAI, Hugging Face, or the fine-tuned model."""
    context = relevant_chunk

    if model_choice == "openai":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}]
        )
        answer = response.choices[0].message['content']
    elif model_choice == "hugging_face":
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
        response = qa_pipeline(question=question, context=context)
        answer = response['answer']
    elif model_choice == "finetuned_roberta":
        inputs = tokenizer_finetuned.encode_plus(question, context, add_special_tokens=True, return_tensors="pt").to(model_finetuned.device)
        with torch.no_grad():
            outputs = model_finetuned(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
        answer = tokenizer_finetuned.convert_tokens_to_string(tokenizer_finetuned.convert_ids_to_tokens(answer_tokens))
        qa_pipeline = pipeline("question-answering", model=model_finetuned,
                               tokenizer=tokenizer_finetuned)
        response = qa_pipeline(question=question, context=context)
        answer = response['answer']
    else:
        answer = "Unsupported model choice."

    return answer

def answer_question(question, model_choice="openai", filename="faiss_index.pkl"):
    index, urls, chunks = load_faiss_index(filename)
    question_embedding = SentenceTransformer('all-mpnet-base-v2').encode([question])
    D, I = index.search(question_embedding, 1)
    print("Debug",D[0][0])
    if D[0][0] > 1:
        return "I couldn't find the answer in my database.", None

    relevant_chunk, source_url = chunks[I[0][0]], urls[I[0][0]]
    answer = generate_answer(question, relevant_chunk, model_choice)
    print(question, relevant_chunk)
    if not answer or answer.startswith("Unsupported model choice"):
        return "I couldn't find the answer in my database.", None

    # Now including source URL in the return value
    return answer, source_url