import torch
import torch.nn.functional as F
from torch import Tensor
from modelscope import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import hnswlib
import numpy as np
import streamlit as st

from pathlib import Path

model_dir = Path(__file__).parent / 'qwenEmbedding'
max_length = 8192

@st.cache_resource
def init(force_cpu=False):
    print("Qwen3 Embedding model loading...")
    # Init Embedding Model
    if force_cpu:
        model = AutoModel.from_pretrained(model_dir, torch_dtype="auto", device_map="cpu")
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModel.from_pretrained(model_dir, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
    print("Qwen3 Embedding model loaded successfully.")
    # Init HNSWLIB index
    p = hnswlib.Index(space='cosine', dim=1024)
    p.init_index(max_elements=10000, ef_construction=200, M=16)
    return model, tokenizer, p

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embed(documents, model_tokenizer_p, is_query=False):
    model, tokenizer, p = model_tokenizer_p

    # Tokenize the input texts
    batch_dict = tokenizer(
        documents,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_cpu = embeddings.cpu().detach().to(torch.float32).numpy()
    # print(f"Embedding shape: {embeddings_cpu.shape}")

    if not is_query:
        p.add_items(embeddings_cpu, np.arange(len(documents)))
    return embeddings_cpu

def query(query_text, model_tokenizer_p, k=20):
    _, _, p = model_tokenizer_p
    query_vec = embed([query_text], model_tokenizer_p, is_query=True)
    # print(query_vec)
    labels, distances = p.knn_query(query_vec, k=k if p.get_current_count() > k else p.get_current_count())
    retrieved_indices = labels[0]
    return retrieved_indices