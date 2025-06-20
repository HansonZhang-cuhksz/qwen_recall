import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import heapq

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs, prefix_tokens, suffix_tokens, model, tokenizer):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, model, token_true_id, token_false_id, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

model_dir = r".\qwenReranking"
max_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def init(force_cpu=False):
    print("Qwen3 Reranking model loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
    if force_cpu:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="cpu").eval()
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="auto", quantization_config=quantization_config).eval()
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    print("Qwen3 Reranking model loaded successfully.")
    return model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens
        
task = 'Given a database query, retrieve relevant passages that answer the query'

def rerank(query, documents, model_tokenizer_id_tokens, top_k=5):
    if not documents:
        return []

    model, tokenizer, token_true_id, token_false_id, prefix_tokens, suffix_tokens = model_tokenizer_id_tokens

    pairs = [format_instruction(task, query, doc) for doc in documents]

    # Tokenize the input texts
    inputs = process_inputs(pairs, prefix_tokens, suffix_tokens, model, tokenizer)
    scores = compute_logits(inputs, model, token_true_id, token_false_id)

    top_indices = [i for i, _ in heapq.nlargest(top_k, enumerate(scores), key=lambda x: x[1])]
    return top_indices