from embedding import query
from infer_qwen import infer
from embed_task import timestamp_map
from reranking import rerank
from utils import descriptions_file, memory_dir

import os

prompt = """You are a helpful assistant that answers questions based on provided background knowledge.
The knowledges are provided after \"Background knowledges:\", in the format of \"timestamp: knowledge\".
The format of the timestamp is YYYYmmDD_HHMMSS. Every knowledge describes a single object or event.
You should answer the question based on the background knowledges.
"""
messages = [{"role": "system", "content": prompt}]

def find_type(attachment):
    ext = os.path.splitext(attachment)[1]
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
        return "image"
    elif ext.lower() in ['.mp4', '.avi', '.mov']:
        return "video"
    else:
        raise ValueError(f"Unsupported attachment type: {ext}")

def readlines(file_pth, linenos):
    output = []
    valid_linenos = []
    with open(file_pth, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in linenos:
                output.append(line.strip())
                valid_linenos.append(i)
    return output, valid_linenos

def get_images(indices):
    global timestamp_map, memory_dir

    indices = sorted(indices)
    timestamps = []

    if len(timestamp_map) > 0:
        last_timestamp = timestamp_map[0][0]
        last_ptr = 0
        for timestamp, ptr in timestamp_map:
            if indices[0] in range(last_ptr, ptr):
                timestamps.append(last_timestamp)
                indices.pop(0)
            if not indices:
                break
            last_timestamp = timestamp
            last_ptr = ptr
        if indices:
            for _ in indices:
                timestamps.append(timestamp_map[-1][0])

    paths = []
    for timestamp in set(timestamps):
        image = memory_dir / "images" / f"{timestamp}.png"
        if image.exists():
            paths.append(str(image))
    return paths, timestamps

def get_knowledge(question, embed_driver, rerank_driver, embed_lock):
    with embed_lock:
        retrieved_indices = query(question, embed_driver)
    # print(retrieved_indices)
    if len(retrieved_indices) == 0:
        return [], [], []
    texts, _ = readlines(descriptions_file, retrieved_indices)
    new_indices = rerank(question, texts, rerank_driver)
    retrieved_indices = [retrieved_indices[i] for i in new_indices]
    texts, valid_indices = readlines(descriptions_file, retrieved_indices)
    images, timestamps = get_images(valid_indices)
    print(timestamp_map, valid_indices)
    return texts, images, timestamps

def interact(infer_driver, embed_driver, rerank_driver, infer_lock, embed_lock, message_lock, attachments, question):
    global messages
    with infer_lock:
        new_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "User query: " + question},
                ],
            }
        ]
        for attachment in attachments:
            if not attachment.strip():
                continue
            try:
                type = find_type(attachment.strip())
                new_message[0]["content"].insert(0,
                    {
                        "type": type,
                        type: "file://" + attachment.strip(),
                    }
                )
            except ValueError as e:
                print(f"Error: {e}. Skipping attachment {attachment.strip()}")
                continue
        
        knowledges, images, timestamps = get_knowledge(question, embed_driver, rerank_driver, embed_lock)
        print("Knowledges:", knowledges, len(knowledges))
        print("Timestamps:", timestamps, len(timestamps))
        assert len(knowledges) == len(timestamps)
        background = ""
        for i in range(len(knowledges)):
            background += f"{timestamps[i]}: {knowledges[i]}\n"
        new_message[0]["content"].append(
            {"type": "text", "text": "\nBackground knowledges:\n" + background}
        )
        with message_lock:
            messages += new_message

        print("Qwen:", end=" ")
        output = infer(messages, infer_driver, use_streamer=True)

    new_response = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": output[0]},
            ],
        }
    ]
    messages += new_response

    return output[0]