from utils import descriptions_file
from embedding import embed

import shared
import time

file_ptr = 0
timestamp_map = []

def embed_task(embed_driver, embed_lock):
    global file_ptr, descriptions_file, timestamp_map

    while True:
        if not shared.decode_to_embed_1:
            time.sleep(1)
            continue

        timestamp_knowledges = shared.decode_to_embed_1
        shared.decode_to_embed_1 = None
        timestamp, knowledges = timestamp_knowledges
        knowledges = knowledges.split("\n")
        for knowledge in knowledges:
            knowledge = knowledge.strip()
            if knowledge:
                with open(descriptions_file, "a", encoding="utf-8") as f:
                    f.write(f"{knowledge}\n")
        timestamp_map.append((timestamp, file_ptr))
        file_ptr += len(knowledges)
        with embed_lock:
            embed(knowledges, embed_driver)