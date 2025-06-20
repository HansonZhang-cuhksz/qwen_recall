prompt = "This image is a screenshot. Describe what the user is doing in detail. Respond in the format that the information in the screenshot can be easily retrieved from your response. Respond sentence by sentence, use newline to separate sentences, do not use bully points."

from infer_qwen import infer
from utils import memory_dir

import shared

import time

def decode_task(infer_driver, infer_lock):
    global memory_dir

    while True:
        if not shared.capture_to_decode_1:
            time.sleep(1)
            continue

        timestamp = shared.capture_to_decode_1
        shared.capture_to_decode_1 = None

        start_time = time.time()

        pth = memory_dir / "images" / f"{timestamp}.png"
        if not pth.exists():
            raise FileNotFoundError(f"Image file {pth} does not exist.")
    
        messages = [
            {"role": "system", "content": "You are a helpful assistant that describes screenshots in detail."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://" + str(pth),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        with infer_lock:
            output = infer(messages, infer_driver)

        shared.capture_to_decode_2 = time.time() - start_time
        shared.decode_to_embed_1 = (timestamp, output[0])

        time.sleep(0.1)