from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from streamer import StreamlitTextStreamer
import streamlit as st

from pathlib import Path

model_dir = Path(__file__).parent / 'qwenVL'

@st.cache_resource
def init(force_cpu=False):
    print("Qwen2.5 VL model loading...")
    if force_cpu:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="cpu", attn_implementation="sdpa"
        )
    else:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto", quantization_config=quantization_config, attn_implementation="sdpa",
        )
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=True)
    print("Qwen2.5 VL model loaded successfully.")
    return model, processor

def infer(msg, model_processor, use_streamer=False):
    # print(msg)
    model, processor = model_processor

    # Preparation for inference
    text = processor.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    if use_streamer:
        placeholder = st.empty()
        streamer = StreamlitTextStreamer(processor.tokenizer, placeholder)
        generated_ids = model.generate(**inputs, max_new_tokens=1024, streamer=streamer)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text