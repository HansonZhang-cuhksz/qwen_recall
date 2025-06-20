from transformers import TextStreamer
import streamlit as st

class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.text_placeholder = st.empty()
        self.accumulated_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.accumulated_text += text
        self.text_placeholder.markdown(self.accumulated_text, unsafe_allow_html=True)