from transformers import TextStreamer

class StreamlitTextStreamer(TextStreamer):
    def __init__(self, tokenizer, placeholder, **kwargs):
        super().__init__(tokenizer, skip_special_tokens=True, **kwargs)
        self.placeholder = placeholder
        self.generated_text = ""

    def on_text(self, text, **kwargs):
        self.generated_text += text
        self.placeholder.markdown(self.generated_text)

    def end(self):
        pass