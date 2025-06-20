if __name__ == "__main__":
    import streamlit as st
    import threading
    import os
    import markdown

    from utils import memory_dir, image_dir, descriptions_file
    import interact

    st.title("Qwen Recall")

    st.session_state.message_lock = threading.Lock()

    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        import infer_qwen
        import embedding
        import reranking
        import image_diff

        with open(descriptions_file, "w", encoding="utf-8") as f:
            f.write("")
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        for file in os.listdir(memory_dir / "temp"):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        st.session_state.infer_driver = infer_qwen.init(force_cpu=False)
        st.session_state.embed_driver = embedding.init(force_cpu=True)
        st.session_state.rerank_driver = reranking.init(force_cpu=True)
        st.session_state.image_diff_driver = image_diff.init(force_cpu=True)

        from capture import capture_task
        from decode import decode_task
        from embed_task import embed_task

        st.session_state.infer_lock = threading.Lock()
        st.session_state.embed_lock = threading.Lock()

        if "capture_thread" not in st.session_state:
            st.session_state.capture_thread = threading.Thread(target=capture_task, args=(st.session_state.image_diff_driver,))
            st.session_state.capture_thread.start()
            print("Capture thread on ", st.session_state.capture_thread.native_id)
        if "decode_thread" not in st.session_state:
            st.session_state.decode_thread = threading.Thread(target=decode_task, args=(st.session_state.infer_driver, st.session_state.infer_lock))
            st.session_state.decode_thread.start()
            print("Decode thread on ", st.session_state.decode_thread.native_id)
        if "embed_thread" not in st.session_state:
            st.session_state.embed_thread = threading.Thread(target=embed_task, args=(st.session_state.embed_driver,st.session_state.embed_lock))
            st.session_state.embed_thread.start()
            print("Embed thread on ", st.session_state.embed_thread.native_id)

    def ask(attachments, question):
        return interact.interact(
            st.session_state.infer_driver, st.session_state.embed_driver, st.session_state.rerank_driver, st.session_state.infer_lock, st.session_state.embed_lock, st.session_state.message_lock, attachments, question
        )
    
    def get_query(content):
        start_idx = content.find("User query: ") + len("User query: ")
        end_idx = content.find("Background knowledge:")
        if end_idx == -1:
            end_idx = len(content)
        return content[start_idx:end_idx].strip()

    def update_history(chat_container):
        for message in interact.messages:
            with st.session_state.message_lock:
                if message["role"] == "user":
                    content = f"**You:** {get_query(message['content'][0]['text'])}"
                    html = markdown.markdown(content)
                    chat_container.markdown(
                        f"""
                        <div style="background-color: #101020; padding: 20px; border-radius: 10px;">
                            <p>{html}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif message["role"] == "assistant":
                    content = f"**Qwen:** {message['content'][0]['text']}"
                    html = markdown.markdown(content)
                    chat_container.markdown(
                        f"""
                        <div style="background-color: #102010; padding: 20px; border-radius: 10px;">
                            <p>{html}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Create a scrollable chat container
    chat_container = st.container()
    with chat_container:
        update_history(st)

    # Place the chat bar at the bottom
    with st.form("chat_form", clear_on_submit=True):
        attached_files = st.file_uploader("Upload files", type=["png", "jpg", "jpeg", "mp4", "mkv", "webm"], accept_multiple_files=True, key="file_uploader")
        file_paths = []
        for attached_file in attached_files:
            pth = memory_dir / "temp" / attached_file.name
            with open(pth, "wb") as f:
                f.write(attached_file.getvalue())
            file_paths.append(str(pth))

        user_input = st.text_input("Ask Qwen Recall", "")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            ask(file_paths, user_input)
            with chat_container:
                update_history(st)
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)