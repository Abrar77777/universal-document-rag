import streamlit as st
import requests
import uuid

API = "http://127.0.0.1:8000"

st.title("🧠 Multi-Document RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

use_web = st.checkbox("Use Internet (Web Search)", value=True)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        requests.post(f"{API}/upload", files={"file": f})
    st.success("Files indexed successfully")

question = st.text_input("Ask a question")

if st.button("Ask"):
    response = requests.post(
        f"{API}/ask",
        json={
            "question": question,
            "session_id": st.session_state.session_id,
            "use_web": use_web
        }
    ).json()

    st.markdown("### ✅ Answer")
    st.write(response["answer"])

    st.markdown("### 📌 Sources")
    for c in response["citations"]:
        st.write(f"- {c}")

    st.markdown(f"### 🔍 Confidence Score: **{response['confidence']}**")

    if response["web_used"]:
        st.info("Internet sources were used.")


