import os
import pickle
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


def configure_secrets():
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "RAG_EMBEDDING_BACKEND", "RAG_EMBEDDING_MODEL"):
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value and not os.getenv(key):
            os.environ[key] = str(value)


def upload_direct(file_obj):
    from embeddings.build_index_from_file import build_index

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    safe_name = Path(file_obj.name or "uploaded_file").name
    file_path = upload_dir / f"{file_id}_{safe_name}"
    file_path.write_bytes(file_obj.getvalue())

    index_dir = build_index(str(file_path), file_id, source_name=safe_name)
    st.session_state.index_dirs.append(index_dir)

    analytics_path = Path(index_dir) / "analytics.pkl"
    if analytics_path.exists():
        with open(analytics_path, "rb") as f:
            st.session_state.analytics.append(pickle.load(f))


def ask_direct(question: str, session_id: str, use_web: bool, top_k: int):
    from rag.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(st.session_state.index_dirs)
    return pipeline.answer(
        question=question,
        session_id=session_id,
        use_web=use_web,
        top_k=top_k,
        analytics_profiles=st.session_state.analytics,
    )


def render_numeric_profile():
    if not st.session_state.analytics:
        return

    st.subheader("Numeric profile")
    for profile in st.session_state.analytics:
        st.markdown(f"**{profile.get('file', 'uploaded file')}**")
        for sheet in profile.get("sheets", []):
            st.write(f"{sheet['name']}: {sheet['rows']} rows, {sheet['columns']} columns")
            if sheet.get("numeric_summary"):
                st.dataframe(pd.DataFrame(sheet["numeric_summary"]).T, use_container_width=True)


def render_answer(data: dict):
    st.markdown("### Answer")
    st.write(data.get("answer", "No answer returned."))

    metric_cols = st.columns(3)
    metric_cols[0].metric("Confidence", data.get("confidence", 0))
    metric_cols[1].metric("Chunks used", len(data.get("retrieved_chunks", [])))
    metric_cols[2].metric("Web used", "Yes" if data.get("web_used") else "No")

    if data.get("llm_error"):
        st.warning("LLM generation failed, so the app returned a retrieved-context fallback answer.")

    with st.expander("Sources and retrieved chunks", expanded=True):
        for chunk in data.get("retrieved_chunks", []):
            st.markdown(
                f"**{chunk.get('source')} - chunk {chunk.get('chunk_id')} "
                f"(score {chunk.get('score')})**"
            )
            st.write(chunk.get("text", "")[:900])

    if data.get("chart_specs") and st.session_state.analytics:
        st.markdown("### Suggested visual analysis")
        for chart in data["chart_specs"][:4]:
            st.markdown(f"**{chart['title']}**")
            chart_df = pd.DataFrame(chart.get("data", []))
            if chart_df.empty:
                st.info(f"{chart['type']} chart recommended.")
                continue
            if chart["type"] == "line":
                st.line_chart(chart_df, x=chart["x"], y=chart["y"])
            elif chart["type"] == "bar":
                st.bar_chart(chart_df, x=chart["x"], y=chart["y"])
            else:
                st.bar_chart(chart_df, x=chart["x"], y="count")


configure_secrets()

st.set_page_config(page_title="Intelligent RAG Assistant", layout="wide")
st.title("Intelligent Universal Document RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "analytics" not in st.session_state:
    st.session_state.analytics = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()
if "index_dirs" not in st.session_state:
    st.session_state.index_dirs = []

with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)
    use_web = st.toggle("Use web search fallback", value=False)
    st.caption("Mode: Streamlit Cloud direct mode")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

left, right = st.columns([0.38, 0.62])

with left:
    st.subheader("Upload knowledge")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, CSV, Excel, JSON",
        type=["pdf", "txt", "md", "docx", "csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file_obj in uploaded_files:
            upload_key = f"{file_obj.name}:{file_obj.size}"
            if upload_key in st.session_state.uploaded_names:
                continue

            try:
                with st.spinner(f"Indexing {file_obj.name}..."):
                    upload_direct(file_obj)
                st.session_state.uploaded_names.add(upload_key)
                st.success(f"Indexed {file_obj.name}")
            except Exception as exc:
                st.error(f"Upload failed for {file_obj.name}: {exc}")

    render_numeric_profile()

with right:
    st.subheader("Ask across documents")
    question = st.text_area(
        "Question",
        placeholder="Example: What are the main risks? Summarize trends in revenue by region.",
        height=90,
    )

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence and generating answer..."):
                    data = ask_direct(
                        question=question,
                        session_id=st.session_state.session_id,
                        use_web=use_web,
                        top_k=top_k,
                    )
                render_answer(data)
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
