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
    from utils.file_loader import load_file
    from utils.text_analytics import analyze_feedback_text

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

    raw_text = load_file(str(file_path))
    if raw_text.strip():
        text_profile = analyze_feedback_text(raw_text, source=safe_name)
        st.session_state.text_analytics.append(text_profile)


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

    st.subheader("Data analysis")
    for profile in st.session_state.analytics:
        st.markdown(f"**{profile.get('file', 'uploaded file')}**")
        for sheet in profile.get("sheets", []):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", sheet["rows"])
            c2.metric("Columns", sheet["columns"])
            c3.metric("Numeric columns", len(sheet.get("numeric_columns", [])))
            if sheet.get("numeric_summary"):
                st.markdown("**Numeric summary**")
                st.dataframe(pd.DataFrame(sheet["numeric_summary"]).T, use_container_width=True)
            if sheet.get("missing_percent"):
                missing_df = pd.DataFrame(
                    [{"column": k, "missing_percent": v} for k, v in sheet["missing_percent"].items()]
                )
                st.markdown("**Missing values**")
                st.bar_chart(missing_df, x="column", y="missing_percent")
            if sheet.get("top_categories"):
                with st.expander("Top categories", expanded=False):
                    for col, values in sheet["top_categories"].items():
                        cat_df = pd.DataFrame({"category": list(values.keys()), "count": list(values.values())})
                        st.markdown(f"**{col}**")
                        st.bar_chart(cat_df, x="category", y="count")
            if sheet.get("correlations"):
                corr_df = pd.DataFrame(sheet["correlations"])
                st.markdown("**Strongest correlations**")
                st.dataframe(corr_df, use_container_width=True)
            if sheet.get("outliers"):
                st.markdown("**Potential outliers**")
                st.dataframe(pd.DataFrame(sheet["outliers"]), use_container_width=True)


def render_text_analytics():
    if not st.session_state.text_analytics:
        return

    st.subheader("Customer feedback analytics")
    for profile in st.session_state.text_analytics:
        st.markdown(f"**{profile.get('source', 'uploaded text')}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Responses / sentences", profile.get("responses", 0))
        c2.metric("Analyzed words", profile.get("words", 0))
        c3.metric("Net sentiment", profile.get("net_sentiment", 0))

        sentiment_df = pd.DataFrame(
            [{"sentiment": k, "count": v} for k, v in profile.get("sentiment", {}).items()]
        )
        if not sentiment_df.empty:
            st.markdown("**Sentiment distribution**")
            st.bar_chart(sentiment_df, x="sentiment", y="count")

        col_a, col_b = st.columns(2)
        with col_a:
            keywords = pd.DataFrame(profile.get("keywords", []), columns=["keyword", "count"])
            if not keywords.empty:
                st.markdown("**Top keywords**")
                st.dataframe(keywords, use_container_width=True, hide_index=True)
        with col_b:
            themes = pd.DataFrame(profile.get("themes", []), columns=["theme", "count"])
            if not themes.empty:
                st.markdown("**Detected themes**")
                st.bar_chart(themes, x="theme", y="count")

        emotions = pd.DataFrame(profile.get("emotions", []), columns=["emotion", "count"])
        if not emotions.empty:
            st.markdown("**Emotion signals**")
            st.bar_chart(emotions, x="emotion", y="count")

        with st.expander("Pain points and praise", expanded=False):
            st.markdown("**Pain points**")
            pain_points = profile.get("pain_points", [])
            if pain_points:
                for item in pain_points[:6]:
                    st.write(f"- {item['text']}")
            else:
                st.caption("No strong negative signals detected.")
            st.markdown("**Praise**")
            praise_points = profile.get("praise_points", [])
            if praise_points:
                for item in praise_points[:6]:
                    st.write(f"- {item['text']}")
            else:
                st.caption("No strong positive signals detected.")


def render_answer(data: dict):
    st.markdown("### Answer")
    answer = data.get("answer", "No answer returned.")
    for line in str(answer).splitlines():
        clean = line.strip()
        if clean:
            st.markdown(clean)

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
            elif chart["type"] == "scatter":
                st.scatter_chart(chart_df, x=chart["x"], y=chart["y"])
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
if "text_analytics" not in st.session_state:
    st.session_state.text_analytics = []

with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)
    use_web = st.toggle("Use web search fallback", value=False)
    st.caption(f"Groq key: {'detected' if os.getenv('GROQ_API_KEY') else 'missing'}")
    st.caption("Mode: Streamlit Cloud direct mode")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

left, right = st.columns([0.38, 0.62])

with left:
    st.subheader("Upload knowledge")
    st.caption("Upload documents, spreadsheets, or thousands of feedback replies. The app will index content, analyze sentiment/themes, and profile data.")
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
    render_text_analytics()

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
