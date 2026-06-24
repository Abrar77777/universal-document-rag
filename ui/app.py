import uuid

import pandas as pd
import requests
import streamlit as st  # pyright: ignore[reportMissingImports]


API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Intelligent RAG Assistant", layout="wide")
st.title("Intelligent Universal Document RAG Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "analytics" not in st.session_state:
    st.session_state.analytics = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()

with st.sidebar:
    st.subheader("Retrieval settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)
    use_web = st.toggle("Use web search fallback", value=False)
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
                response = requests.post(
                    f"{API}/upload",
                    params={"session_id": st.session_state.session_id},
                    files={"file": (file_obj.name, file_obj.getvalue())},
                    timeout=120,
                )
                data = response.json()
                if response.status_code != 200 or data.get("error"):
                    st.error(f"Upload failed for {file_obj.name}: {data.get('error', response.text)}")
                else:
                    st.session_state.uploaded_names.add(upload_key)
                    if data.get("analytics"):
                        st.session_state.analytics.append(data["analytics"])
                    st.success(f"Indexed {file_obj.name}")
            except Exception as exc:
                st.error(f"Backend not reachable: {exc}")

    if st.session_state.analytics:
        st.subheader("Numeric profile")
        for profile in st.session_state.analytics:
            st.markdown(f"**{profile.get('file', 'uploaded file')}**")
            for sheet in profile.get("sheets", []):
                st.write(
                    f"{sheet['name']}: {sheet['rows']} rows, {sheet['columns']} columns"
                )
                if sheet.get("numeric_summary"):
                    st.dataframe(pd.DataFrame(sheet["numeric_summary"]).T, use_container_width=True)

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
                response = requests.post(
                    f"{API}/ask",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                        "use_web": use_web,
                        "top_k": top_k,
                    },
                    timeout=180,
                )
                data = response.json()
                if response.status_code != 200 or data.get("error"):
                    st.error(data.get("error", response.text))
                    st.stop()

                st.markdown("### Answer")
                st.write(data.get("answer", "No answer returned."))

                metric_cols = st.columns(3)
                metric_cols[0].metric("Confidence", data.get("confidence", 0))
                metric_cols[1].metric("Chunks used", len(data.get("retrieved_chunks", [])))
                metric_cols[2].metric("Web used", "Yes" if data.get("web_used") else "No")

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
            except Exception as exc:
                st.error(f"Backend not reachable: {exc}")
