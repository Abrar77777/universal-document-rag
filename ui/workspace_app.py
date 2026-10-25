import json
import logging
import os
import uuid

import pandas as pd
import streamlit as st

from utils.workspace import answer_report, discard_documents, fingerprint, ingest

logger = logging.getLogger(__name__)


def configure_secrets():
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "GROQ_MODEL",
                "RAG_EMBEDDING_BACKEND", "RAG_EMBEDDING_MODEL"):
        try:
            value = st.secrets.get(key)
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            value = None
        if value:
            os.environ[key] = str(value)


def reset_workspace():
    from rag.rag_pipeline import CHAT_HISTORY

    discard_documents(st.session_state.get("documents", {}).values())
    previous = st.session_state.get("session_id", "")
    for key in list(CHAT_HISTORY):
        if previous and key.startswith(previous + ":"):
            del CHAT_HISTORY[key]
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.documents = {}
    st.session_state.answers = []
    st.session_state.pipeline = None
    st.session_state.scope = None
    st.session_state.lab_result = None
    st.session_state.feedback_result = None
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    for key in list(st.session_state):
        if key.startswith("ui_"):
            del st.session_state[key]


def csv_bytes(frame):
    safe = frame.copy()
    for column in safe.select_dtypes(include=["object", "str"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
        )
    return safe.to_csv(index=False).encode("utf-8-sig")


def render_evidence(answer):
    chunks = answer.get("retrieved_chunks", [])
    if not chunks:
        return
    with st.expander(f"Evidence ledger ({len(chunks)} passages)"):
        table = [{"Source": item["source"], "Chunk": item["chunk_id"] + 1,
                  "Rank score": item["score"], "Vector rank": item.get("vector_rank"),
                  "Keyword rank": item.get("lexical_rank")} for item in chunks]
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        for item in chunks:
            st.markdown(f"**[{item.get('citation_id', '')}] {item['source']} / chunk {item['chunk_id'] + 1}**")
            st.text(item["text"])


def render_ask(documents, top_k, mode, use_web, evidence_only):
    st.subheader("Ask your documents")
    with st.form("ask_form"):
        question = st.text_area("Question", max_chars=4000, height=100,
                                placeholder="What is the refund policy, and which evidence supports it?")
        submitted = st.form_submit_button("Ask", type="primary", disabled=not documents)
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence..."):
                    answer = st.session_state.pipeline.answer(
                        question.strip(), st.session_state.session_id, use_web, top_k,
                        [document["profile"] for document in documents if document["profile"]],
                        retrieval_mode=mode, generate=not evidence_only,
                    )
                st.session_state.answers.append({"question": question.strip(), "result": answer})
                st.session_state.answers = st.session_state.answers[-10:]
            except Exception as exc:
                logger.warning("Question failed (%s).", type(exc).__name__)
                st.error("The request could not be completed. Check the selected documents and try again.")
    if not st.session_state.answers:
        st.info("No answers in this document scope yet.")
        return
    answers = st.session_state.answers
    entry = answers[-1]
    st.markdown(f"#### {entry['question']}")
    result = entry["result"]
    st.markdown(result["answer"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence passages", result["evidence_count"])
    c2.metric("Retrieval", f"{result['retrieval_ms']:,.0f} ms")
    c3.metric("Total time", f"{result['total_ms'] / 1000:.2f} s")
    st.caption(f"{result['generation_status'].title()} | {result['retrieval_mode'].title()} retrieval | "
               f"Web evidence: {'yes' if result['web_used'] else 'no'}")
    if result.get("llm_error"):
        st.warning("AI analysis is temporarily unavailable. This response contains source excerpts only.")
    render_evidence(result)
    st.download_button("Download answer report", answer_report(entry["question"], result),
                       file_name="document_report.md", mime="text/markdown", icon=":material/download:")
    if len(answers) > 1:
        with st.expander(f"Earlier questions ({len(answers) - 1})"):
            for previous in reversed(answers[:-1]):
                st.markdown(f"**{previous['question']}**")
                st.markdown(previous["result"]["answer"])


def select_table(documents, prefix):
    choices = [(document, name) for document in documents for name in document["tables"]]
    if not choices:
        st.info("No spreadsheet data in the selected documents.")
        return None, None, None
    selected = st.selectbox("Dataset", range(len(choices)),
                            format_func=lambda i: f"{choices[i][0]['name']} / {choices[i][1]}", key=f"ui_{prefix}_dataset")
    document, name = choices[selected]
    return document, name, document["tables"][name]


def render_data(documents):
    st.subheader("Data explorer")
    document, name, frame = select_table(documents, "data")
    if frame is None:
        return
    key = f"ui_chart_{document['id']}_{name}"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(frame):,}")
    c2.metric("Columns", len(frame.columns))
    c3.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if frame.empty or not len(frame.columns):
        st.info("This sheet has no rows to analyze.")
        return
    st.markdown("#### Chart builder")
    chart_type = st.radio("Chart", ["Bar", "Line", "Scatter", "Distribution"], horizontal=True, key=key + "type")
    if not numeric:
        st.info("No numeric columns found. Showing category counts.")
        column = st.selectbox("Category", frame.columns.tolist(), key=key + "category")
        counts = frame[column].fillna("(missing)").astype(str).value_counts().head(30).rename_axis("category").reset_index(name="count")
        st.bar_chart(counts, x="category", y="count", color="#28b5a0")
    else:
        left, middle, right = st.columns(3)
        value = middle.selectbox("Value", numeric, key=key + "value")
        if chart_type == "Distribution":
            bins = left.slider("Bins", 5, 40, 10, key=key + "bins")
            series = pd.to_numeric(frame[value], errors="coerce").replace([float("inf"), -float("inf")], float("nan")).dropna()
            if series.empty:
                st.info("No finite numeric values in this column.")
            else:
                counts = pd.cut(series, bins=bins).value_counts(sort=False).rename_axis("range").reset_index(name="count")
                counts["range"] = counts["range"].astype(str)
                st.bar_chart(counts, x="range", y="count", color="#28b5a0")
        elif chart_type == "Scatter":
            x = left.selectbox("X axis", numeric, key=key + "scatter_x")
            if x == value:
                st.info("Choose different numeric columns for the X axis and value.")
            else:
                points = frame[[x, value]].dropna()
                if len(points) > 2000:
                    points = points.sample(2000, random_state=42)
                    st.caption("2,000-row reproducible sample")
                st.scatter_chart(points, x=x, y=value, color="#28b5a0")
        else:
            dimensions = [column for column in frame.columns if column != value]
            if not dimensions:
                st.info("A second column is needed for grouped charts.")
            else:
                dimension = left.selectbox("Group by", dimensions, key=key + "dimension")
                aggregation = right.selectbox("Aggregation", ["sum", "mean", "median", "count"], key=key + "agg")
                grouped = frame.groupby(dimension, dropna=False)[value].agg(aggregation).reset_index()
                if chart_type == "Line":
                    grouped = grouped.sort_values(dimension)
                    st.line_chart(grouped, x=dimension, y=value, color="#28b5a0")
                else:
                    grouped = grouped.sort_values(value, ascending=False)
                    if len(grouped) > 30:
                        st.caption("Top 30 groups by selected aggregation")
                    st.bar_chart(grouped.head(30), x=dimension, y=value, color="#28b5a0", sort=False)
                st.download_button("Download grouped data", csv_bytes(grouped), "grouped_data.csv", "text/csv", key=key + "export")
    with st.expander("Data quality and statistics"):
        sheet = next((sheet for sheet in document["profile"].get("sheets", []) if sheet["name"] == name), {})
        missing = frame.isna().sum().rename_axis("column").reset_index(name="missing")
        st.dataframe(missing, hide_index=True, use_container_width=True)
        if sheet.get("numeric_summary"):
            st.dataframe(pd.DataFrame(sheet["numeric_summary"]).T, use_container_width=True)
        for label, field in (("Correlations (not causation)", "correlations"), ("Potential outliers (1.5 x IQR)", "outliers")):
            if sheet.get(field):
                st.markdown(f"**{label}**")
                st.dataframe(pd.DataFrame(sheet[field]), hide_index=True, use_container_width=True)
    st.markdown("#### Data preview")
    st.dataframe(frame.head(500), hide_index=True, use_container_width=True)
    st.caption(f"First {min(len(frame), 500):,} of {len(frame):,} rows; aggregate charts use the full sheet.")


def render_feedback(documents):
    from utils.text_analytics import analyze_feedback_text

    st.subheader("Feedback intelligence")
    if not documents:
        st.info("No documents selected.")
        return
    selected = st.selectbox("Feedback source", range(len(documents)), format_func=lambda i: documents[i]["name"], key="ui_feedback_source")
    document = documents[selected]
    signature = document["id"]
    responses = None
    if document["tables"]:
        sheet = st.selectbox("Feedback sheet", list(document["tables"]), key=f"ui_feedback_sheet_{signature}")
        frame = document["tables"][sheet]
        columns = frame.select_dtypes(exclude="number").columns.tolist()
        if not columns:
            st.info("This sheet contains no text columns.")
            return
        preferred = next((column for column in columns if any(word in column.lower() for word in ("feedback", "review", "comment", "response"))), columns[0])
        column = st.selectbox("Response column", columns, index=columns.index(preferred), key=f"ui_feedback_column_{signature}_{sheet}")
        responses = frame[column].dropna().astype(str).tolist()
        signature += f"/{sheet}/{column}"
    if st.button("Analyze feedback", type="primary"):
        with st.spinner("Scoring responses..."):
            result = analyze_feedback_text(document["text"], document["name"], responses=responses)
        st.session_state.feedback_result = (signature, result)
    stored = st.session_state.feedback_result
    if not stored or stored[0] != signature:
        return
    profile = stored[1]
    st.caption(f"{profile['method']} | Unit: {profile['unit']} | Sentiment signals, not verified customer intent")
    counts = profile["sentiment"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Responses", f"{profile['responses']:,}")
    c2.metric("Positive", counts.get("positive", 0))
    c3.metric("Negative", counts.get("negative", 0))
    c4.metric("Net sentiment", f"{profile['net_sentiment']:+.2f}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Sentiment distribution**")
        st.bar_chart(pd.DataFrame([{"sentiment": label, "count": counts.get(label, 0)} for label in ("positive", "neutral", "negative")]), x="sentiment", y="count", color="#28b5a0")
    with right:
        st.markdown("**Theme keyword mentions**")
        if profile["themes"]:
            st.bar_chart(pd.DataFrame(profile["themes"], columns=["theme", "mentions"]), x="theme", y="mentions", color="#dc8b68")
    st.markdown("#### Response explorer")
    sentiment = st.selectbox("Sentiment", ["All", "negative", "neutral", "positive"], key="ui_sentiment_filter")
    query = st.text_input("Contains text", key="ui_feedback_search")
    scored = pd.DataFrame(profile["scored_responses"], columns=["response", "text", "sentiment", "score"])
    filtered = scored if sentiment == "All" else scored[scored.sentiment == sentiment]
    if query:
        filtered = filtered[filtered.text.str.contains(query, case=False, regex=False)]
    st.dataframe(filtered, hide_index=True, use_container_width=True)
    st.download_button("Download scored responses", csv_bytes(filtered), "feedback_analysis.csv", "text/csv", icon=":material/download:")
    with st.expander("Keywords and method limits"):
        st.dataframe(pd.DataFrame(profile["keywords"], columns=["keyword", "mentions"]), hide_index=True)
        st.caption("English rule-based baseline. Sarcasm, mixed sentiment, domain language, and other languages require human review. Themes are keyword matches, not learned topics.")


def render_lab(documents, top_k):
    from evaluation.evaluate_rag import _precision_at_k, _recall_at_k, _mrr

    st.subheader("Retrieval lab")
    with st.form("retrieval_lab"):
        query = st.text_input("Evaluation question")
        expected = st.multiselect("Expected source documents", list(dict.fromkeys(document["name"] for document in documents)))
        run = st.form_submit_button("Compare retrieval", disabled=not documents)
    if run:
        if not query.strip() or not expected:
            st.warning("Enter a question and label its expected sources.")
        else:
            comparisons = []
            for mode in ("vector", "hybrid"):
                results = st.session_state.pipeline.retriever.search(query, top_k, mode)
                comparisons.append({"mode": mode, "precision_at_k": _precision_at_k(results, set(expected), top_k),
                                    "source_recall_at_k": _recall_at_k(results, set(expected), top_k),
                                    "mrr": _mrr(results, set(expected)), "results": results})
            st.session_state.lab_result = {"question": query, "top_k": top_k, "expected_sources": expected, "comparisons": comparisons}
    result = st.session_state.lab_result
    if result:
        st.caption(f"Single labeled query | k = {result['top_k']} | Source-level relevance; not an answer accuracy score")
        st.dataframe(pd.DataFrame([{key: value for key, value in row.items() if key != "results"} for row in result["comparisons"]]), hide_index=True, use_container_width=True)
        for column, comparison in zip(st.columns(2), result["comparisons"]):
            with column:
                st.markdown(f"**{comparison['mode'].title()}**")
                for rank, item in enumerate(comparison["results"], 1):
                    st.write(f"{rank}. {item['source']} / chunk {item['chunk_id'] + 1}")
        st.download_button("Download evaluation", json.dumps(result, indent=2), "retrieval_evaluation.json", "application/json")


def main():
    st.set_page_config(page_title="Document Intelligence Workspace", layout="wide")
    configure_secrets()
    if "documents" not in st.session_state:
        reset_workspace()
    with st.sidebar:
        st.subheader("Workspace")
        uploads = st.file_uploader("Documents", type=["pdf", "docx", "txt", "md", "csv", "xlsx", "xls", "json"],
                                   accept_multiple_files=True, key=f"uploads_{st.session_state.upload_generation}")
        replace = st.checkbox("Replace current documents", value=True)
        st.caption("20 MB per file / 10 documents per workspace")
        if st.button("Index files", disabled=not uploads, use_container_width=True):
            count = len(uploads) + (0 if replace else len(st.session_state.documents))
            if count > 10:
                st.error("A workspace can contain up to 10 documents.")
            else:
                if replace:
                    reset_workspace()
                for upload in uploads:
                    try:
                        content = upload.getvalue()
                        if fingerprint(upload.name, content) not in st.session_state.documents:
                            with st.spinner(f"Indexing {upload.name}..."):
                                document = ingest(upload.name, content)
                                st.session_state.documents[document["id"]] = document
                            st.session_state["ui_scope"] = list(st.session_state.documents)
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        logger.warning("Ingestion failed (%s).", type(exc).__name__)
                        st.error(f"Could not read {upload.name}. Check that the file is valid and contains readable text.")
        st.divider()
        registry = st.session_state.documents
        selected = st.multiselect("Active documents", list(registry), default=list(registry),
                                  format_func=lambda key: registry[key]["name"], key="ui_scope")
        top_k = st.slider("Evidence passages", 1, 10, 5)
        mode = st.radio("Retrieval", ["hybrid", "vector"], format_func=str.title, horizontal=True)
        evidence_only = st.toggle("Evidence only", value=False)
        use_web = st.toggle("Include web evidence", value=False,
                            help="Sends your question to a third-party search service.")
        if st.button("Clear workspace", icon=":material/delete:", use_container_width=True):
            reset_workspace()
            st.rerun()
    documents = [registry[key] for key in selected]
    scope = tuple(sorted(selected))
    if scope != st.session_state.scope:
        from rag.rag_pipeline import RAGPipeline

        st.session_state.pipeline = RAGPipeline([document["index_dir"] for document in documents])
        st.session_state.scope = scope
        st.session_state.answers = []
        st.session_state.lab_result = None
        st.session_state.feedback_result = None
    st.title("Document Intelligence Workspace")
    c1, c2, c3 = st.columns(3)
    c1.metric("Active documents", len(documents))
    c2.metric("Indexed chunks", sum(document["chunks"] for document in documents))
    c3.metric("Table rows", f"{sum(len(frame) for document in documents for frame in document['tables'].values()):,}")
    if documents:
        backends = ", ".join(sorted({document["embedding_backend"] for document in documents}))
        st.caption(f"Scope: {', '.join(document['name'] for document in documents)} | Embeddings: {backends}")
    else:
        st.info("No documents in this workspace yet.")
    ask, data, feedback, lab = st.tabs(["Ask & evidence", "Data explorer", "Feedback intelligence", "Retrieval lab"])
    with ask:
        render_ask(documents, top_k, mode, use_web, evidence_only)
    with data:
        render_data(documents)
    with feedback:
        render_feedback(documents)
    with lab:
        render_lab(documents, top_k)
