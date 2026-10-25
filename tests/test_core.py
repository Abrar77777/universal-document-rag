import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from embeddings.build_index_from_file import build_index
from evaluation.evaluate_rag import _precision_at_k, _recall_at_k, _mrr
from rag.rag_pipeline import RAGPipeline, CHAT_HISTORY
from rag.retriever import Retriever
from utils.tabular_analyzer import analyze_tabular_file
from utils.text_analytics import analyze_feedback_text
from utils.text_cleaner import clean_text
from utils.text_splitter import split_text
from utils.workspace import fingerprint, ingest, discard_documents


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"RAG_EMBEDDING_BACKEND": "hashing", "GEMINI_API_KEY": "", "GROQ_API_KEY": ""})
        self.env.start()
        CHAT_HISTORY.clear()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def index(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return build_index(str(path), name, source_name=name, index_root=str(self.root / "indexes"))

    def test_hybrid_ranks_source_and_records_signals(self):
        policy = self.index("policy.txt", "Refund requests must arrive within thirty days. Customers need a receipt.")
        unrelated = self.index("roster.txt", "Students study mathematics in the science department.")
        results = Retriever([policy, unrelated]).search("refund receipt", mode="hybrid")
        self.assertEqual(results[0]["source"], "policy.txt")
        self.assertGreater(results[0]["lexical_score"], 0)
        self.assertEqual(results[0]["lexical_rank"], 1)

    def test_scope_excludes_old_document(self):
        old = self.index("old.txt", "Refunds require a receipt.")
        new = self.index("new.txt", "Revenue increased in the North region.")
        results = Retriever([new]).search("revenue")
        self.assertEqual({item["source"] for item in results}, {"new.txt"})
        self.assertNotIn(old, [item["index_dir"] for item in results])

    def test_unrelated_hash_collisions_are_not_evidence(self):
        index = self.index("policy.txt", "Refund receipt thirty days.")
        self.assertEqual(Retriever([index]).search("quantum astronomy nebula"), [])

    def test_index_metadata_wins_over_environment(self):
        index = self.index("policy.txt", "Refund receipt thirty days.")
        with patch.dict(os.environ, {"RAG_EMBEDDING_BACKEND": "sentence-transformers"}):
            retriever = Retriever([index])
            self.assertEqual(next(iter(retriever.models))[0], "hashing")

    def test_gemini_failure_tries_groq(self):
        pipeline = RAGPipeline([self.index("policy.txt", "Refunds take thirty days.")])
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder", "GROQ_API_KEY": "test-placeholder"}), \
             patch.object(pipeline, "_generate_with_gemini", side_effect=RuntimeError("secret-not-for-output")), \
             patch.object(pipeline, "_generate_with_groq", return_value="1. Thirty days. [S1]"):
            result = pipeline.answer("Refunds?", "test", False)
        self.assertEqual(result["llm_provider"], "groq")
        self.assertFalse(result["llm_error"])
        self.assertNotIn("secret-not-for-output", str(result))

    def test_whitespace_response_falls_back(self):
        pipeline = RAGPipeline([self.index("policy.txt", "Refunds take thirty days.")])
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), \
             patch.object(pipeline, "_generate_with_gemini", return_value="  \n "):
            result = pipeline.answer("Refunds?", "test", False)
        self.assertEqual(result["generation_status"], "extractive")
        self.assertIn("Retrieved evidence", result["answer"])
        self.assertNotIn("API_KEY", result["answer"])

    def test_citations_are_sent_in_prompt(self):
        pipeline = RAGPipeline([self.index("policy.txt", "Refunds take thirty days.")])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), \
             patch.object(pipeline, "_generate_with_groq", return_value="Thirty days. [S1]") as generator:
            pipeline.answer("Refunds?", "test", False)
        messages = generator.call_args.args[0]
        self.assertIn("[S1] policy.txt", messages[-1]["content"])
        self.assertIn("untrusted", messages[0]["content"])

    def test_no_evidence_skips_provider(self):
        pipeline = RAGPipeline([])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(pipeline, "_generate_with_groq") as generator:
            result = pipeline.answer("Refunds?", "test", False)
        generator.assert_not_called()
        self.assertEqual(result["evidence_count"], 0)

    def test_evidence_only_never_calls_provider(self):
        pipeline = RAGPipeline([self.index("policy.txt", "Refunds take thirty days.")])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}), patch.object(pipeline, "_generate_with_groq") as generator:
            pipeline.answer("Refunds?", "test", False, generate=False)
        generator.assert_not_called()

    def test_sentiment_preserves_rows_and_negation(self):
        profile = analyze_feedback_text("", responses=["This is not good.", "Excellent! I love it.", "", "It arrived Tuesday."])
        self.assertEqual(profile["responses"], 3)
        self.assertEqual(profile["scored_responses"][0]["sentiment"], "negative")
        self.assertEqual(profile["scored_responses"][1]["sentiment"], "positive")
        self.assertEqual(profile["unit"], "rows")

    def test_empty_numeric_column_does_not_break_profile(self):
        path = self.root / "empty.csv"
        pd.DataFrame({"amount": [float("nan"), float("nan")], "region": ["A", "B"]}).to_csv(path, index=False)
        profile = analyze_tabular_file(str(path))
        self.assertEqual(profile["sheets"][0]["rows"], 2)
        self.assertEqual(profile["sheets"][0]["numeric_summary"]["amount"]["sum"], None)

    def test_correlations_are_not_duplicated(self):
        path = self.root / "numbers.csv"
        pd.DataFrame({"revenue": [1, 2, 3], "orders": [2, 4, 6]}).to_csv(path, index=False)
        profile = analyze_tabular_file(str(path))
        self.assertEqual(len(profile["sheets"][0]["correlations"]), 1)

    def test_chunking_validation_and_paragraphs(self):
        self.assertIn("\n", clean_text("First paragraph.\nSecond paragraph."))
        for size, overlap in [(0, 0), (10, -1), (10, 10)]:
            with self.assertRaises(ValueError):
                split_text("abc", size, overlap)
        self.assertGreater(len(split_text("word " * 500, 100, 20)), 1)

    def test_fingerprint_uses_content_not_size(self):
        self.assertNotEqual(fingerprint("a.txt", b"one"), fingerprint("a.txt", b"two"))

    def test_source_metrics(self):
        retrieved = [{"source": "a"}, {"source": "b"}]
        self.assertEqual(_precision_at_k(retrieved, {"b"}, 5), 0.2)
        self.assertEqual(_recall_at_k(retrieved, {"a", "c"}, 5), 0.5)
        self.assertEqual(_mrr(retrieved, {"b"}), 0.5)

    def test_api_validates_query_and_upload(self):
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            for question, top_k in [(" ", 5), ("Hello", 0), ("Hello", 1000)]:
                self.assertEqual(client.post("/ask", json={"question": question, "top_k": top_k, "session_id": "test"}).status_code, 422)
            self.assertEqual(client.post("/upload", params={"session_id": "test"}, files={"file": ("bad.exe", b"bad")}).status_code, 415)


class WorkspaceUITests(unittest.TestCase):
    def test_upload_backed_workspace_controls(self):
        from streamlit.testing.v1 import AppTest

        with patch.dict(os.environ, {"RAG_EMBEDDING_BACKEND": "hashing", "GEMINI_API_KEY": "", "GROQ_API_KEY": ""}):
            document = ingest("test_feedback.csv", b"region,revenue,feedback\nNorth,10,Excellent service\nSouth,20,Not good at all\n")
            app_test = AppTest.from_file("streamlit_app.py").run(timeout=60)
            try:
                self.assertFalse(app_test.exception)
                app_test.session_state["documents"][document["id"]] = document
                app_test.run(timeout=60)
                app_test.multiselect(key="ui_scope").set_value([document["id"]]).run(timeout=60)
                self.assertFalse(app_test.exception)
                self.assertEqual(len(app_test.tabs), 4)
                self.assertNotIn("Load demo workspace", [button.label for button in app_test.button])
                next(toggle for toggle in app_test.toggle if toggle.label == "Evidence only").set_value(True)
                next(area for area in app_test.text_area if area.label == "Question").set_value("revenue")
                next(button for button in app_test.button if button.label == "Ask").click().run(timeout=60)
                self.assertFalse(app_test.exception)
                self.assertEqual(len(app_test.session_state["answers"]), 1)
                self.assertGreater(app_test.session_state["answers"][0]["result"]["evidence_count"], 0)
                next(slider for slider in app_test.slider if slider.label == "Evidence passages").set_value(3).run()
                self.assertEqual(len(app_test.session_state["answers"]), 1)
                next(button for button in app_test.button if button.label == "Analyze feedback").click().run()
                self.assertFalse(app_test.exception)
                self.assertEqual(app_test.session_state["feedback_result"][1]["responses"], 2)
                next(button for button in app_test.button if button.label == "Clear workspace").click().run()
                self.assertFalse(app_test.exception)
                self.assertEqual(app_test.session_state["documents"], {})
                self.assertFalse(Path(document["path"]).exists())
            finally:
                discard_documents([document])


if __name__ == "__main__":
    unittest.main()
