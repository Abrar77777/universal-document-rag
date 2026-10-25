import os
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from rag.provider_config import DEFAULT_GEMINI_MODEL, allow_groq_fallback, failure_category, gemini_model
from rag.rag_pipeline import RAGPipeline
from rag.tabular_qa import answer_table_question
from utils.workspace import answer_report, ingest, discard_documents


class TableQuestionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"Item": ["Notebook", "Desk lamp", "Pen"],
                                   "Price": [25.0, 420.0, 40.0], "Quantity": [32, 16, 10]})
        self.sources = [{"source": "inventory.xlsx", "sheet": "Inventory", "data": self.frame}]

    def test_screenshot_question_returns_item_and_price(self):
        result = answer_table_question("which item costs more", self.sources)
        self.assertEqual(result["generation_status"], "computed")
        self.assertIn("Desk lamp", result["answer"])
        self.assertIn("420", result["answer"])
        self.assertEqual(result["table_evidence"][0]["rows"][0]["data_row"], 2)

    def test_price_synonyms_and_minimum(self):
        for question, expected in [("Which is the most expensive item?", 420.0),
                                   ("Which item is cheapest?", 25.0),
                                   ("Which item is least expensive?", 25.0),
                                   ("What is the highest quantity?", 32.0)]:
            with self.subTest(question=question):
                result = answer_table_question(question, self.sources)
                self.assertEqual(result["table_evidence"][0]["result"], expected)

    def test_missing_values_and_ties(self):
        self.frame.loc[0, "Price"] = float("nan")
        self.frame.loc[2, "Price"] = 420.0
        result = answer_table_question("Highest price?", self.sources)
        evidence = result["table_evidence"][0]
        self.assertEqual(evidence["ties"], 2)
        self.assertEqual(evidence["valid_values"], 2)

    def test_filtered_questions_are_not_silently_global(self):
        for question in ["Which item costs more in South?", "Highest price under 100?",
                         "Which item costs more than the pen?", "Top 2 most expensive items",
                         "Which item is not the most expensive?", "Highest price per category?"]:
            with self.subTest(question=question):
                self.assertIsNone(answer_table_question(question, self.sources))

    def test_ambiguous_sheets_require_clarification(self):
        second = dict(self.sources[0], sheet="Other inventory")
        result = answer_table_question("Highest price?", self.sources + [second])
        self.assertEqual(result["generation_status"], "clarification")
        self.assertFalse(result["table_evidence"])

    def test_ambiguous_columns_require_clarification(self):
        self.frame["Cost"] = [10, 100, 20]
        result = answer_table_question("Which item costs more?", self.sources)
        self.assertEqual(result["generation_status"], "clarification")

    def test_no_finite_values(self):
        self.frame["Price"] = [float("inf"), float("nan"), -float("inf")]
        result = answer_table_question("Highest price?", self.sources)
        self.assertEqual(result["generation_status"], "clarification")

    def test_computation_skips_llm_and_web(self):
        pipeline = RAGPipeline([])
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder", "GROQ_API_KEY": "test-placeholder"}), \
             patch.object(pipeline, "_generate_with_gemini") as gemini, \
             patch.object(pipeline, "_generate_with_groq") as groq, \
             patch("rag.rag_pipeline.web_search") as web:
            result = pipeline.answer("which item costs more", "test", True, tabular_sources=self.sources)
        gemini.assert_not_called()
        groq.assert_not_called()
        web.assert_not_called()
        self.assertFalse(result["llm_error"])
        self.assertEqual(result["evidence_count"], 1)
        self.assertIn("Desk lamp", answer_report("which item costs more", result))

    def test_looks_at_whole_table_not_first_rows(self):
        frame = pd.DataFrame({"Item": [f"Row {i}" for i in range(1500)], "Price": range(1500)})
        sources = [dict(self.sources[0], data=frame)]
        result = answer_table_question("which item costs more", sources)
        self.assertEqual(result["table_evidence"][0]["rows"][0]["data_row"], 1500)
        self.assertEqual(result["table_evidence"][0]["result"], 1499)

    def test_excel_upload_with_analysis_sheet(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            self.frame.to_excel(writer, sheet_name="Inventory", index=False)
            pd.DataFrame({"Note": ["Inventory summary"]}).to_excel(writer, sheet_name="Analysis", index=False)
        with patch.dict(os.environ, {"RAG_EMBEDDING_BACKEND": "hashing"}):
            document = ingest("inventory_test.xlsx", buffer.getvalue())
        try:
            sources = [{"source": document["name"], "sheet": name, "data": data}
                       for name, data in document["tables"].items()]
            result = RAGPipeline([document["index_dir"]]).answer(
                "which item costs more", "test-excel", False, tabular_sources=sources)
            self.assertEqual(result["generation_status"], "computed")
            self.assertEqual(result["table_evidence"][0]["rows"][0]["values"]["Item"], "Desk lamp")
        finally:
            discard_documents([document])


class ProviderPolicyTests(unittest.TestCase):
    def test_free_tier_capable_default_and_no_paid_model_substitution(self):
        for requested in ["", "gemini-3.8-flash", "gemini-4-argon", "unexpected-pro-model"]:
            with self.subTest(requested=requested), patch.dict(os.environ, {"GEMINI_MODEL": requested}):
                self.assertEqual(gemini_model(), DEFAULT_GEMINI_MODEL)

    def test_other_provider_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(allow_groq_fallback())
        with patch.dict(os.environ, {"RAG_ALLOW_GROQ_FALLBACK": "true"}):
            self.assertTrue(allow_groq_fallback())

    def test_safe_error_categories(self):
        self.assertEqual(failure_category(TimeoutError("private response")), ("timeout", None))
        self.assertEqual(failure_category(SimpleNamespace(status_code=429)), ("rate_limit", 429))
        self.assertEqual(failure_category(SimpleNamespace(code=403)), ("access_denied", 403))
        self.assertEqual(failure_category(SimpleNamespace(code=404)), ("model_unavailable", 404))

    def test_gemini_request_uses_fast_model_and_closes_client(self):
        from google import genai

        pipeline = RAGPipeline([])
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder", "GEMINI_MODEL": DEFAULT_GEMINI_MODEL}), \
             patch.object(genai, "Client") as constructor:
            client = constructor.return_value
            client.models.generate_content.return_value.text = "An answer"
            answer = pipeline._generate_with_gemini([{"role": "system", "content": "Use sources"}, {"role": "user", "content": "Question"}])
        self.assertEqual(answer, "An answer")
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"], DEFAULT_GEMINI_MODEL)
        self.assertEqual(constructor.call_args.kwargs["http_options"].retry_options.attempts, 1)
        client.close.assert_called_once()

    def test_gemini_client_closed_on_timeout(self):
        from google import genai

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}), patch.object(genai, "Client") as constructor:
            constructor.return_value.models.generate_content.side_effect = TimeoutError("private response")
            with self.assertRaises(TimeoutError):
                RAGPipeline([])._generate_with_gemini([{"role": "system", "content": "Use sources"}])
            constructor.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
