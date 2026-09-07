import json
import math

from ez_tool.data.toolbench import SUBSET_OF_STAGE, document_text
from ez_tool.eval.metrics import ndcg_at_k, ndcg_at_k_toolgen


def test_document_text_matches_toolgen_concatenation():
    doc = {
        "category_name": "Finance",
        "tool_name": "Quotes",
        "api_name": "Get Quote",
        "api_description": "Returns a quote",
        "required_parameters": [{"name": "sym", "type": "string"}],
        "optional_parameters": [],
    }
    out = document_text(doc)
    assert out.startswith("Finance, Quotes, Get Quote, Returns a quote, required_params: ")
    # template_response is absent from the released corpus; ToolGen still emits
    # the key with a json-dumped empty string.
    assert out.endswith('return_schema: ""')
    assert json.dumps(doc["required_parameters"]) in out


def test_document_text_tolerates_null_fields():
    # The corpus stores explicit nulls; `or ''` must absorb them, not crash.
    out = document_text({"category_name": None, "tool_name": None, "api_name": None})
    assert out.startswith(", , , , required_params: ")


def test_toolgen_ndcg_ignores_missed_relevant_docs():
    # 1 of 3 relevant retrieved, at rank 1.
    ranked, relevant = ["a", "x", "y"], {"a", "b", "c"}
    # Standard NDCG divides by the ideal over all 3 relevant docs.
    assert ndcg_at_k(ranked, relevant, 5) < 0.5
    # ToolGen's builds the ideal from retrieved relevant docs only, so a single
    # hit at rank 1 scores a perfect 1.0.
    assert ndcg_at_k_toolgen(ranked, relevant, 5) == 1.0


def test_toolgen_ndcg_equals_standard_when_all_relevant_retrieved():
    ranked, relevant = ["a", "b", "z"], {"a", "b"}
    assert ndcg_at_k_toolgen(ranked, relevant, 5) == ndcg_at_k(ranked, relevant, 5)


def test_toolgen_ndcg_is_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k_toolgen(["x", "y"], {"a"}, 5) == 0.0


def test_toolgen_ndcg_nan_without_labels():
    assert math.isnan(ndcg_at_k_toolgen(["a"], set(), 5))


def test_stage_to_subset_mapping():
    assert SUBSET_OF_STAGE == {"G1": "I1", "G2": "I2", "G3": "I3"}
