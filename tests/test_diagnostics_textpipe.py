"""Guard the EZR replica used by the ToolBench diagnostics.

The diagnostics may not import ``ezr.py``, so :mod:`ez_tool.diagnostics.textpipe`
re-implements its pipeline from source. These tests pin the two behaviours that
would silently change every number in the report if they drifted.
"""

import json
from pathlib import Path

import pytest

from ez_tool.diagnostics import textpipe

REFERENCE_DIR = Path("/mnt/beegfs/lli66/ez-tool/runs/textmine/reference")


def test_underscores_swallow_the_whole_identifier():
    # \b treats _ as a word character, so there is no boundary to match on.
    assert textpipe.tokenize("get_weather_forecast") == []
    assert textpipe.tokenize("get_weather_forecast", split_identifiers=True) == [
        "get",
        "weather",
        "forecast",
    ]


def test_camel_case_survives_unsplit_without_the_flag():
    assert textpipe.tokenize("sendMessage") == ["sendmessage"]
    assert textpipe.tokenize("sendMessage", split_identifiers=True) == [
        "send",
        "message",
    ]


def test_short_tokens_are_dropped():
    assert textpipe.tokenize("a an the api") == ["the", "api"]


def test_tfidf_ranks_by_ctf_times_idf():
    # "rare" appears twice in one doc of three; "common" is in all three and so
    # has log(3/3) == 0 weight.
    docs = [["common", "rare", "rare"], ["common"], ["common"]]
    vocab, scores = textpipe.top_terms(docs, None)
    assert vocab[0] == "rare"
    assert scores["common"] == 0.0


@pytest.mark.parametrize("name", ["Hall", "Radjenovic", "Wahono", "Kitchenham"])
def test_replica_matches_vendored_ezr_vocabulary(name):
    """The replica must reproduce ezr's own top-100 vocabulary, tie order included."""
    ref_path = REFERENCE_DIR / f"{name}.json"
    if not ref_path.exists():
        pytest.skip("reference dump is on beegfs; run on a compute node")

    from ez_tool.textmine import corpus

    ref = json.loads(ref_path.read_text())["features"]
    c = corpus.load(name, include_title=False, klass="label")
    cache: dict = {}
    docs = [textpipe.prepare(t, cache) for t in c.texts]
    vocab, _ = textpipe.top_terms(docs, ref["n_features"])
    assert vocab == ref["vocab"]
