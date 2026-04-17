"""
Tests for rag_engine.py: build_index(), get_or_build_index(),
RAGEngine.query() (truncation + source filtering), and find_similar().

Run with:
    python -m pytest test_rag_engine.py -v

All FAISS, OpenAI embeddings, and LLM calls are mocked —
no API key, no disk index, no network access needed.
"""

from unittest.mock import patch, MagicMock

import rag_engine
from rag_engine import build_index, get_or_build_index, RAGEngine
from exhibits_data import EXHIBITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(name: str, score: float = 0.5) -> tuple:
    """Return a (LangChain Document mock, L2 score) pair."""
    doc = MagicMock()
    doc.metadata = {"name": name}
    return doc, score


def make_rag(answer: str = "A great painting.", sources: list[tuple] = None):
    """
    Build a RAGEngine whose internal FAISS and QA chain are fully mocked.

    sources: list of (name, score) pairs for similarity_search_with_score.
             Default: one doc "The Card Players" at score 0.5.
    """
    if sources is None:
        sources = [("The Card Players", 0.5)]

    source_docs = [make_doc(name, score) for name, score in sources]

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = source_docs
    mock_vs.similarity_search.return_value = [d for d, _ in source_docs]

    mock_qa = MagicMock()
    mock_qa.invoke.return_value = {"result": answer, "source_documents": []}

    with patch("rag_engine.get_or_build_index", return_value=mock_vs), \
         patch("rag_engine.build_qa_chain", return_value=mock_qa):
        engine = RAGEngine()

    return engine


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:

    def test_creates_one_document_per_exhibit(self):
        mock_vs = MagicMock()
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            mock_faiss_cls.from_documents.return_value = mock_vs
            mock_vs.save_local = MagicMock()
            build_index()
        docs_passed = mock_faiss_cls.from_documents.call_args[0][0]
        assert len(docs_passed) == len(EXHIBITS)

    def test_each_document_contains_exhibit_name(self):
        mock_vs = MagicMock()
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            mock_faiss_cls.from_documents.return_value = mock_vs
            mock_vs.save_local = MagicMock()
            build_index()
        docs = mock_faiss_cls.from_documents.call_args[0][0]
        exhibit_names = {e["name"] for e in EXHIBITS}
        for doc in docs:
            assert any(name in doc.page_content for name in exhibit_names)

    def test_metadata_includes_artist_and_year(self):
        mock_vs = MagicMock()
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            mock_faiss_cls.from_documents.return_value = mock_vs
            mock_vs.save_local = MagicMock()
            build_index()
        docs = mock_faiss_cls.from_documents.call_args[0][0]
        for doc in docs:
            assert "artist" in doc.metadata
            assert "year" in doc.metadata

    def test_saves_index_to_disk(self):
        mock_vs = MagicMock()
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            mock_faiss_cls.from_documents.return_value = mock_vs
            build_index()
        mock_vs.save_local.assert_called_once_with(rag_engine.FAISS_INDEX_PATH)

    def test_returns_vectorstore(self):
        mock_vs = MagicMock()
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            mock_faiss_cls.from_documents.return_value = mock_vs
            result = build_index()
        assert result is mock_vs


# ---------------------------------------------------------------------------
# get_or_build_index
# ---------------------------------------------------------------------------

class TestGetOrBuildIndex:

    def test_loads_existing_index_when_path_exists(self):
        with patch("os.path.exists", return_value=True), \
             patch("rag_engine.load_index") as mock_load, \
             patch("rag_engine.build_index") as mock_build:
            get_or_build_index()
        mock_load.assert_called_once()
        mock_build.assert_not_called()

    def test_builds_index_when_path_missing(self):
        with patch("os.path.exists", return_value=False), \
             patch("rag_engine.load_index") as mock_load, \
             patch("rag_engine.build_index") as mock_build:
            get_or_build_index()
        mock_build.assert_called_once()
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# load_index — FAISS.load_local parameters
# ---------------------------------------------------------------------------

class TestLoadIndex:

    def test_passes_correct_index_path(self):
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            from rag_engine import load_index
            load_index()
        call_args = mock_faiss_cls.load_local.call_args
        assert call_args[0][0] == rag_engine.FAISS_INDEX_PATH

    def test_allow_dangerous_deserialization_is_true(self):
        """FAISS.load_local requires allow_dangerous_deserialization=True —
        removing it raises an exception at runtime. This test pins the flag."""
        with patch("rag_engine.OpenAIEmbeddings"), \
             patch("rag_engine.FAISS") as mock_faiss_cls:
            from rag_engine import load_index
            load_index()
        call_kwargs = mock_faiss_cls.load_local.call_args.kwargs
        assert call_kwargs.get("allow_dangerous_deserialization") is True

    def test_uses_openai_embeddings(self):
        with patch("rag_engine.OpenAIEmbeddings") as mock_emb_cls, \
             patch("rag_engine.FAISS"):
            from rag_engine import load_index
            load_index()
        mock_emb_cls.assert_called_once_with(model=rag_engine.EMBED_MODEL)


# ---------------------------------------------------------------------------
# RAGEngine.query() — answer passthrough
# ---------------------------------------------------------------------------

class TestQueryAnswer:

    def test_returns_answer_key(self):
        engine = make_rag(answer="Bruegel painted this in 1565.")
        result = engine.query("Who painted this?")
        assert "answer" in result

    def test_returns_sources_key(self):
        engine = make_rag()
        result = engine.query("Tell me about this painting.")
        assert "sources" in result

    def test_answer_is_stripped(self):
        engine = make_rag(answer="  Some answer with whitespace.  ")
        result = engine.query("test")
        assert result["answer"] == "Some answer with whitespace."

    def test_answer_is_correct(self):
        engine = make_rag(answer="Van Gogh painted Wheat Field with Cypresses in 1889.")
        result = engine.query("When was this painted?")
        assert "Van Gogh" in result["answer"]


# ---------------------------------------------------------------------------
# RAGEngine.query() — max_length truncation
# ---------------------------------------------------------------------------

class TestQueryTruncation:

    def test_no_truncation_when_max_length_none(self):
        long_answer = "A " * 200   # 400 chars
        engine = make_rag(answer=long_answer.strip())
        result = engine.query("test", max_length=None)
        assert "…" not in result["answer"]

    def test_truncates_when_answer_exceeds_max_length(self):
        engine = make_rag(answer="word " * 60)   # 300 chars
        result = engine.query("test", max_length=100)
        assert len(result["answer"]) <= 104   # a few chars for "…" + word boundary

    def test_truncated_answer_ends_with_ellipsis(self):
        engine = make_rag(answer="word " * 60)
        result = engine.query("test", max_length=50)
        assert result["answer"].endswith("…")

    def test_no_truncation_when_answer_fits_exactly(self):
        engine = make_rag(answer="short answer")
        result = engine.query("test", max_length=200)
        assert "…" not in result["answer"]

    def test_truncation_breaks_on_word_boundary(self):
        original = "one two three four five six seven eight nine ten"
        engine = make_rag(answer=original)
        result = engine.query("test", max_length=20)
        answer = result["answer"]
        assert answer.endswith("…")
        body = answer[:-1]   # strip ellipsis
        # body must be a valid word-boundary prefix of the original
        assert original.startswith(body)
        assert len(body) <= 20


# ---------------------------------------------------------------------------
# RAGEngine.query() — source filtering by L2 distance
# ---------------------------------------------------------------------------

class TestQuerySourceFiltering:

    def test_source_below_threshold_is_included(self):
        engine = make_rag(sources=[("Madame X", 0.8)])
        result = engine.query("test")
        assert "Madame X" in result["sources"]

    def test_source_at_threshold_boundary_is_included(self):
        """Score exactly at 1.29 (just below 1.3) should be included."""
        engine = make_rag(sources=[("Madame X", 1.29)])
        result = engine.query("test")
        assert "Madame X" in result["sources"]

    def test_source_above_threshold_is_excluded(self):
        """L2 score ≥ 1.3 means the doc is not relevant enough."""
        engine = make_rag(sources=[("Unrelated Painting", 1.5)])
        result = engine.query("test")
        assert "Unrelated Painting" not in result["sources"]

    def test_source_at_exact_threshold_is_excluded(self):
        """Score == 1.3 should be excluded (condition is strict <)."""
        engine = make_rag(sources=[("Borderline Painting", 1.3)])
        result = engine.query("test")
        assert "Borderline Painting" not in result["sources"]

    def test_multiple_sources_filtered_correctly(self):
        engine = make_rag(sources=[
            ("The Harvesters", 0.6),        # included
            ("Young Woman with a Water Pitcher", 1.4),  # excluded
        ])
        result = engine.query("test")
        assert "The Harvesters" in result["sources"]
        assert "Young Woman with a Water Pitcher" not in result["sources"]

    def test_empty_sources_when_all_above_threshold(self):
        engine = make_rag(sources=[
            ("Painting A", 1.5),
            ("Painting B", 2.0),
        ])
        result = engine.query("test")
        assert result["sources"] == []

    def test_sources_list_length_matches_filtered_count(self):
        engine = make_rag(sources=[
            ("Wheat Field with Cypresses", 0.4),
            ("The Card Players", 0.7),
        ])
        result = engine.query("test")
        assert len(result["sources"]) == 2


# ---------------------------------------------------------------------------
# RAGEngine.find_similar()
# ---------------------------------------------------------------------------

class TestFindSimilar:

    def test_returns_list_of_names(self):
        engine = make_rag(sources=[("The Harvesters", 0.5), ("Madame X", 0.8)])
        names = engine.find_similar("The Harvesters")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_default_k_is_2(self):
        engine = make_rag()
        engine.find_similar("Wheat Field with Cypresses")
        engine._vectorstore.similarity_search.assert_called_with(
            "Wheat Field with Cypresses", k=2
        )

    def test_custom_k_is_forwarded(self):
        engine = make_rag()
        engine.find_similar("Madame X", k=3)
        engine._vectorstore.similarity_search.assert_called_with("Madame X", k=3)

    def test_returns_exhibit_names_from_metadata(self):
        engine = make_rag(sources=[("The Card Players", 0.5)])
        names = engine.find_similar("card game")
        assert "The Card Players" in names
