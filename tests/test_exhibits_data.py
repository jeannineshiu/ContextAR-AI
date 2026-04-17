"""
Tests for exhibits_data.py: data integrity of the EXHIBITS list.

Run with:
    python -m pytest test_exhibits_data.py -v

No mocking needed — this file tests the data directly.
These tests act as a guard rail: if anyone accidentally removes an exhibit,
renames a key, or swaps in a non-Met painting, a test fails immediately.
"""

import pytest
from exhibits_data import EXHIBITS


# ---------------------------------------------------------------------------
# Expected values — single source of truth for the test suite
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"id", "name", "artist", "year", "type", "period",
                 "met_accession", "content"}

EXPECTED_PAINTINGS = [
    {
        "name":          "The Harvesters",
        "artist":        "Pieter Bruegel the Elder",
        "year":          "1565",
        "met_accession": "19.164",
    },
    {
        "name":          "Young Woman with a Water Pitcher",
        "artist":        "Johannes Vermeer",
        "year":          "c. 1662",
        "met_accession": "89.15.21",
    },
    {
        "name":          "Aristotle with a Bust of Homer",
        "artist":        "Rembrandt van Rijn",
        "year":          "1653",
        "met_accession": "61.198",
    },
    {
        "name":          "Madame X (Madame Pierre Gautreau)",
        "artist":        "John Singer Sargent",
        "year":          "1883–84",
        "met_accession": "16.53",
    },
    {
        "name":          "Wheat Field with Cypresses",
        "artist":        "Vincent van Gogh",
        "year":          "1889",
        "met_accession": "49.30",
    },
    {
        "name":          "The Card Players",
        "artist":        "Paul Cézanne",
        "year":          "c. 1890–95",
        "met_accession": "61.101.1",
    },
]

# Paintings that were considered but are NOT in the Met collection
EXCLUDED_PAINTINGS = [
    "Self-Portrait with Two Circles",   # Kenwood House, London
]


# ---------------------------------------------------------------------------
# Collection-level checks
# ---------------------------------------------------------------------------

class TestCollectionSize:

    def test_exhibits_is_a_list(self):
        assert isinstance(EXHIBITS, list)

    def test_exactly_six_exhibits(self):
        assert len(EXHIBITS) == 6

    def test_all_ids_are_unique(self):
        ids = [e["id"] for e in EXHIBITS]
        assert len(ids) == len(set(ids))

    def test_all_names_are_unique(self):
        names = [e["name"] for e in EXHIBITS]
        assert len(names) == len(set(names))

    def test_all_met_accessions_are_unique(self):
        accessions = [e["met_accession"] for e in EXHIBITS]
        assert len(accessions) == len(set(accessions))


# ---------------------------------------------------------------------------
# Schema: every exhibit must have all required keys
# ---------------------------------------------------------------------------

class TestSchema:

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_has_all_required_keys(self, exhibit):
        missing = REQUIRED_KEYS - set(exhibit.keys())
        assert not missing, f"Missing keys in '{exhibit.get('name')}': {missing}"

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_no_empty_string_values(self, exhibit):
        empty = [k for k, v in exhibit.items() if isinstance(v, str) and v.strip() == ""]
        assert not empty, f"Empty values for keys {empty} in '{exhibit.get('name')}'"

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_type_is_painting(self, exhibit):
        assert exhibit["type"] == "painting"

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_content_is_at_least_100_chars(self, exhibit):
        assert len(exhibit["content"].strip()) >= 100, \
            f"Content too short for '{exhibit['name']}'"

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_id_uses_underscores_not_spaces(self, exhibit):
        assert " " not in exhibit["id"], \
            f"ID '{exhibit['id']}' contains spaces — use underscores"

    @pytest.mark.parametrize("exhibit", EXHIBITS, ids=[e["name"] for e in EXHIBITS])
    def test_id_is_lowercase(self, exhibit):
        assert exhibit["id"] == exhibit["id"].lower(), \
            f"ID '{exhibit['id']}' must be lowercase"


# ---------------------------------------------------------------------------
# Presence of the correct six Met paintings
# ---------------------------------------------------------------------------

class TestExpectedPaintings:

    def _by_name(self, name: str) -> dict:
        matches = [e for e in EXHIBITS if e["name"] == name]
        assert matches, f"Painting '{name}' not found in EXHIBITS"
        return matches[0]

    @pytest.mark.parametrize("expected", EXPECTED_PAINTINGS,
                              ids=[p["name"] for p in EXPECTED_PAINTINGS])
    def test_painting_is_present(self, expected):
        names = [e["name"] for e in EXHIBITS]
        assert expected["name"] in names

    @pytest.mark.parametrize("expected", EXPECTED_PAINTINGS,
                              ids=[p["name"] for p in EXPECTED_PAINTINGS])
    def test_artist_is_correct(self, expected):
        exhibit = self._by_name(expected["name"])
        assert exhibit["artist"] == expected["artist"]

    @pytest.mark.parametrize("expected", EXPECTED_PAINTINGS,
                              ids=[p["name"] for p in EXPECTED_PAINTINGS])
    def test_year_is_correct(self, expected):
        exhibit = self._by_name(expected["name"])
        assert exhibit["year"] == expected["year"]

    @pytest.mark.parametrize("expected", EXPECTED_PAINTINGS,
                              ids=[p["name"] for p in EXPECTED_PAINTINGS])
    def test_met_accession_is_correct(self, expected):
        exhibit = self._by_name(expected["name"])
        assert exhibit["met_accession"] == expected["met_accession"]


# ---------------------------------------------------------------------------
# Guard against non-Met paintings slipping back in
# ---------------------------------------------------------------------------

class TestExcludedPaintings:

    @pytest.mark.parametrize("name", EXCLUDED_PAINTINGS)
    def test_excluded_painting_not_present(self, name):
        names = [e["name"] for e in EXHIBITS]
        assert name not in names, \
            f"'{name}' is not in the Met collection and must not be in EXHIBITS"


# ---------------------------------------------------------------------------
# Content sanity: exhibit descriptions must mention key identifiers
# ---------------------------------------------------------------------------

class TestContentSanity:

    def _content(self, name: str) -> str:
        return next(e["content"] for e in EXHIBITS if e["name"] == name)

    def test_harvesters_content_mentions_bruegel(self):
        assert "Bruegel" in self._content("The Harvesters")

    def test_vermeer_content_mentions_water_pitcher_or_window(self):
        content = self._content("Young Woman with a Water Pitcher")
        assert "pitcher" in content.lower() or "window" in content.lower()

    def test_aristotle_content_mentions_homer(self):
        assert "Homer" in self._content("Aristotle with a Bust of Homer")

    def test_madame_x_content_mentions_sargent(self):
        assert "Sargent" in self._content("Madame X (Madame Pierre Gautreau)")

    def test_wheat_field_content_mentions_van_gogh(self):
        assert "Van Gogh" in self._content("Wheat Field with Cypresses")

    def test_card_players_content_mentions_cezanne(self):
        content = self._content("The Card Players")
        assert "Cézanne" in content or "Cezanne" in content

    def test_all_content_mentions_metropolitan_or_met(self):
        for exhibit in EXHIBITS:
            content = exhibit["content"]
            assert "Metropolitan" in content or "Met" in content, \
                f"'{exhibit['name']}' content should reference the Met"
