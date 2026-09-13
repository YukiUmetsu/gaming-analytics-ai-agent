import pytest

import tools


def test_game_web_search_returns_internal_results_when_retrieval_is_sufficient(
    monkeypatch,
):
    retrieved_docs = [
        [
            "Pokémon Gold and Silver were released for Game Boy Color in 1999."
        ]
    ]

    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: retrieved_docs,
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": True,
            "score": 0.9,
            "description": "Internal documents are sufficient.",
        },
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("Tavily should not be called when internal retrieval is sufficient")

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        fail_if_called,
    )

    result = tools.game_web_search(
        "When were Pokémon Gold and Silver released?"
    )

    assert result == [
        {
            "content": (
                "Pokémon Gold and Silver were released "
                "for Game Boy Color in 1999."
            ),
            "source_type": "internal",
        }
    ]


def test_game_web_search_uses_web_when_internal_results_are_not_useful(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [["Unrelated game information"]],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": False,
            "score": 0.2,
            "description": "Documents do not answer the question.",
        },
    )

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        lambda question, max_results=5: {
            "results": [
                {
                    "title": "Rockstar Games",
                    "url": "https://example.com/rockstar",
                    "content": "Rockstar Games is developing GTA VI.",
                    "score": 0.95,
                    "published_date": "2026-09-01",
                }
            ]
        },
    )

    result = tools.game_web_search(
        "What is Rockstar Games working on right now?"
    )

    assert result == [
        {
            "title": "Rockstar Games",
            "url": "https://example.com/rockstar",
            "content": "Rockstar Games is developing GTA VI.",
            "score": 0.95,
            "published_date": "2026-09-01",
            "source_type": "web",
        }
    ]


def test_game_web_search_combines_internal_and_web_results_when_internal_is_useful_but_low_confidence(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [
            [
                "Rockstar Games develops the Grand Theft Auto series."
            ]
        ],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": True,
            "score": 0.4,
            "description": "Relevant but does not contain current information.",
        },
    )

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        lambda question, max_results=5: {
            "results": [
                {
                    "title": "Current Rockstar Projects",
                    "url": "https://example.com/gta6",
                    "content": "Rockstar Games is currently developing GTA VI.",
                    "score": 0.97,
                    "published_date": "2026-09-10",
                }
            ]
        },
    )

    result = tools.game_web_search(
        "What is Rockstar Games working on right now?"
    )

    assert len(result) == 2

    assert result[0] == {
        "content": "Rockstar Games develops the Grand Theft Auto series.",
        "source_type": "internal",
    }

    assert result[1]["source_type"] == "web"
    assert result[1]["title"] == "Current Rockstar Projects"
    assert result[1]["score"] == 0.97


def test_game_web_search_respects_confidence_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [["Relevant internal information"]],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": True,
            "score": 0.7,
            "description": "Relevant internal information.",
        },
    )

    web_called = False

    def fake_web_search(question, max_results=5):
        nonlocal web_called
        web_called = True

        return {
            "results": [
                {
                    "title": "Web result",
                    "url": "https://example.com",
                    "content": "Additional information",
                    "score": 0.9,
                }
            ]
        }

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        fake_web_search,
    )

    tools.game_web_search(
        "Some question",
        confidence_threshold=0.8,
    )

    assert web_called is True


def test_game_web_search_passes_max_results_to_tavily(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": False,
            "score": 0.0,
            "description": "No useful documents.",
        },
    )

    received_max_results = None

    def fake_web_search(question, max_results=5):
        nonlocal received_max_results
        received_max_results = max_results
        return {"results": []}

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        fake_web_search,
    )

    tools.game_web_search(
        "Some question",
        max_results=8,
    )

    assert received_max_results == 8


def test_game_web_search_handles_empty_tavily_results(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": False,
            "score": 0.0,
            "description": "No useful internal results.",
        },
    )

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        lambda question, max_results=5: {
            "results": []
        },
    )

    result = tools.game_web_search(
        "Unknown game question"
    )

    assert result == []


def test_game_web_search_handles_missing_results_key(
    monkeypatch,
):
    monkeypatch.setattr(
        tools,
        "retrieve_game",
        lambda question: [],
    )

    monkeypatch.setattr(
        tools,
        "evaluate_retrieval",
        lambda question, retrieved_docs: {
            "useful": False,
            "score": 0.0,
            "description": "Insufficient.",
        },
    )

    monkeypatch.setattr(
        tools,
        "tavily_web_search",
        lambda question, max_results=5: {},
    )

    result = tools.game_web_search(
        "Unknown question"
    )

    assert result == []