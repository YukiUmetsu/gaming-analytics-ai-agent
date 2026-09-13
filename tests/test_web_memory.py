import tools

def test_save_web_result_stores_content_and_source(
    monkeypatch,
):
    captured = {}

    def fake_register(memory_fragment, metadata=None):
        captured["fragment"] = memory_fragment
        captured["metadata"] = metadata

    monkeypatch.setattr(
        tools.long_term_memory,
        "register",
        fake_register,
    )

    tools.save_web_result(
        question="What is Rockstar working on?",
        result={
            "title": "Rockstar Games News",
            "url": "https://example.com/rockstar",
            "content": "Rockstar Games is developing GTA VI.",
        },
    )

    fragment = captured["fragment"]
    metadata = captured["metadata"]

    assert fragment.content == (
        "Rockstar Games is developing GTA VI."
    )
    assert fragment.owner == "udaplay"
    assert fragment.namespace == "game_knowledge"

    assert metadata["question"] == (
        "What is Rockstar working on?"
    )
    assert metadata["source_url"] == (
        "https://example.com/rockstar"
    )
    assert metadata["source_title"] == (
        "Rockstar Games News"
    )
    assert metadata["source_type"] == "web"


def test_save_web_result_ignores_empty_content(
    monkeypatch,
):
    called = False

    def fake_register(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        tools.long_term_memory,
        "register",
        fake_register,
    )

    tools.save_web_result(
        question="Question",
        result={
            "title": "Empty",
            "url": "https://example.com",
            "content": "",
        },
    )

    assert called is False