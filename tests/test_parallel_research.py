import threading

import tools


def test_research_subquestions_runs_in_parallel(monkeypatch):
    barrier = threading.Barrier(2)

    def fake_research(question: str, worker_id: int) -> dict:
        # Both workers must reach this point concurrently.
        barrier.wait(timeout=2)

        return {
            "question": question,
            "answer": f"Answer {worker_id}",
            "tools_used": [],
        }

    monkeypatch.setattr(
        tools,
        "_research_subquestion",
        fake_research,
    )

    results = tools.research_subquestions(
        subquestions=[
            "Question A",
            "Question B",
        ],
        max_workers=2,
    )

    assert [result["question"] for result in results] == [
        "Question A",
        "Question B",
    ]