import json
from datetime import datetime, timedelta

from lib.messages import AIMessage
from lib.results import save_result


class FakeRun:
    def __init__(self):
        self.start_timestamp = datetime.now()
        self.end_timestamp = (
            self.start_timestamp + timedelta(seconds=1.5)
        )
        self.snapshots = []

    def get_final_state(self):
        return {
            "messages": [
                AIMessage(
                    content="Pokémon Gold and Silver were released in 1999.",
                )
            ],
            "total_tokens": 123,
        }


def test_save_result_writes_jsonl(tmp_path):
    output_path = tmp_path / "answers.jsonl"

    save_result(
        question="When was Pokémon Gold and Silver released?",
        run=FakeRun(),
        output_path=str(output_path),
    )

    data = json.loads(
        output_path.read_text(encoding="utf-8").strip()
    )

    assert data["question"] == (
        "When was Pokémon Gold and Silver released?"
    )
    assert data["answer"] == (
        "Pokémon Gold and Silver were released in 1999."
    )
    assert data["tools_used"] == []
    assert data["tokens"] == 123
    assert data["duration_seconds"] == 1.5
    assert "timestamp" in data