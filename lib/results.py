import json
from datetime import datetime, timezone
from pathlib import Path

from lib.utils import get_final_answer, get_tools_used


def save_result(
    question: str,
    run,
    output_path: str = "outputs/answers.jsonl",
) -> None:
    state = run.get_final_state() or {}

    duration = (
        run.end_timestamp - run.start_timestamp
    ).total_seconds()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": get_final_answer(run),
        "tools_used": get_tools_used(run),
        "tokens": state.get("total_tokens", 0),
        "duration_seconds": round(duration, 2),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )