import logging

from lib.messages import AIMessage

logger = logging.getLogger(__name__)


def get_final_answer(run) -> str:
    state = run.get_final_state()

    if not state:
        return ""

    for message in reversed(state.get("messages", [])):
        if (
            isinstance(message, AIMessage)
            and message.content
            and not message.tool_calls
        ):
            return message.content

    return ""


def get_tools_used(run) -> list[str]:
    tools = []

    for snapshot in run.snapshots:
        if snapshot.step_id != "llm_processor":
            continue

        tool_calls = snapshot.state_data.get("current_tool_calls") or []

        tools.extend(
            call.function.name
            for call in tool_calls
        )

    return tools


def log_result(question: str, run) -> None:
    state = run.get_final_state() or {}

    duration = (
        run.end_timestamp - run.start_timestamp
    ).total_seconds()

    tools = get_tools_used(run)
    answer = get_final_answer(run)

    logger.info("=" * 70)
    logger.info("Question: %s", question)
    logger.info(
        "Tools used: %s",
        ", ".join(tools) if tools else "None",
    )
    logger.info(
        "Duration: %.2fs | Tokens: %s",
        duration,
        state.get("total_tokens", 0),
    )
    logger.info("-" * 70)
    logger.info("%s", answer)
    logger.info("=" * 70)
    logger.info("")

class _HideHttpRequests(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("HTTP Request:")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )

    for handler in logging.getLogger().handlers:
        handler.addFilter(_HideHttpRequests())