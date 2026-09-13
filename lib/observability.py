import json
from functools import wraps

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

load_dotenv()

langfuse = get_client()


def _json_safe(value):
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    )


def trace_agent(name: str):
    def decorator(func):
        @wraps(func)
        def wrapped(
            self,
            query: str,
            session_id=None,
            *args,
            **kwargs,
        ):
            session_id = session_id or "default"

            with langfuse.start_as_current_observation(
                name=name,
                as_type="agent",
                input={
                    "query": query,
                    "session_id": session_id,
                },
            ) as observation:
                try:
                    with propagate_attributes(
                        session_id=session_id,
                    ):
                        result = func(
                            self,
                            query,
                            session_id,
                            *args,
                            **kwargs,
                        )

                    state = result.get_final_state() or {}

                    observation.update(
                        output={
                            "total_tokens": state.get(
                                "total_tokens",
                                0,
                            )
                        }
                    )

                    return result

                except Exception as exc:
                    observation.update(
                        level="ERROR",
                        status_message=str(exc),
                    )
                    raise

        return wrapped

    return decorator


def trace_tool(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        tool_name = getattr(
            self,
            "name",
            func.__name__,
        )

        tool_input = (
            kwargs
            if kwargs
            else list(args)
        )

        with langfuse.start_as_current_observation(
            name=tool_name,
            as_type="tool",
            input=_json_safe(tool_input),
        ) as observation:
            try:
                result = func(
                    self,
                    *args,
                    **kwargs,
                )

                observation.update(
                    output=_json_safe(result),
                )

                return result

            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message=str(exc),
                )
                raise

    return wrapped


def configure_observability():
    return langfuse


def flush_observability():
    langfuse.flush()