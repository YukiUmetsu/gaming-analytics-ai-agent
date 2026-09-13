import json
import inspect
import datetime
from typing import (
    Any, Callable, 
    Literal, Optional, Union, TypeAlias,
    get_type_hints, get_origin, get_args,
)
from functools import wraps
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import SpanAttributes

# Type alias for OpenAI's tool call implementation
ToolCall: TypeAlias = ChatCompletionMessageToolCall

class Tool:
    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func)
        self.signature = inspect.signature(func, eval_str=True)
        self.type_hints = get_type_hints(func)

        self.parameters = [
            self._build_param_schema(key, param)
            for key, param in self.signature.parameters.items()
        ]

    def _build_param_schema(self, name: str, param: inspect.Parameter):
        param_type = self.type_hints.get(name, str)
        schema = self._infer_json_schema_type(param_type)
        return {
            "name": name,
            "schema": schema,
            "required": param.default == inspect.Parameter.empty
        }

    def _infer_json_schema_type(self, typ: Any) -> dict:
        origin = get_origin(typ)

        # Handle Literal (enums)
        if origin is Literal:
            return {
                "type": "string",
                "enum": list(get_args(typ))
            }

        # Handle Optional[T]
        if origin is Union:
            args = get_args(typ)
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1:
                return self._infer_json_schema_type(non_none[0])
            return {"type": "string"}  # fallback

        # Handle collections
        if origin is list:
            return {
                "type": "array",
                "items": self._infer_json_schema_type(get_args(typ)[0] if get_args(typ) else str)
            }

        if origin is dict:
            return {
                "type": "object",
                "additionalProperties": self._infer_json_schema_type(get_args(typ)[1] if get_args(typ) else str)
            }

        # Primitive mappings
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            datetime.date: "string",
            datetime.datetime: "string",
        }

        return {"type": mapping.get(typ, "string")}

    def dict(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param["name"]: param["schema"]
                        for param in self.parameters
                    },
                    "required": [
                        param["name"] for param in self.parameters if param["required"]
                    ],
                    "additionalProperties": False
                }
            }
        }

    def __call__(self, *args, **kwargs):
        tracer = trace.get_tracer("udaplay")

        input_value = json.dumps(
            kwargs if kwargs else args,
            ensure_ascii=False,
            default=str,
        )

        with tracer.start_as_current_span(
            self.name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
                SpanAttributes.TOOL_NAME: self.name,
                SpanAttributes.TOOL_PARAMETERS: input_value,
                SpanAttributes.INPUT_VALUE: input_value,
            },
        ) as span:
            try:
                result = self.func(*args, **kwargs)

                output_value = json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )

                span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE,
                    output_value[:5000],
                )
                span.set_status(Status(StatusCode.OK))

                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    Status(StatusCode.ERROR, str(exc))
                )
                raise

    def __repr__(self):
        return f"<Tool name={self.name} params={[p['name'] for p in self.parameters]}>"

    @classmethod
    def from_func(cls, func: Callable):
        return cls(func)



def tool(func=None, *, name: str = None, description: str = None):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return Tool(f, name=name, description=description)
    
    # @tool ou @tool(name="foo")
    return wrapper(func) if func else wrapper