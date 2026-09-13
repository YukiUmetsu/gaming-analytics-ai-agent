from typing import TypedDict, List, Optional, Union
import json
from lib.observability import trace_agent
from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory
import logging
logger = logging.getLogger(__name__)

# Define the state schema
class AgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[dict]  # List of conversation messages
    current_tool_calls: Optional[List[ToolCall]]  # Current pending tool calls
    total_tokens: int  # Track the cumulative total

class Agent:
    def __init__(self,
                 model_name: str,
                 instructions: str,
                 tools: List[Tool] = None,
                 temperature: float = 0.7):
        """
        Initialize an Agent

        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
            temperature: Temperature parameter for LLM (default: 0.7)
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature

        # Initialize memory and state machine
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])
        # If no messages exist, start with system message
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))
        return {
            "messages": messages,
            "session_id": state["session_id"]
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        """Step logic: Process the current state through the LLM"""
        # Initialize LLM
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content, 
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Step logic: Execute any pending tool calls"""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []
        for call in tool_calls:
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id

            tool = next(
                (t for t in self.tools if t.name == function_name),
                None,
            )

            if not tool:
                result = {
                    "error": f"Tool '{function_name}' not found"
                }
            else:
                result = tool(**function_args)

            content = (
                result
                if isinstance(result, str)
                else json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                )
            )

            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
            )

        # Clear tool calls and add results to messages
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"]
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)

        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step, log_message=self._log_llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step, log_message=self._log_tool_step)
        termination = Termination[AgentState]()

        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])

        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        # Transition based on whether there are tool calls
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there are tool calls"""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)  # Go back to llm after tool execution

        return machine

    @trace_agent(name="udaplay-agent")
    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query

        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)

        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
        }
        run_object = self.workflow.run(initial_state)
        # Store the complete run object in memory
        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session
        Args:
            session_id: Optional session ID (uses "default" if None)
        Returns:
            List of Run objects in the session
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session
        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)

    def _log_llm_step(self, state: AgentState) -> str:
        tool_calls = state.get("current_tool_calls") or []

        if not tool_calls:
            return "LLM → final answer"

        calls = []

        for call in tool_calls:
            args = json.loads(call.function.arguments)

            args_text = str(args)

            if len(args_text) > 250:
                args_text = args_text[:250] + "..."

            calls.append(
                f"{call.function.name}({args_text})"
            )

        return "LLM → " + ", ".join(calls)

    def _log_tool_step(self, state: AgentState) -> str:
        messages = state.get("messages", [])

        if not messages:
            return "Tool completed"

        message = messages[-1]
        tool_name = getattr(message, "name", None)

        return (
            f"Tool {tool_name} completed"
            if tool_name
            else "Tool completed"
        )