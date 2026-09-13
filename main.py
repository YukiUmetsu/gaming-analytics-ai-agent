from lib.agents import Agent
from lib.utils import log_result, configure_logging
from lib.results import save_result
from tools import (
    retrieve_game,
    game_web_search,
    evaluate_retrieval,
    exact_game_lookup,
    search_memory,
    get_recent_game_news,
    plan_query,
    research_subquestions,
    research_entities,
)
from lib.observability import (
    configure_observability,
    flush_observability,
)

def main():
    configure_observability()
    configure_logging()

    agentic_rag = Agent(
        model_name="gpt-4o-mini",
        tools=[
            retrieve_game,
            game_web_search,
            evaluate_retrieval,
            exact_game_lookup,
            search_memory,
            get_recent_game_news,
            plan_query,
            research_entities,
            research_subquestions,
        ],
        instructions=(
            "You are an Agentic RAG assistant that intelligently decides which tools "
            "to use to answer game-industry questions. "
            "For simple questions, use the appropriate research tools directly. "
            "For questions involving multiple entities, multiple research tasks, or "
            "a combination of discovery and research, call plan_query before doing the research. "
            "If plan_query returns 'parallel', pass all returned subquestions together "
            "to research_subquestions. Do not research those entities individually from the parent agent. "
            "If plan_query returns 'sequential_then_parallel', first complete only the "
            "prerequisite using the appropriate tool. Once the prerequisite result "
            "identifies the entities or topics, create one independent research "
            "question for each and pass them together to research_subquestions. "
            "Use the subagent results to synthesize one final answer to the user's original question. "
            "Do not use research_subquestions for dependent tasks. "
            "Reason about the response and retry or reformulate searches when needed."
        ),
        temperature=0.0,
    )

    questions = [
        # "When Pokémon Gold and Silver was released?",
        # "Which one was the first 3D platformer Mario game?",
        # "Was Mortal Kombat X realeased for Playstation 5?",
        # # Custom-tool demonstrations.
        # "What is Rockstar Games currently working on?",
        # "What did we previously learn about Rockstar's current projects?",
        # "Compare Marvel's Spider-Man, Grand Theft Auto: San Andreas, and Gran Turismo 5. For each game, find the release year and platform, then tell me which was released first and which was released most recently.",
        # Grand Theft Auto: San Andreas — 2004
        # Gran Turismo 5 — 2010
        # Marvel's Spider-Man — 2018
        "Find three games that have had major news or announcements in the past month. For each game, research its developer or publisher, release date and platforms, and explain briefly why it is currently in the news. Then compare the three"
    ]

    try:
        for question in questions:
            run = agentic_rag.invoke(
                query=question,
                session_id="pokemon",
            )

            log_result(question, run)
            save_result(question, run)
    finally:
        flush_observability()

if __name__ == "__main__":
    main()