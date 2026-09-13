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
    decompose_query,
    research_subquestions,
)

def main():
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
            decompose_query,
            research_subquestions,
        ],
        instructions=(
            "You are an Agentic RAG assistant that can intelligently decide which tools to use "
            "to answer user questions. Reason about the response, change the query and call "
            "the tool again if needed to get better results. "
            "For questions requiring multiple independent facts, use decompose_query first. "
            "If it returns multiple independent subquestions, pass them together to "
            "research_subquestions so separate research agents can investigate them concurrently. "
            "Use their results to synthesize one answer to the original question. "
            "Do not decompose simple single-fact questions. "
            "Always explain your reasoning for tool selection and provide comprehensive answers."
        ),
    )

    questions = [
        "When Pokémon Gold and Silver was released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X realeased for Playstation 5?",
        # Custom-tool demonstrations.
        "What is Rockstar Games currently working on?",
        "What did we previously learn about Rockstar's current projects?",
        "Compare Marvel's Spider-Man, Grand Theft Auto: San Andreas, and Gran Turismo 5. For each game, find the release year and platform, then tell me which was released first and which was released most recently.",
        # Grand Theft Auto: San Andreas — 2004
        # Gran Turismo 5 — 2010
        # Marvel's Spider-Man — 2018
        "Find three games that have had major news or announcements in the past month. For each game, research its developer or publisher, release date and platforms, and explain briefly why it is currently in the news. Then compare the three"
    ]

    for question in questions:
        run = agentic_rag.invoke(
            query=question,
            session_id="pokemon",
        )

        log_result(question, run)
        save_result(question, run)

if __name__ == "__main__":
    main()