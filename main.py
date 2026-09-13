from lib.agents import Agent
from lib.utils import log_result, configure_logging
from tools import retrieve_game, game_web_search, evaluate_retrieval, exact_game_lookup, search_memory, get_recent_game_news

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
        ],
        instructions=(
            "You are an Agentic RAG assistant that can intelligently decide which tools to use "
            "to answer user questions. Reason about the response, change the query and call "
            "the tool again if needed in order to get better results. Always explain your reasoning "
            "for tool selection and provide comprehensive answers."
        ),
    )

    questions = [
        "When Pokémon Gold and Silver was released?",
        "Which one was the first 3D platformer Mario game?",
        "Was Mortal Kombat X realeased for Playstation 5?",
        # Custom-tool demonstrations.
        "What is Rockstar Games currently working on?",
        "What did we previously learn about Rockstar's current projects?"
    ]

    for question in questions:
        run = agentic_rag.invoke(
            query=question,
            session_id="pokemon",
        )

        log_result(question, run)


if __name__ == "__main__":
    main()