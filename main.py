from tools import retrieve_game, game_web_search, evaluate_retrieval
from lib.agents import Agent
import os

def main():
    agentic_rag = Agent(
        model_name="gpt-4o-mini",
        tools=[retrieve_game, game_web_search, evaluate_retrieval],
        instructions=(
            "You are an Agentic RAG assistant that can intelligently decide which tools to use "
            "to answer user questions. Reason about about the response, change the query and call the tool again if needed "
            "in order to get better results. Always explain your reasoning for tool selection and provide comprehensive answers."
        )
    )
    run_1 = agentic_rag.invoke(
        query="When Pokémon Gold and Silver was released?",
        session_id="pokemon",
    )
    run_2 = agentic_rag.invoke(
        query="Which one was the first 3D platformer Mario game?",
        session_id="pokemon",
    )
    run_3 = agentic_rag.invoke(
        query="Was Mortal Kombat X realeased for Playstation 5?",
        session_id="pokemon",
    )
    print(run_1.get_final_state()["messages"])
    print(run_2.get_final_state()["messages"])
    print(run_3.get_final_state()["messages"])


if __name__ == "__main__":
    main()
