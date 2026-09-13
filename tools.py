import os
from typing import Literal
import chromadb
from lib.tooling import tool
from lib.vector_db import VectorStore, VectorStoreManager
from tavily import TavilyClient
from lib.memory import MemoryFragment, LongTermMemory
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from lib.llm import LLM
from lib.parsers import PydanticOutputParser
from lib.agents import Agent
from lib.utils import get_final_answer, get_tools_used

load_dotenv()

chroma_client = chromadb.PersistentClient(path="chromadb")
collection = chroma_client.get_collection("udaplay")
vector_store_manager = VectorStoreManager(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    persist_path="chromadb",
)

long_term_memory = LongTermMemory(
    vector_store_manager
)


@tool
def retrieve_game(query: str) -> list[str]:
    """
    Semantic search for a single game-industry research question.

    Use this directly for one research task.

    If there are multiple independent research questions, do not call this
    tool repeatedly from the parent agent. Use research_subquestions so each
    question can be researched concurrently by separate agents.

    Args:
        query: One game-industry research question.
    """
    vector_store = VectorStore(chroma_collection=collection)
    results = vector_store.query(query_texts=[query], n_results=5)
    return results.get("documents", [[]])[0]

from lib.evaluation import AgentEvaluator

@tool
def evaluate_retrieval(question: str, retrieved_docs: list[dict]):
    """
    Based on the user's question and on the list of retrieved documents,
    it will analyze the usability of the documents to respond to that question.
        args:
        - question: original question from user
        - retrieved_docs: retrieved documents most similar to the user query in the Vector Database

    The result includes:
        - useful: whether the documents are useful to answer the question
        - description: description about the evaluation result
    """
    result = {
        "useful": False,
        "description": "",
        "score": 0.0
    }

    agent_evaluator = AgentEvaluator(api_key=os.getenv("OPENAI_API_KEY"))
    evaluation = agent_evaluator.evaluate_retrieval(question, retrieved_docs)
    if evaluation.task_completion.task_completed:
        result["useful"] = True
        result["description"] = evaluation.feedback
        result["score"] = evaluation.overall_score
    else:
        result["useful"] = False
        result["description"] = evaluation.feedback
        result["score"] = evaluation.overall_score
    return result

@tool
def tavily_web_search(
    question: str,
    max_results: int = 5,
) -> dict:
    """
    Search the web for video-game industry information using Tavily.

    Args:
        question: A question about the video game industry.
        max_results: Maximum number of search results to return.
    """
    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    return tavily_client.search(
        query=question,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

@tool
def game_web_search(
    question: str,
    max_results: int = 5,
    confidence_threshold: float = 0.5,
) -> list[dict]:
    retrieved_docs = retrieve_game(question)

    evaluation = evaluate_retrieval(
        question=question,
        retrieved_docs=retrieved_docs,
    )

    internal_results = [
        {
            "content": doc,
            "source_type": "internal",
        }
        for doc in retrieved_docs
    ]

    internal_results = [
        {
            "content": doc,
            "source_type": "internal",
        }
        for doc in retrieved_docs
    ]

    useful = evaluation.get("useful", False)
    score = evaluation.get("score", 0.0)

    if useful and score >= confidence_threshold:
        return internal_results

    web_response = tavily_web_search(
        question=question,
        max_results=max_results,
    )

    raw_web_results = web_response.get("results", [])

    web_results = [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content", ""),
            "score": result.get("score"),
            "published_date": result.get("published_date"),
            "source_type": "web",
        }
        for result in raw_web_results
    ]

    for result in web_results[:2]:
        save_web_result(
            question=question,
            result=result,
        )

    if useful and internal_results:
        return internal_results + web_results

    return web_results

@tool
def search_memory(
    query: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search information learned from previous research.

    Args:
        query: Information to search for.
        limit: Maximum number of memories to return.
    """
    result = long_term_memory.search(
        query_text=query,
        owner="udaplay",
        namespace="game_knowledge",
        limit=limit,
    )

    distances = result.metadata.get(
        "distances",
        [],
    )

    return [
        {
            "content": fragment.content,
            "timestamp": fragment.timestamp,
            "distance": distance,
            "source_type": "memory",
        }
        for fragment, distance in zip(
            result.fragments,
            distances,
        )
    ]

@tool
def save_memory(
    content: str,
    source_url: str = "",
    source_title: str = "",
) -> dict:
    """
    Save useful game-industry information into long-term memory.

    Args:
        content: Information to remember.
        source_url: URL where the information came from.
        source_title: Title of the source.
    """
    fragment = MemoryFragment(
        content=content,
        owner="udaplay",
        namespace="game_knowledge",
    )

    long_term_memory.register(
        fragment,
        metadata={
            "source_url": source_url,
            "source_title": source_title,
            "source_type": "web",
        },
    )

    return {
        "saved": True,
        "content": content,
    }

@tool
def exact_game_lookup(
    name: str,
    platform: str | None = None,
) -> list[dict]:
    """
    Find games using exact metadata rather than semantic similarity.
    Args:
        name: Game title.
        platform: Optional platform name.

    Look up one game's exact structured metadata.
    Use this directly for a single lookup.
    For multiple independent games, use research_subquestions instead of
    repeatedly calling this tool from the parent agent.
    """
    where_conditions = [
        {
            "Name": {
                "$eq": name
            }
        }
    ]

    if platform:
        where_conditions.append(
            {
                "Platform": {
                    "$eq": platform
                }
            }
        )

    where = (
        where_conditions[0]
        if len(where_conditions) == 1
        else {"$and": where_conditions}
    )

    result = collection.get(
        where=where,
        include=[
            "documents",
            "metadatas",
        ],
    )

    return [
        {
            "id": doc_id,
            "content": document,
            "metadata": metadata,
            "source_type": "internal",
        }
        for doc_id, document, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )
    ]

@tool
def get_recent_game_news(
    question: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search recent video-game industry news.

    Args:
        question: Question about recent gaming information.
        max_results: Maximum number of results.
    """
    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily_client.search(
        query=question,
        topic="news",
        search_depth="advanced",
        time_range="month",
        max_results=max_results,
    )

    return [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content", ""),
            "score": result.get("score"),
            "published_date": result.get("published_date"),
            "source_type": "news",
        }
        for result in response.get("results", [])
    ]

def save_web_result(
    question: str,
    result: dict,
):
    content = result.get("content", "").strip()

    if not content:
        return

    fragment = MemoryFragment(
        content=content,
        owner="udaplay",
        namespace="game_knowledge",
    )

    long_term_memory.register(
        memory_fragment=fragment,
        metadata={
            "question": question,
            "source_url": result.get("url", ""),
            "source_title": result.get("title", ""),
            "source_type": "web",
        },
    )

class QueryPlan(BaseModel):
    strategy: Literal[
        "direct",
        "parallel",
        "sequential_then_parallel",
    ]
    prerequisite: str | None = Field(
        default=None,
        description=(
            "A task that must be completed before parallel research can begin."
        ),
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description="Independent questions that can be researched concurrently.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this strategy was selected.",
    )


@tool
def plan_query(question: str) -> dict:
    """
    Choose how a complex game-industry question should be executed.

    Strategies:
    - direct: answer using normal tools.
    - parallel: independent research questions can run concurrently.
    - sequential_then_parallel: first discover information required to
      create independent research questions, then run those in parallel.
    """
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    prompt = f"""
Plan how to answer this game-industry question:

{question}

Choose exactly one strategy:

direct:
Use when the question can be answered normally without splitting it
into multiple independent research tasks.
Return no prerequisite and no subquestions.

parallel:
Use when the question already contains two or more independent
research tasks that can run at the same time.
Return each independent task as a subquestion.
Return no prerequisite.

sequential_then_parallel:
Use when one prerequisite must be completed before the independent
research tasks are known.
For example, finding three currently trending games must happen before
each game can be researched independently.
Return the prerequisite task.
Do not invent unknown entities in subquestions.

Keep the reasoning brief.
Do not answer the original question.
"""

    response = llm.invoke(
        prompt,
        response_format=QueryPlan,
    )

    parser = PydanticOutputParser(
        model_class=QueryPlan,
    )

    plan = parser.parse(response)

    return plan.model_dump()

def _research_subquestion(
    question: str,
    worker_id: int,
) -> dict:
    researcher = Agent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        tools=[
            retrieve_game,
            game_web_search,
            evaluate_retrieval,
            exact_game_lookup,
            search_memory,
            get_recent_game_news,
        ],
        instructions=(
            "You are a focused game-industry research agent. "
            "Answer only the assigned subquestion. "
            "Use local knowledge first and use web or news search only when needed. "
            "Do not decompose the question further. "
            "Return a concise factual answer supported by the available evidence."
        ),
    )

    run = researcher.invoke(
        query=question,
        session_id=f"research-{worker_id}",
    )

    return {
        "question": question,
        "answer": get_final_answer(run),
        "tools_used": get_tools_used(run),
    }


@tool
def research_subquestions(
    subquestions: list[str],
    max_workers: int = 10,
) -> list[dict]:
    """
    Research independent subquestions concurrently using separate agents.

    Args:
        subquestions: Independent questions that can be researched in parallel.
        max_workers: Maximum number of concurrent research agents.
    """
    if not subquestions:
        return []

    worker_count = min(
        max(1, max_workers),
        len(subquestions),
        4,
    )

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = []

        for worker_id, question in enumerate(subquestions):
            # Preserve the parent trace across worker threads.
            context = copy_context()

            futures.append(
                executor.submit(
                    context.run,
                    _research_subquestion,
                    question,
                    worker_id,
                )
            )

        # Preserve the same order as the decomposed questions.
        return [
            future.result()
            for future in futures
        ]

@tool
def research_entities(
    entities: list[str],
    research_task: str,
    max_workers: int = 10,
) -> list[dict]:
    """
    Research multiple independent entities concurrently.

    Use this after another tool discovers a set of games, companies,
    people, or other entities that now need the same type of research.

    Args:
        entities: Names of entities to research independently.
        research_task: What should be researched about each entity.
        max_workers: Maximum number of concurrent researchers.
    """
    subquestions = [
        f"{research_task}\nEntity: {entity}"
        for entity in entities
    ]

    return research_subquestions(
        subquestions=subquestions,
        max_workers=max_workers,
    )