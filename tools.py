import os
import chromadb
from lib.tooling import tool
from lib.vector_db import VectorStore, VectorStoreManager
from tavily import TavilyClient
from lib.memory import MemoryFragment, LongTermMemory
from lib.observability import tracer
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
@tracer.tool
def retrieve_game(query: str) -> list[str]:
    """
    Semantic search: Finds most results in the vector DB
    args:
    - query: a question about game industry.
    """
    vector_store = VectorStore(chroma_collection=collection)
    results = vector_store.query(query_texts=[query], n_results=5)
    return results.get("documents", [[]])[0]

from lib.evaluation import AgentEvaluator

@tool
@tracer.tool
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
@tracer.tool
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
@tracer.tool
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
@tracer.tool
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
@tracer.tool
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
@tracer.tool
def exact_game_lookup(
    name: str,
    platform: str | None = None,
) -> list[dict]:
    """
    Find games using exact metadata rather than semantic similarity.

    Args:
        name: Game title.
        platform: Optional platform name.
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
@tracer.tool
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
    needs_decomposition: bool = Field(
        description="Whether the question requires multiple independent facts"
    )
    subquestions: list[str] = Field(
        description="Independent questions needed to answer the original question"
    )


@tool
def decompose_query(question: str) -> dict:
    """
    Break a complex game-industry question into independent subquestions.

    Use this when answering the original question requires multiple facts,
    lookups, comparisons, filters, or intermediate steps.

    Do not use this for simple single-fact questions.
    """
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    prompt = f"""Analyze this video-game question: {question}
Determine whether answering it requires multiple independent facts or steps.
If it is simple, set needs_decomposition to false and return no subquestions.
If it is complex, split it into the smallest useful independent
subquestions. Each subquestion should be answerable using one or more
available retrieval/search tools.
Do not answer the question itself.
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
    max_workers: int = 3,
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