from typing import Optional
import os
import chromadb
from lib.tooling import tool
from lib.vector_db import VectorStore
from tavily import TavilyClient

chroma_client = chromadb.PersistentClient(path="chromadb")
collection = chroma_client.get_collection("udaplay")

@tool
def retrieve_game(query: str) -> list[dict]:
    """
    Semantic search: Finds most results in the vector DB
    args:
    - query: a question about game industry.
    """
    vector_store = VectorStore(chroma_collection=collection)
    results = vector_store.query(query_texts=[query], n_results=5)
    return results["documents"]

from lib.evaluation import AgentEvaluator, TestCase, EvaluationResult

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

    agent_evaluator = AgentEvaluator(api_key=os.getenv("OPENAI_API_KEY"), tools=[retrieve_game])
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
def tavily_web_search(question: str, max_results: Optional[int] = 5) -> list[dict]:
    """
    Semantic search: Finds most results in the web
    args:
    - question: a question about game industry.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily_client.search(query=question, max_results=max_results)
    return results

@tool
def game_web_search(question: str, tavily_api_key: str) -> list[dict]:
    """
    Semantic search: Finds most results in the vector DB
    args:
    - question: a question about game industry.
    """
    # First, try to retrieve documents from the vector DB
    retrieved_docs = retrieve_game(question)
    evaluation = evaluate_retrieval(question, retrieved_docs)
    if evaluation["useful"] and evaluation["score"] > 0.5:
        return retrieved_docs

    # if the documents are not useful, try to search the web
    web_results = tavily_web_search(question, tavily_api_key)
    if evaluation["score"] > 0.5:
        return retrieved_docs + web_results

    return web_results