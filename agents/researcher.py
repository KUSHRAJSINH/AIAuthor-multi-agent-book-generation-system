"""
ResearcherAgent — performs RAG retrieval (FAISS + BM25) and creates
structured research packets with fact registries and citations.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.models import BookState, ResearchPacket
from prompts.templates import PromptTemplates
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.faiss_store import FAISSStore
from rag.retriever import HybridRetriever
from utils.logger import ObservabilityLogger
from utils.cost_tracker import CostTracker


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        max_retries=3,
    )


# Seed corpus — domain knowledge injected at startup so retrieval has material to work with
SEED_CORPUS = [
    "Artificial intelligence refers to the simulation of human intelligence processes by machines, "
    "especially computer systems. These processes include learning, reasoning, and self-correction.",
    "Machine learning is a subset of AI that enables systems to learn and improve from experience "
    "without being explicitly programmed. It focuses on developing computer programs that can access "
    "data and use it to learn for themselves.",
    "Deep learning is part of a broader family of machine learning methods based on artificial "
    "neural networks with representation learning. Learning can be supervised, semi-supervised, or unsupervised.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial "
    "intelligence concerned with the interactions between computers and human language.",
    "Large language models (LLMs) are trained on vast text datasets and can generate, summarize, "
    "translate, and answer questions in human language. Examples include GPT-4, Claude, and LLaMA.",
    "Retrieval-Augmented Generation (RAG) combines the parametric knowledge of LLMs with non-parametric "
    "retrieval from external documents, improving factual accuracy and reducing hallucinations.",
    "Multi-agent systems consist of multiple intelligent agents that interact with each other. "
    "They are used in AI orchestration, simulation, and collaborative problem-solving.",
    "Vector databases store high-dimensional vector embeddings, enabling semantic similarity search "
    "that goes beyond keyword matching. Examples include FAISS, Pinecone, and Chroma.",
    "Prompt engineering is the practice of designing inputs to AI language models to elicit desired outputs. "
    "Effective prompts can dramatically improve model performance on specific tasks.",
    "Transfer learning allows a model trained on one task to be fine-tuned for another. "
    "This approach has revolutionized NLP and computer vision by reducing the need for large labeled datasets.",
    "Reinforcement learning from human feedback (RLHF) is a technique used to align LLMs with human values "
    "and preferences by training reward models on human comparisons.",
    "Agentic AI refers to systems where AI models take sequences of actions, plan, and use tools "
    "to complete complex multi-step tasks autonomously.",
    "Embeddings are dense vector representations of text that capture semantic meaning. "
    "Similar texts have similar embeddings, enabling clustering, search, and classification.",
    "Hallucination in AI refers to confident but factually incorrect outputs from language models. "
    "RAG and fact-checking pipelines are common mitigation strategies.",
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
    "It uses a graph-based approach where nodes are agents or functions and edges define flow.",
    "FAISS (Facebook AI Similarity Search) is an efficient library for similarity search "
    "and clustering of dense vectors. It supports billions of vectors at scale.",
    "Groq is an AI inference company that provides extremely fast token generation "
    "using their Language Processing Unit (LPU) hardware.",
    "Chain-of-thought prompting encourages LLMs to explain their reasoning step by step, "
    "which has been shown to improve performance on complex reasoning tasks.",
    "Semantic search uses meaning and context rather than exact keyword matching, "
    "enabling more relevant information retrieval from large document collections.",
    "The transformer architecture, introduced in 'Attention Is All You Need' (2017), "
    "uses self-attention mechanisms to process sequential data and powers most modern LLMs.",
]


class ResearcherAgent:
    """
    Performs hybrid RAG retrieval for each chapter and structures the results
    into ResearchPacket objects with fact registries.
    """

    def __init__(self, logger: ObservabilityLogger, cost_tracker: CostTracker, session_id: str):
        self.logger = logger
        self.cost = cost_tracker
        self.session_id = session_id
        self.llm = _get_llm()
        self._retriever: HybridRetriever | None = None

    def _init_retriever(self, extra_context: List[str] = None) -> HybridRetriever:
        embedder = Embedder()
        faiss_store = FAISSStore(session_id=self.session_id, dim=embedder.dim)

        retriever = HybridRetriever(faiss_store=faiss_store, embedder=embedder)

        # Index seed corpus + any extra context
        corpus = SEED_CORPUS[:]
        if extra_context:
            corpus.extend(extra_context)

        chunker = TextChunker(chunk_size=400, chunk_overlap=50)
        chunks = chunker.split_documents(corpus)
        retriever.index(chunks)

        self.logger.info("ResearcherAgent", f"Indexed {len(chunks)} chunks into retriever")
        return retriever

    def run(self, state: BookState) -> BookState:
        outline = state.get("outline", {})
        tone = state.get("tone_name", "conversational")
        chapters_plans = outline.get("chapters", [])
        genre = outline.get("genre", "Non-fiction")
        core_thesis = outline.get("core_thesis", "")

        if self._retriever is None:
            self._retriever = self._init_retriever()

        research_packets: Dict[int, Dict[str, Any]] = state.get("research_packets", {}) or {}

        logs = state.get("agent_logs", [])
        logs.append(f"[ResearcherAgent] Running research for {len(chapters_plans)} chapters...")
        state["agent_logs"] = logs

        system_tmpl, user_tmpl = PromptTemplates.researcher(tone)

        for plan in chapters_plans:
            ch_num = plan.get("chapter_number", 1)
            t0 = self.logger.agent_start("ResearcherAgent", chapter=ch_num)

            # Build retrieval query from chapter plan
            query = f"{plan.get('title', '')} {' '.join(plan.get('key_concepts', []))}"
            retrieved = self._retriever.retrieve_texts(query, top_k=5)

            self.logger.memory_read("FAISSStore", query, len(retrieved))

            context_str = "\n---\n".join(retrieved) if retrieved else "No context retrieved."

            user_prompt = user_tmpl.format(
                chapter_plan_json=json.dumps(plan, indent=2),
                retrieved_context=context_str,
                genre=genre,
                core_thesis=core_thesis,
            )

            self.logger.prompt_log("ResearcherAgent", system_tmpl, user_prompt, chapter=ch_num)

            packet = self._invoke_with_retry(system_tmpl, user_prompt, ch_num, retrieved)
            research_packets[ch_num] = packet

            self.logger.agent_end("ResearcherAgent", t0, chapter=ch_num)

        state["research_packets"] = research_packets
        logs = state.get("agent_logs", [])
        logs.append(f"[ResearcherAgent] ✓ Research complete for all {len(chapters_plans)} chapters")
        state["agent_logs"] = logs

        return state

    def _invoke_with_retry(self, system: str, user: str, chapter_num: int,
                           fallback_chunks: List[str], max_retries: int = 3) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                messages = [SystemMessage(content=system), HumanMessage(content=user)]
                response = self.llm.invoke(messages)
                raw = response.content.strip()

                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()

                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]

                parsed = json.loads(raw)
                parsed.setdefault("chapter_number", chapter_num)
                parsed.setdefault("retrieved_chunks", fallback_chunks)
                parsed.setdefault("citations", [])
                parsed.setdefault("fact_registry", [])
                parsed.setdefault("key_facts", [])
                parsed.setdefault("bm25_results", [])
                return parsed

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.error("ResearcherAgent", f"Attempt {attempt+1} failed: {e}", chapter=chapter_num)
                if attempt == max_retries - 1:
                    return {
                        "chapter_number": chapter_num,
                        "retrieved_chunks": fallback_chunks,
                        "citations": [],
                        "fact_registry": [],
                        "key_facts": fallback_chunks[:3],
                        "bm25_results": [],
                    }
                time.sleep(2 ** attempt)

        return {"chapter_number": chapter_num, "retrieved_chunks": fallback_chunks,
                "citations": [], "fact_registry": [], "key_facts": [], "bm25_results": []}
