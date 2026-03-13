"""
LLM-based scoring for RAG evaluation.

Provides an abstract ScoreLLM class and implementations for different LLM providers.
Currently supports Ollama; OpenAI can be added later.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import requests

from test import config


@dataclass
class ScoreResult:
    """Result of scoring a query against retrieved chunks."""

    question: str
    answered: bool
    score: int
    reasoning: str


class ScoreLLM(ABC):
    """Abstract base class for LLM-based scoring."""

    @abstractmethod
    def score(self, question: str, chunks: list[dict]) -> ScoreResult:
        """
        Evaluate whether the retrieved chunks answer the question.

        Args:
            question: The original question asked
            chunks: List of retrieved chunks from the RAG system

        Returns:
            ScoreResult with answered flag, score (1-10), and reasoning
        """
        pass


class OllamaScorer(ScoreLLM):
    """Ollama-based scorer using local LLM."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL

    def score(self, question: str, chunks: list[dict]) -> ScoreResult:
        chunks_text = self._format_chunks(chunks)

        prompt = f"""You are an expert Godot GDScript evaluator. Your task is to evaluate whether the retrieved context answers the user's question.

        Question: {question}
        
        Retrieved Context:
        {chunks_text}
        
        Evaluate the following:
        
        1. Is the question answered? (Does the context contain relevant information?)
        2. Give a score from 1-10 based on relevance and quality.
        
        Respond in JSON format:
        {{
            "answered": true/false,
            "score": 1-10,
            "reasoning": "brief explanation"
        }}
        
        Only respond with valid JSON, no other text."""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(f"Ollama request failed: {response.text}")

        result = response.json()
        response_text = result.get("response", "")

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = self._parse_fallback(response_text)

        return ScoreResult(
            question=question,
            answered=parsed.get("answered", False),
            score=min(max(parsed.get("score", 5), 1), 10),
            reasoning=parsed.get("reasoning", "No reasoning provided"),
        )

    def _format_chunks(self, chunks: list[dict]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")[:500]
            source = chunk.get("source", "unknown")
            source_type = chunk.get("source_type", "unknown")
            formatted.append(f"[Chunk {i}] (Source: {source}, Type: {source_type})\n{text}\n")
        return "\n".join(formatted)

    def _parse_fallback(self, response_text: str) -> dict:
        answered = "yes" in response_text.lower() or "true" in response_text.lower()

        import re
        score_match = re.search(r"\b([1-9]|10)\b", response_text)
        score = int(score_match.group(1)) if score_match else 5

        return {
            "answered": answered,
            "score": score,
            "reasoning": response_text[:200],
        }


def get_scorer(scorer_type: str = "ollama", **kwargs) -> ScoreLLM:
    """
    Factory function to get a scorer instance.

    Args:
        scorer_type: Type of scorer ("ollama" or "openai")
        **kwargs: Additional arguments passed to the scorer constructor

    Returns:
        ScoreLLM instance
    """
    if scorer_type == "ollama":
        return OllamaScorer(**kwargs)
    elif scorer_type == "openai":
        raise NotImplementedError("OpenAI scorer not implemented yet")
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")
