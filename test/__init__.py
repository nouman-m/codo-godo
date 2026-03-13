"""
Testing module for the Codo-Godo RAG system.

This module provides tools for evaluating the RAG system using LLM-based scoring.
"""

from test import config, scorer
from test.scorer import ScoreResult, get_scorer, OllamaScorer
from test.run_queries import run_queries, load_queries

__all__ = [
    "config",
    "scorer",
    "ScoreResult",
    "get_scorer",
    "OllamaScorer",
    "run_queries",
    "load_queries",
]
