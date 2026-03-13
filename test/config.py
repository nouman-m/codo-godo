"""
Configuration settings for the RAG testing system.
"""

API_URL = "http://localhost:8000"
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"

QUERIES_TEST_FILE = "test/queries_small.txt"
QUERIES_FILE = "test/queries.txt"
RESULTS_DIR = "test/results"
QUERY_RESULTS_FILE = "test/results/query_results.json"
SCORES_FILE = "test/results/scores.json"

QUERY_TOP_K = 5
