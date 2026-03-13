"""
Ollama LLM client for generating responses.

Provides a generic HTTP client for interacting with Ollama's /api/generate endpoint.
"""

import json
from typing import Generator

import requests

from src import config


class OllamaClient:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.temperature = config.OLLAMA_TEMPERATURE

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        format: str = "json",
        temperature: float = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "format": format,
            "temperature": temperature or self.temperature,
        }
        print()
        print()
        print(json.dumps(payload, indent=2))
        print()
        print()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180,
        )

        if response.status_code != 200:
            raise Exception(f"Ollama request failed: {response.text}")

        return response.json()

    def stream_generate(
        self,
        prompt: str,
        format: str = "json",
        temperature: float = None,
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "format": format,
            "temperature": temperature or self.temperature,
        }
        print()
        print()
        print(json.dumps(payload, indent=2))
        print()
        print()

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=180,
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Ollama request failed: {response.text}")

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break


def get_client(model: str = None, base_url: str = None) -> OllamaClient:
    return OllamaClient(model=model, base_url=base_url)
