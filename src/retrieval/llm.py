"""
LLM client for generating responses.

Supports both Ollama (local) and OpenCode (cloud) providers.
"""

import json
from typing import Generator

import requests

from src import config


class OpenCodeClient:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None):
        self.model = model or config.OPENCODE_MODEL
        self.base_url = base_url or config.OPENCODE_BASE_URL
        self.api_key = api_key or config.OPENCODE_API_KEY
        self.temperature = config.OPENCODE_TEMPERATURE

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "temperature": temperature or self.temperature,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=180,
        )

        if response.status_code != 200:
            raise Exception(f"OpenCode request failed: {response.text}")

        return response.json()

    def stream_generate(
        self,
        prompt: str,
        temperature: float = None,
    ) -> Generator[str, None, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": temperature or self.temperature,
        }

        with requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=180,
        ) as response:
            if response.status_code != 200:
                raise Exception(f"OpenCode request failed: {response.text}")

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                            if data["choices"][0].get("finish_reason") == "stop":
                                break


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


def get_opencode_client(model: str = None, base_url: str = None, api_key: str = None):
    return OpenCodeClient(model=model, base_url=base_url, api_key=api_key)


def get_llm_client(provider: str = None, **kwargs):
    provider = provider or config.DEFAULT_LLM_PROVIDER
    if provider == "opencode":
        return get_opencode_client(**kwargs)
    else:
        return get_client(**kwargs)
