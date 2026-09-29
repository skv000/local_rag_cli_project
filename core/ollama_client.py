import json
import urllib.error
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Union


class OllamaClient:
    """Client for local Ollama HTTP API using standard Python libraries."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> bool:
        """Verifies if the Ollama server is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as res:
                return res.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns the list of locally available model names."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as res:
                data = json.loads(res.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise ConnectionError(f"Failed to fetch models from Ollama: {e}")

    def get_embedding(self, text: str, model: str = "embeddinggemma:latest") -> List[float]:
        """
        Generates embedding vector for a single text string using Ollama.
        """
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
                embedding = data.get("embedding")
                if not embedding:
                    raise ValueError(f"No embedding returned for text by model {model}")
                return embedding
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding error ({e.code}): {err_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama embedding endpoint: {e}")

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "embeddinggemma:latest",
        show_progress: bool = True,
    ) -> List[List[float]]:
        """Generates embeddings for a batch of texts."""
        embeddings: List[List[float]] = []
        total = len(texts)
        for idx, text in enumerate(texts):
            if show_progress:
                print(f"\r  Generating embeddings: {idx + 1}/{total}...", end="", flush=True)
            emb = self.get_embedding(text, model=model)
            embeddings.append(emb)
        if show_progress and total > 0:
            print("\r  Generating embeddings: complete!          ")
        return embeddings

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "llama3.2:3b",
        stream: bool = False,
        temperature: float = 0.2,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates text completion using the specified local Ollama model.
        Supports both one-shot return (str) and real-time streaming (Generator).
        """
        payload_dict: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload_dict["system"] = system

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            res = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama generation error ({e.code}): {err_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama generate endpoint: {e}")

        if not stream:
            with res:
                data = json.loads(res.read().decode("utf-8"))
                return data.get("response", "")

        def stream_generator() -> Generator[str, None, None]:
            with res:
                for line in res:
                    if line:
                        chunk_data = json.loads(line.decode("utf-8"))
                        token = chunk_data.get("response", "")
                        yield token
                        if chunk_data.get("done", False):
                            break

        return stream_generator()
