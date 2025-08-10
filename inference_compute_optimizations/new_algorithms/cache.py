import time
from typing import Optional, Dict, Any
import hashlib
import json

class SimpleCache:
    """
    A basic in-memory LRU cache implementation for prompt caching.
    In a real-world scenario, this would be replaced by a persistent,
    distributed cache like Redis, potentially incorporating Semantic Caching.
    """
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache = {}
        self.access_times = {}

    def _normalize_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        # 1. Normalize prompt (simple whitespace normalization)
        normalized_prompt = " ".join(prompt.strip().split())

        # 2. Normalize parameters (sort keys)
        normalized_params = json.dumps(params, sort_keys=True)

        # 3. Create the combined key string
        key_string = f"model:{model}|params:{normalized_params}|prompt:{normalized_prompt}"

        # 4. Hash the key
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[str]:
        key = self._normalize_key(prompt, model, params)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, prompt: str, model: str, params: Dict[str, Any], response: str):
        key = self._normalize_key(prompt, model, params)
        if len(self.cache) >= self.capacity:
            self._evict()
        self.cache[key] = response
        self.access_times[key] = time.time()

    def _evict(self):
        # Find the least recently used item
        if not self.access_times:
            return
        oldest_key = min(self.access_times, key=self.access_times.get)
        if oldest_key:
            del self.cache[oldest_key]
            del self.access_times[oldest_key]