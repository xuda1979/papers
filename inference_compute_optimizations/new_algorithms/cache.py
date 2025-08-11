import time
from typing import Optional, Dict, Any
import hashlib
import json

class InMemoryLRUCache:
    """
    A basic in-memory LRU cache implementation for prompt caching.
    This implementation is for demonstration purposes and is not thread-safe.
    In a real-world scenario, this would be replaced by a persistent,
    distributed cache like Redis.
    """
    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache: Dict[str, str] = {}
        self.access_times: Dict[str, float] = {}

    def _normalize_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Creates a deterministic, hash-based key from the request details.
        """
        # 1. Normalize prompt (simple whitespace normalization)
        normalized_prompt = " ".join(prompt.strip().split())

        # 2. Normalize parameters by sorting keys for consistent ordering
        normalized_params = json.dumps(params, sort_keys=True)

        # 3. Create the combined key string
        key_string = f"model:{model}|params:{normalized_params}|prompt:{normalized_prompt}"

        # 4. Hash the key to get a fixed-length, uniform key
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def get(self, prompt: str, model: str, params: Dict[str, Any]) -> Optional[str]:
        key = self._normalize_key(prompt, model, params)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, prompt: str, model: str, params: Dict[str, Any], response: str):
        key = self._normalize_key(prompt, model, params)
        if len(self.cache) >= self.capacity and key not in self.cache:
            self._evict()
        self.cache[key] = response
        self.access_times[key] = time.time()

    def _evict(self):
        """
        Evicts the least recently used item from the cache.
        Note: This is a simple, non-performant implementation for demonstration.
        A production-grade LRU cache would use a more efficient data structure
        like a doubly linked list + hash map.
        """
        if not self.access_times:
            return

        # Find the key with the oldest access time
        oldest_key = min(self.access_times, key=self.access_times.get)

        # Remove it from both the cache and access time records
        if oldest_key:
            del self.cache[oldest_key]
            del self.access_times[oldest_key]