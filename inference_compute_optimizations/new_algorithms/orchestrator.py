import asyncio
from typing import Dict, Any, List
import logging

# Assuming these modules are in the same package
try:
    from .cache import SimpleCache
    from .router import HeuristicRouter
    from .batcher import DynamicMicroBatcher
except ImportError:
    from cache import SimpleCache
    from router import HeuristicRouter
    from batcher import DynamicMicroBatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceOrchestrator:
    """
    Combines caching, routing, and batching into a single asynchronous pipeline.
    """
    def __init__(self, router_config: Dict[str, Any], batcher_config: Dict[str, Any]):
        self.cache = SimpleCache()
        self.router = HeuristicRouter(router_config)
        self.batcher = DynamicMicroBatcher(**batcher_config)
        # The executor function simulates the actual LLM API call
        self.executor_func = self._mock_llm_executor
        self.stats = {"cache_hits": 0, "requests_processed": 0, "api_calls": 0}

    async def _mock_llm_executor(self, batch: List[str], model: str) -> List[str]:
        """Simulates a batched API call to an LLM."""
        logger.info(f"\n[API CALL] Executing batch (Size: {len(batch)}) on Model: {model}")
        self.stats["api_calls"] += 1
        # Simulate latency (e.g., 100ms base + 10ms per request)
        await asyncio.sleep(0.1 + len(batch) * 0.01) 
        logger.info(f"[API CALL] Finished batch on {model}")
        return [f"Response from {model} for prompt: {prompt[:15]}..." for prompt in batch]

    async def process_request(self, prompt: str, params: Dict[str, Any]) -> str:
        self.stats["requests_processed"] += 1
        
        # 1. Routing
        # We must route first to determine the model for the cache key and batch queue.
        model = self.router.route(prompt)

        # 2. Caching (Check)
        cached_response = self.cache.get(prompt, model, params)
        if cached_response:
            logger.info(f"[Cache] HIT for prompt: {prompt[:15]}...")
            self.stats["cache_hits"] += 1
            return cached_response
        
        logger.info(f"[Cache] MISS. Routing prompt '{prompt[:15]}...' to {model}.")

        # 3. Batching
        # The batcher handles the queuing and execution via the provided executor_func.
        # We await the future returned by the batcher.
        response = await self.batcher.process(prompt, model, self.executor_func)

        # 4. Caching (Update)
        self.cache.set(prompt, model, params, response)

        return response

# Example Usage (if run directly):
async def main():
    print("Starting InferenceOrchestrator example...")
    router_config = {"length_threshold": 50, "easy_model": "FastModel-A", "complex_model": "StrongModel-B"}
    # T_wait_ms=20ms allows time for requests to accumulate
    batcher_config = {"T_wait_ms": 20.0, "B_max": 4}
    orchestrator = InferenceOrchestrator(router_config, batcher_config)

    params = {"temperature": 0.5}
    tasks = []

    # Simulate concurrent requests
    
    # Request 1 (Easy)
    tasks.append(orchestrator.process_request("What is 2+2?", params))
    # Request 2 (Complex)
    tasks.append(orchestrator.process_request("Analyze the implications of P=NP on modern cryptography.", params))
    # Request 3 (Easy, likely batched with 1)
    tasks.append(orchestrator.process_request("Capital of Spain?", params))
    
    # Wait briefly to allow these requests to enter the batching window
    await asyncio.sleep(0.005)

    # Request 4 (Duplicate of 1)
    # In this async setup, if the prompt is identical and the model is the same, 
    # it will join the batch queue. The cache is only updated *after* the batch returns.
    tasks.append(orchestrator.process_request("What is 2+2?", params))

    # Request 5 (Complex, likely batched with 2)
    tasks.append(orchestrator.process_request("Debug this code: def f(x): return 1/0", params))

    # Request 6 (Easy, might trigger B_max=4 for FastModel-A)
    tasks.append(orchestrator.process_request("Translate 'Hello' to French.", params))

    results = await asyncio.gather(*tasks)

    # Wait for potential subsequent cache hits to resolve
    await asyncio.sleep(0.2) 
    print("\n--- Final Test: Cache Check ---")
    # Request 7 (Duplicate of 2, definitely Cache HIT now)
    result_cached = await orchestrator.process_request("Analyze the implications of P=NP on modern cryptography.", params)
    
    print("\n--- Results ---")
    for i, result in enumerate(results):
        print(f"Request {i+1}: {result}")
    print(f"Request 7 (Cached): {result_cached}")

    print("\n--- Stats ---")
    print(f"Total Requests: {orchestrator.stats['requests_processed']}")
    print(f"Cache Hits: {orchestrator.stats['cache_hits']}")
    print(f"API Calls (Batches): {orchestrator.stats['api_calls']}")


if __name__ == "__main__":
    # To run this example, ensure cache.py, router.py, and batcher.py are in the same directory.
    asyncio.run(main())