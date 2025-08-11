import asyncio
from typing import Dict, Any, List, AsyncGenerator
import logging

# Assuming these modules are in the same package
try:
    from .cache import SimpleCache
    from .router import HeuristicRouter
    from .batcher import DynamicMicroBatcher
    from .early_stop import EarlyStopStreamer
except ImportError:
    from cache import SimpleCache
    from router import HeuristicRouter
    from batcher import DynamicMicroBatcher
    from early_stop import EarlyStopStreamer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceOrchestrator:
    """
    Combines caching, routing, batching, and early-stop streaming into a single pipeline.
    """
    def __init__(self, router_config: Dict[str, Any], batcher_config: Dict[str, Any], early_stop_config: Dict[str, Any] = None):
        self.cache = SimpleCache()
        self.router = HeuristicRouter(router_config)
        self.batcher = DynamicMicroBatcher(**batcher_config)

        if early_stop_config:
            self.early_stopper = EarlyStopStreamer(**early_stop_config)
        else:
            self.early_stopper = None

        self.executor_func = self._mock_llm_executor
        self.stats = {"cache_hits": 0, "requests_processed": 0, "api_calls": 0, "early_stops": 0}

    async def _mock_llm_executor(self, batch: List[str], model: str) -> List[AsyncGenerator[str, None]]:
        """Simulates a batched API call that returns a list of streams."""
        logger.info(f"\n[API CALL] Executing batch (Size: {len(batch)}) on Model: {model}")
        self.stats["api_calls"] += 1

        streams = []
        for prompt in batch:
            async def stream_generator(p: str, m: str) -> AsyncGenerator[str, None]:
                # Simulate token generation
                response = f"Response from {m} for prompt: {p[:30]}... This response is being streamed. I hope this helps!"
                tokens = response.split()
                for token in tokens:
                    await asyncio.sleep(0.05) # Simulate network latency per token
                    yield token + " "

            streams.append(stream_generator(prompt, model))

        return streams

    async def process_request(self, prompt: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Processes a request, yielding streamed response chunks.
        Handles routing, caching, batching, and early stopping.
        """
        self.stats["requests_processed"] += 1
        
        model = self.router.route(prompt)
        cached_response = self.cache.get(prompt, model, params)

        if cached_response:
            logger.info(f"[Cache] HIT for prompt: {prompt[:15]}...")
            self.stats["cache_hits"] += 1
            yield cached_response
            return

        logger.info(f"[Cache] MISS. Routing prompt '{prompt[:15]}...' to {model}.")

        raw_stream = await self.batcher.process(prompt, model, self.executor_func)

        # Wrap the raw stream with the early stopper if configured
        if self.early_stopper:
            processed_stream = self.early_stopper.process_stream(raw_stream)
        else:
            processed_stream = raw_stream

        full_response_chunks = []
        original_len = 0
        final_len = 0

        full_response_chunks = []
        async for chunk in processed_stream:
            yield chunk
            full_response_chunks.append(chunk)

        if self.early_stopper and self.early_stopper.stopped:
            self.stats["early_stops"] += 1

        final_response = "".join(full_response_chunks)
        self.cache.set(prompt, model, params, final_response)

# Example Usage (if run directly):
async def main():
    print("Starting InferenceOrchestrator example with streaming...")
    router_config = {"length_threshold": 50, "easy_model": "FastModel-A", "complex_model": "StrongModel-B"}
    batcher_config = {"T_wait_ms": 20.0, "B_max": 4}
    early_stop_config = {"stop_phrases": ["I hope this helps!"]}

    orchestrator = InferenceOrchestrator(router_config, batcher_config, early_stop_config)

    params = {"temperature": 0.5}

    # --- Test 1: Batching and Streaming ---
    print("\n--- Test 1: Concurrent requests to trigger batching ---")
    
    async def stream_and_print(request_id, prompt):
        print(f"Request {request_id} (sent): '{prompt[:30]}...'")
        response_stream = orchestrator.process_request(prompt, params)
        full_response = ""
        async for chunk in response_stream:
            full_response += chunk
        print(f"Request {request_id} (full response): {full_response.strip()}")

    # Simulate three concurrent requests
    tasks = [
        stream_and_print(1, "What is the capital of France?"),
        stream_and_print(2, "Explain the theory of relativity step-by-step in detail."),
        stream_and_print(3, "What is 1+1?")
    ]
    await asyncio.gather(*tasks)

    # --- Test 2: Caching ---
    print("\n--- Test 2: Cached request ---")
    await stream_and_print(4, "What is the capital of France?")
    
    # --- Test 3: Early Stopping ---
    print("\n--- Test 3: Early stopping ---")
    # This prompt includes a stop phrase
    await stream_and_print(5, "Tell me about black holes. I hope this helps!")


    print("\n--- Final Stats ---")
    print(f"Total Requests: {orchestrator.stats['requests_processed']}")
    print(f"Cache Hits: {orchestrator.stats['cache_hits']}")
    print(f"API Calls (Batches): {orchestrator.stats['api_calls']}")
    print(f"Early Stops: {orchestrator.stats['early_stops']}")


if __name__ == "__main__":
    asyncio.run(main())