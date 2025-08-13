import asyncio
from collections import defaultdict
from typing import List, Callable, Awaitable
import logging

logger = logging.getLogger(__name__)

class DynamicMicroBatcher:
    """
    Implements dynamic micro-batching using asyncio.
    Waits for a short duration (T_wait) or until a max batch size (B_max) is reached.
    """
    def __init__(self, T_wait_ms: float = 10.0, B_max: int = 32):
        self.T_wait = T_wait_ms / 1000.0
        self.B_max = B_max
        # Queues organized by model
        self.queues = defaultdict(list)
        # Locks for thread-safe access to queues/futures per model
        self.locks = defaultdict(asyncio.Lock)
        # Futures to hold results for individual requests
        self.futures = defaultdict(list)
        # Tasks managing the dispatch timer
        self.tasks = {}

    async def process(self, prompt: str, model: str, executor_func: Callable[[List[str], str], Awaitable[List[str]]]):
        """
        Adds a request to the batch queue and waits for the result.
        """
        async with self.locks[model]:
            # Create a future for this specific request
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self.queues[model].append(prompt)
            self.futures[model].append(future)

            # If max batch size reached (B_max), dispatch immediately
            if len(self.queues[model]) >= self.B_max:
                logger.info(f"Dispatching batch for {model} (B_max reached)")
                self._dispatch(model, executor_func)
            # Otherwise, ensure a task is scheduled to dispatch later (T_wait)
            elif model not in self.tasks or self.tasks[model].done():
                self.tasks[model] = loop.create_task(self._wait_and_dispatch(model, executor_func))

        return await future

    async def _wait_and_dispatch(self, model: str, executor_func: Callable):
        """Waits for T_wait before dispatching the batch."""
        await asyncio.sleep(self.T_wait)
        async with self.locks[model]:
            # Check if the batch hasn't been dispatched already (e.g., by B_max trigger)
            if len(self.queues[model]) > 0:
                logger.info(f"Dispatching batch for {model} (T_wait elapsed)")
                self._dispatch(model, executor_func)

    def _dispatch(self, model: str, executor_func: Callable):
        """Executes the batch and resolves the futures. Assumes lock is held."""
        assert self.locks[model].locked(), "Dispatch must be called under lock"
        batch = self.queues[model]
        futures = self.futures[model]

        # Clear the queue and futures list for the next batch
        self.queues[model] = []
        self.futures[model] = []
        
        # Cancel the timer task if it's running
        if model in self.tasks and not self.tasks[model].done():
            self.tasks[model].cancel()

        # Execute the batch asynchronously.
        # This task will resolve the futures with their corresponding streams.
        asyncio.create_task(self._execute_batch_and_resolve_streams(batch, model, futures, executor_func))

    async def _execute_batch_and_resolve_streams(self, batch, model, futures, executor_func):
        """
        Executes the batched call and resolves each future with its own stream.
        """
        try:
            # The executor now returns a list of async generators (streams)
            streams = await executor_func(batch, model)
            if len(streams) != len(futures):
                raise ValueError(f"Executor returned {len(streams)} streams for a batch of size {len(futures)}")

            for future, stream in zip(futures, streams):
                if not future.done():
                    future.set_result(stream)
        except Exception as e:
            logger.error(f"Batch execution failed for model {model}: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)