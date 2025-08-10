from typing import AsyncGenerator, List, Optional
import asyncio

class EarlyStopStreamer:
    """
    A utility class to wrap a streaming response (AsyncGenerator) and terminate 
    it early based on predefined criteria.
    """
    def __init__(self, stop_phrases: Optional[List[str]] = None, stop_on_json_end: bool = False):
        self.stop_phrases = stop_phrases or []
        # Add common verbose closing remarks
        self.stop_phrases.extend(["I hope this helps!", "Let me know if you have any other questions."])
        self.stop_on_json_end = stop_on_json_end

    async def process_stream(self, stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Wraps the stream, monitors content, and stops generation if criteria met.
        """
        accumulated_content = ""
        # We look back only a limited window to detect phrases spanning across chunks
        lookback_window_size = max(len(p) for p in self.stop_phrases) * 2 if self.stop_phrases else 100

        async for chunk in stream:
            yield chunk
            accumulated_content += chunk
            
            # Maintain the lookback window
            recent_content = accumulated_content[-lookback_window_size:]

            # Check for stop phrases
            if self._check_stop_phrases(recent_content):
                print(f"\n[Early Stop Triggered by phrase]")
                return # Terminate the generator

            # Check for JSON completion (simple heuristic)
            if self.stop_on_json_end:
                if self._is_json_complete(accumulated_content):
                    print("\n[Early Stop Triggered by JSON completion]")
                    return

    def _check_stop_phrases(self, content: str) -> bool:
        for phrase in self.stop_phrases:
            # Check if the phrase is present in the recent content
            if phrase in content:
                return True
        return False

    def _is_json_complete(self, content: str) -> bool:
        """A very basic heuristic for checking if a JSON object/array is complete."""
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            # Basic check for balanced braces (insufficient for complex nested JSON, but fast)
            if content.count("{") == content.count("}"):
                return True
        if content.startswith("[") and content.endswith("]"):
            if content.count("[") == content.count("]"):
                return True
        return False