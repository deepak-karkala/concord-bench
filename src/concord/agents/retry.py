import asyncio
import random
import time
from typing import TypeVar

T = TypeVar("T")

_MAX_BACKOFF = 120.0


class AgentRetryError(Exception):
    pass


class AgentTimeoutError(AgentRetryError):
    pass


class AgentRateLimitError(AgentRetryError):
    pass


async def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: float | None = None,
) -> T:
    last_exception = None
    for attempt in range(max_retries):
        try:
            if timeout is not None:
                return await asyncio.wait_for(fn(), timeout=timeout)
            return await fn()
        except asyncio.TimeoutError as e:
            raise AgentTimeoutError(f"Request timed out after {timeout}s") from e
        except AgentRateLimitError:
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), _MAX_BACKOFF)
            time.sleep(delay)
            last_exception = AgentRateLimitError("Rate limited; retries exhausted")
        except Exception as e:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), _MAX_BACKOFF)
                time.sleep(delay)
                last_exception = e
                continue
            raise AgentRetryError(f"Request failed after {max_retries} retries: {e}") from e
    raise last_exception or AgentRetryError(f"Request failed after {max_retries} retries")
