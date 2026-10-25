"""Bounded look-ahead map: decode the next audio files on worker threads while the GPU
scores the current one, without ever holding more than `depth` decoded files."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def prefetch(fn, items, workers=4, depth=16):
    """Yield (item, result, error) in input order; `error` is the exception fn raised."""
    def run(item):
        try:
            return item, fn(item), None
        except Exception as e:  # noqa: BLE001 -- handed to the caller per item
            return item, None, e

    it = iter(items)
    with ThreadPoolExecutor(workers) as pool:
        window = deque(pool.submit(run, x) for _, x in zip(range(depth), it))
        while window:
            done = window.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                window.append(pool.submit(run, nxt))
            yield done
