"""Background history writer: netCDF output of one interval runs in a spawned worker process while the main process prepares and
runs the next interval. netCDF-C is not thread-safe, so a process (not a thread) is used. One worker keeps the files in order.
LISJAX_FORCING_PROCS=1 (the same switch as the parallel forcing reader; set it in scripts without a __main__ guard) writes
synchronously instead."""
import os


class AsyncWriter:
    def __init__(self):
        self.futs = []
        self.ex = None
        n = int(os.environ.get("LISJAX_FORCING_PROCS", min(8, len(os.sched_getaffinity(0)))))
        if n > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self.ex = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))

    def submit(self, fn, *args):
        """fn must be a module-level function and args picklable."""
        if self.ex is None:
            fn(*args)
        else:
            self.futs.append(self.ex.submit(fn, *args))
            self.futs = [f for f in self.futs if not f.done() or f.result() is not None]    # drop finished ones, re-raise errors early

    def close(self):
        """Wait for all pending writes (re-raising any error), then stop the worker."""
        for f in self.futs:
            f.result()
        self.futs = []
        if self.ex is not None:
            self.ex.shutdown()
            self.ex = None
