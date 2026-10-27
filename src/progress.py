"""Live progress bar for analysis.py, read from analysis.log.

    python src/progress.py
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import MODALITIES  # noqa: E402

LOG = Path(__file__).resolve().parent.parent / "analysis.log"
STEPS = MODALITIES + ["combined"]
WIDTH = 30

start = time.time()
while True:
    lines = LOG.read_text().splitlines() if LOG.exists() else []
    done = [s for s in STEPS if any(line.startswith(s) for line in lines)]
    finished = any(line.startswith("done in") for line in lines)
    errored = any("Error" in line for line in lines)

    n = len(done)
    bar = "█" * (WIDTH * n // len(STEPS)) + "░" * (WIDTH - WIDTH * n // len(STEPS))
    current = "finished" if finished else ("ERROR, see analysis.log" if errored else
                                           f"working on: {STEPS[n] if n < len(STEPS) else 'saving'}")
    elapsed = int(time.time() - start)
    print(f"\r[{bar}] {n}/{len(STEPS)}  {current:32s} {elapsed // 60}m{elapsed % 60:02d}s", end="", flush=True)

    if finished or errored:
        print()
        print("\n".join(lines[-10:]))
        break
    time.sleep(2)
