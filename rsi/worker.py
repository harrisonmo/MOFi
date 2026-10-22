"""Execute a bounded scientific replay outside the controller process."""

import json
import os
import random
import sys
from pathlib import Path


def main():
    if "RSI_SCIENCE_SEED" in os.environ:
        import numpy as np
        seed = int(os.environ["RSI_SCIENCE_SEED"])
        random.seed(seed)
        np.random.seed(seed)
    from rsi.science import run_task

    task_path, actions_path, output = map(Path, sys.argv[1:4])
    previous = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")) if len(sys.argv) > 4 else None
    task = json.loads(task_path.read_text(encoding="utf-8"))
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    result = run_task(task, actions, output, previous=previous)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
