from __future__ import annotations

import sys
from pathlib import Path

try:
    from training.image_prediction.ssgrow_transfer_training import main
except ImportError:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.image_prediction.ssgrow_transfer_training import main


if __name__ == "__main__":
    main(default_season="kharif")

