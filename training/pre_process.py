from __future__ import annotations

import sys
from pathlib import Path

try:
    from training.agri_ai.feature_extraction.preprocessing import *  # noqa: F401,F403
    from training.agri_ai.feature_extraction.preprocessing import main
except ImportError:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from training.agri_ai.feature_extraction.preprocessing import *  # noqa: F401,F403
    from training.agri_ai.feature_extraction.preprocessing import main


if __name__ == "__main__":
    main()

