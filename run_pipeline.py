#!/usr/bin/env python
"""Single entry point to run the full pipeline: train -> evaluate -> detect_anomalies -> make_maps.

Run from anywhere with:
    python run_pipeline.py
    python run_pipeline.py --skip-train --skip-evaluate   # only detect anomalies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.run_all import main

if __name__ == "__main__":
    main()
