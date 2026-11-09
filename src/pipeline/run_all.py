"""Single entry point to run the full pipeline: train -> evaluate -> detect_anomalies -> make_maps.

Usage:
    python -m src.pipeline.run_all
    python -m src.pipeline.run_all --skip-train --skip-evaluate   # only detect anomalies
"""

import argparse
import time

from src.pipeline import train, evaluate, detect_anomalies, make_maps
from src.common.log import get_logger, section

logger = get_logger(__name__)

STAGES = [
    ("train", train.main),
    ("evaluate", evaluate.main),
    ("detect_anomalies", detect_anomalies.main),
    ("make_maps", make_maps.main),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full embeddings pipeline.")
    for name, _ in STAGES:
        parser.add_argument(f"--skip-{name.replace('_', '-')}", action="store_true",
                            help=f"Skip the '{name}' stage.")
    return parser.parse_args()


def main():
    args = parse_args()

    for name, run in STAGES:
        if getattr(args, f"skip_{name}"):
            logger.info(f"Skipping stage: {name}")
            continue

        section(f"STAGE: {name}")
        start = time.time()
        run()
        logger.info(f"Stage '{name}' finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
