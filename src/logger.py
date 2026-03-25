import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger("train2")
