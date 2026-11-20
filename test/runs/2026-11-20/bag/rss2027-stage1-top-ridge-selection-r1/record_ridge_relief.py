"""Apply the already tested closure-pressure relief at the selected ridge."""

from pathlib import Path
import sys
from unittest.mock import patch

from record_ridge import main as record  # noqa: E402

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-relief-r1")))
import record_relief  # noqa: E402


if __name__ == "__main__":
    with patch.object(record_relief, "record", record):
        record_relief.main()
