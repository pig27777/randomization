"""Drive the standard top approach from a geometrically prepared package."""

from pathlib import Path
import sys
from unittest.mock import patch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-top-ridge-selection-r1")))
from record_faster_approach import FasterApproach, protocol  # noqa: E402
from record_ridge import refine_parcel  # noqa: E402


if __name__ == "__main__":
    with patch.object(protocol, "NormalPress", FasterApproach):
        refine_parcel.main()
