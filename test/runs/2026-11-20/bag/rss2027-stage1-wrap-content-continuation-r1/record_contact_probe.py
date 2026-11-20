"""Change just the observed dominant support or payload contact normal response."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch
from xml.etree import ElementTree as ET

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.with_name("rss2027-stage1-randomized-top-local-mesh-r1")))
import refine_parcel  # noqa: E402

DOMINANT = {"sample-074": "payload_geom_0", "sample-078": "table"}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    args, _ = parser.parse_known_args()
    name = DOMINANT[args.config.parent.name]
    backend = refine_parcel.backend
    build = backend.parcel_world

    def stiffer_contact(cfg):
        if cfg.contact_time != .001 or cfg.dt != .000025:
            raise ValueError("Expected the archived half-step contact setup")
        parcel, xml = build(cfg)
        root = ET.fromstring(xml)
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f"MuJoCo assembly missing {name}")
        if tuple(map(float, geom.get("solref", ".001 1").split())) != (.001, 1.):
            raise ValueError(f"Unexpected original contact response for {name}")
        geom.set("solref", ".0005 1")
        return parcel, ET.tostring(root, encoding="unicode")

    with patch.object(backend, "parcel_world", stiffer_contact):
        refine_parcel.main()


if __name__ == "__main__":
    main()
