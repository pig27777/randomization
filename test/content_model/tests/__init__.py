"""Discover content tests with their model and experiment imports."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT / "contents/segment"), str(PROJECT / "test/content_model")]
