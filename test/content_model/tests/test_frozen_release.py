"""Release integrity survives driver moves and still rejects source corruption."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from frozen_release import verify


class FrozenReleaseTest(unittest.TestCase):
    def fixture(self, root):
        model = root / "contents/segment"
        tools = root / "test/content_model"
        model.mkdir(parents=True)
        tools.mkdir(parents=True)
        sources = {
            "contents/segment/model.py": b"MODEL = 1\n",
            "contents/segment/validate_model.py": b"check_model()\n",
        }
        archive = model / "release.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, data in sources.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
        (model / "model.py").write_bytes(sources["contents/segment/model.py"])
        (tools / "validate_model.py").write_text("# relocated driver\ncheck_model()\n")
        tag = dict(archive=archive.name,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in sources.items()})
        return model, tools, tag

    def test_relocated_driver_keeps_original_release_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, tools, tag = self.fixture(root)
            result = verify(tag, model, root, tools)
            self.assertTrue(result["intact"])
            self.assertIn("contents/segment/validate_model.py", result["relocated_drivers"])

    def test_changed_or_missing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, tools, tag = self.fixture(root)
            (model / "model.py").write_text("MODEL = 2\n")
            self.assertFalse(verify(tag, model, root, tools)["intact"])
            (model / "model.py").unlink()
            self.assertFalse(verify(tag, model, root, tools)["intact"])

    def test_changed_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, tools, tag = self.fixture(root)
            with (model / tag["archive"]).open("ab") as stream:
                stream.write(b"changed archive")
            result = verify(tag, model, root, tools)
            self.assertFalse(result["intact"])
            self.assertFalse(result["archive_intact"])
