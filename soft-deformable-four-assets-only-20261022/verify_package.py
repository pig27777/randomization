"""Check the classified package's file hashes with Python standard library."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
def main():
    expected=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    errors=[]
    for name,digest in expected.items():
        target=(ROOT/name).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            errors.append('Missing or invalid: '+name)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            errors.append('Changed: '+name)
    result={'passed':not errors,'files_checked':len(expected),'errors':errors,'physics_rerun':False}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(bool(errors))
if __name__=='__main__': raise SystemExit(main())
