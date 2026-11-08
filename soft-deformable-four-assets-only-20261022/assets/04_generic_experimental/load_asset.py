"""Public entry point for the reusable bag and configurable contents."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_DEFAULT = object()


def load_bag(variant=None, *, payload=_DEFAULT, contents=None, params=None,
             support_z=.045, origin_xy=(0., 0.)):
    """Empty by default; payload='book'/'sphere'/dict or contents=[...]."""
    from bag_asset import BagAsset
    if variant not in (None, 'filled', 'empty'):
        raise ValueError("variant must be 'filled' or 'empty'")
    if contents is not None and payload is not _DEFAULT:
        raise ValueError('Choose payload or contents, not both')
    if variant == 'empty':
        if contents or (payload is not _DEFAULT and payload is not None):
            raise ValueError('An empty bag cannot contain objects')
        chosen = []
    elif contents is not None:
        chosen = contents
    elif payload is _DEFAULT:
        chosen = json.loads((ROOT / 'config.json').read_text()).get('contents', ['book']) if variant == 'filled' else []
    else:
        chosen = payload
    return BagAsset(chosen, params=params, support_z=support_z, origin_xy=origin_xy)
