"""Film texture and material coordinates shared by the asset and its scenes."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent


def surface_uv(nx, ny):
    """Material-space UVs: the print follows each panel as it deforms."""
    return np.asarray([(i / nx, j / ny) for _ in (0, 1)
                       for j in range(ny + 1) for i in range(nx + 1)], dtype=np.float32)


def apply_film_texture(root, nx, ny, *, absolute_paths=False):
    assets = root.find("asset")
    if assets is None:
        assets = ET.Element("asset")
        root.insert(2, assets)
    texture = assets.find("texture[@name='bag_film_texture']")
    if texture is None:
        texture = ET.SubElement(assets, "texture", name="bag_film_texture", type="2d")
    texture.set("file", str(ROOT / "textures/bag_film.png") if absolute_paths else "textures/bag_film.png")
    material = assets.find("material[@name='bag_film_material']")
    if material is None:
        material = ET.SubElement(assets, "material", name="bag_film_material")
    material.attrib.update(texture="bag_film_texture", texuniform="false", texrepeat="1 1",
                           rgba=".82 .90 .86 1", specular=".35", shininess=".25")
    flex = root.find("worldbody/flexcomp[@name='bag']")
    flex.set("material", "bag_film_material")
    flex.set("rgba", ".82 .90 .86 .85")
    flex.set("texcoord", " ".join(f"{v:.8g}" for v in surface_uv(nx, ny).ravel()))
