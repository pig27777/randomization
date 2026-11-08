"""Rebuild an authored empty gray PE mailing bag; all geometric units are metres.

The two flat reference panels share welded side/bottom vertices. A very small
separation and approximately isometric ripple belong only to initial_vertices;
neither the observed photo bulge nor a hidden payload becomes a rest shape.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FLAP = .350, .450, .050
NX, NY, NF = 14, 18, 2
THICKNESS, DENSITY = 50e-6, 920.0
SOURCE_URL = ('https://optimaxproducts.com/products/polythene-mailing-bag-350-x-475mm-'
              '50mm-flap-50mu-film-thickness-grey-500-per-box')


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build_geometry():
    vertices, faces, uv, panels, grid = [], [], [], [], {}
    for side in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                if side and (i in (0, NX) or j == 0):
                    grid[side, i, j] = grid[0, i, j]
                    continue
                grid[side, i, j] = len(vertices)
                vertices.append([WIDTH * (i / NX - .5), HEIGHT * (j / NY - .5), 0.0])
                uv.append([(.02 + .46 * i / NX) if side == 0 else (.52 + .46 * i / NX),
                           .04 + .82 * j / NY])
        for j in range(NY):
            for i in range(NX):
                a, b, c, d = [grid[side, u, v] for u, v in [(i,j), (i+1,j), (i,j+1), (i+1,j+1)]]
                # Avoid a duplicate all-welded corner triangle at bottom right.
                triangles = [(a,b,c), (b,d,c)] if j == 0 and i == NX-1 else [(a,b,d), (a,d,c)]
                faces.extend(tri if side == 0 else tri[::-1] for tri in triangles)
                panels.extend([side, side])
    previous = [grid[1, i, NY] for i in range(NX + 1)]
    for j in range(1, NF + 1):
        row = []
        for i in range(NX + 1):
            row.append(len(vertices))
            vertices.append([WIDTH * (i / NX - .5), HEIGHT / 2 + FLAP * j / NF, 0.0])
            uv.append([.52 + .46 * i / NX, .86 + .10 * j / NF])
        for i in range(NX):
            faces.extend([(previous[i], row[i+1], row[i]),
                          (previous[i], previous[i+1], row[i+1])])
            panels.extend([2, 2])
        previous = row
    q, f = np.array(vertices), np.array(faces, dtype=np.int32)
    area = .5 * np.linalg.norm(np.cross(q[f[:,1]]-q[f[:,0]], q[f[:,2]]-q[f[:,0]]), axis=1)
    masses = np.zeros(len(q))
    for k in range(3):
        np.add.at(masses, f[:,k], area * THICKNESS * DENSITY / 3)
    # A piecewise arc-length-preserving corrugation, extruded along Y. Its
    # projection is narrower than the flat sheet; no reference edge is stretched.
    theta = .25 * np.sin(4 * np.pi * (np.arange(NX) + .5) / NX)
    sx = np.concatenate(([0.0], np.cumsum(WIDTH / NX * np.cos(theta))))
    sz = np.concatenate(([0.0], np.cumsum(WIDTH / NX * np.sin(theta))))
    sx -= (sx.min() + sx.max()) / 2
    initial = q.copy()
    for idx, point in enumerate(q):
        i = int(round((point[0] / WIDTH + .5) * NX))
        initial[idx,0], initial[idx,2] = sx[i], sz[i]
    # The front/back gap is only collision preparation, not film thickness or
    # an inflated rest shape. Welds remain joined; there are no fixed nodes.
    gap = .0020
    for side in range(2):
        for j in range(NY + 1):
            for i in range(NX + 1):
                if i in (0, NX) or j == 0:
                    continue
                idx = grid[side,i,j]
                initial[idx,2] += (1 if side == 0 else -1) * gap/2 * np.sin(np.pi*i/NX) * min(1, j/2)
    # Continue the back layer smoothly into its open flap.
    for idx in range(previous[0] - (NF-1)*(NX+1), len(q)):
        x, y, _ = q[idx]
        initial[idx,2] -= gap/2*np.sin(np.pi*(x/WIDTH+.5))
    initial[:,2] += .025 - initial[:,2].min()
    edge_set = {tuple(sorted((int(a),int(b)))) for tri in f for a,b in zip(tri,np.roll(tri,-1))}
    edges = np.array(sorted(edge_set))
    l0 = np.linalg.norm(q[edges[:,1]]-q[edges[:,0]], axis=1)
    li = np.linalg.norm(initial[edges[:,1]]-initial[edges[:,0]], axis=1)
    strain = float(np.max(np.abs(li/l0-1)))
    if strain > .005:
        raise ValueError(f'Authored initial edge strain {strain:.4%} exceeds 0.5%')
    np.savez_compressed(ROOT/'geometry.npz', vertices=q, faces=f, uv=np.array(uv),
                        masses_kg=masses, panel_ids=np.array(panels,dtype=np.int32), initial_vertices=initial)
    lines = ['# Original procedural gray PE bag; flat reference mesh; m; Z up',
             'mtllib surface.mtl', 'usemtl gray_pe']
    lines += ['v '+' '.join(f'{v:.12g}' for v in row) for row in q]
    lines += ['vt '+' '.join(f'{v:.12g}' for v in row) for row in uv]
    lines += ['f '+' '.join(f'{i+1}/{i+1}' for i in row) for row in f]
    (ROOT/'surface.obj').write_text('\n'.join(lines)+'\n')
    (ROOT/'surface.mtl').write_text('newmtl gray_pe\nKd 1 1 1\nKs .15 .15 .15\nNs 15\nmap_Kd texture.png\n')
    return dict(vertices=len(q), triangles=len(f), area_m2=float(area.sum()),
                mass_kg=float(masses.sum()), initial_max_abs_edge_strain=strain,
                initial_placement='empty, approximately isometric shallow transverse ripples, 2 mm maximum layer separation',
                reference_state='two flat PE panels, three welded sides, open mouth and 50 mm sealing flap',
                physical_film_thickness_m=THICKNESS, density_kg_m3=DENSITY,
                source_photo_bulge_used_as_rest_shape=False, world_fixed_nodes=0, payloads=0)


def texture():
    rng = np.random.default_rng(20261022)
    noise = rng.normal(0, 1.2, (1024,1024,1))
    base = np.broadcast_to(np.array([155,163,173]), (1024,1024,3))
    im = Image.fromarray(np.clip(base+noise,0,255).astype('uint8'))
    d = ImageDraw.Draw(im)
    # Original fictional shipping label, with no photographed recipient data.
    d.rectangle((125,300,405,615), fill=(232,231,221), outline=(203,204,199), width=2)
    d.text((143,318), 'PE MAILER / TEST ASSET', fill=(49,49,48))
    d.text((143,350), 'EMPTY - 350 x 450 mm', fill=(61,61,60))
    d.line((142,382,387,382), fill=(88,88,86), width=2)
    for y in (409,433,457):
        d.line((145,y,376,y), fill=(129,129,125), width=2)
    for i in range(66):
        x = 145 + i*3
        if rng.random() > .27:
            d.rectangle((x,490,x+int(rng.integers(1,3)),559), fill=(45,45,44))
    d.text((145,579), '0000 0000 0000', fill=(64,64,62))
    # Adhesive release liner on the sealing flap in the second UV region.
    d.rectangle((532,40,1003,70), fill=(224,224,216))
    d.line((532,86,1003,86), fill=(112,120,130), width=2)
    im.save(ROOT/'texture.png')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    geometry = build_geometry()
    texture()
    write_json('material.json', dict(
        bending_rigidity_nm=2e-6, plastic_yield_angle_deg=35.0,
        plastic_hardening_ratio=.04, plasticity=True, memory_ratio=.25,
        recovery_tau_s=.8, bending_damping_nms=1e-9,
        timestep_s=.00005, frame_dt_s=.01, collision_radius_m=.00025,
        collision_patch_faces=28, friction=.35, velocity_damping_per_s=2.,
        self_contact=True, gravity=True, edge_solref_s=.0002,
        contact_timeconst_s=.0002, self_contact_front_panel_ids=[0], self_contact_back_panel_ids=[1,2],
        edge_solimp=[.99,.999,.0001], iterations=150, tolerance=1e-8, solver='Newton',
        parameter_status='authored_uncalibrated_thin_PE_like_test_asset',
        physical_film_thickness_m=THICKNESS, assumed_density_kg_m3=DENSITY,
        areal_density_gsm=THICKNESS*DENSITY*1000,
        membrane='native low-stretch edge constraints, not fitted tensile elasticity',
        recovery_model='short-time SLS and permanent plastic hinge; authored qualitative approximation',
        collision_radius_basis='numerical contact thickness; distinct from 50 micrometre physical film',
        mechanics_basis='engineering values selected for light, wrinkling, sagging PE-like behavior, not calibrated'))
    write_json('asset.json', dict(
        id='gray_pe_mailer_350x450', category_id='pe_film_mailer', material_type='pe_film_mailer',
        classification_status='gray_PE_like_reference_class; exact PDF polymer grade unconfirmed',
        size_id='nominal_350x450_plus_50_flap', source_variant_index=None,
        form_state='empty_open_flap', fill_state='empty_no_payload', colorway='cool_gray',
        flat_size_m=[WIDTH,HEIGHT], flap_length_m=FLAP, mass_kg=geometry['mass_kg'],
        physics_mode='mujoco_flex_with_python_material_callback', deformable_dynamics=True, calibrated=False,
        entrypoints={'python_factory':'scene.py'}, mesh_files=['geometry.npz','surface.obj'], preview='preview.png',
        source_url=SOURCE_URL, geometry=geometry,
        dimension_basis='authored 350x450 representative size; online 350x475+50 mm analog only',
        mass_basis='flat triangle area times assumed LDPE density times reference 50 micrometre gauge; no label/glue added mass',
        validation_scope='asset VALIDATION.json; finite rollout is not physical calibration'))
    pdf = ROOT.parents[3]/'典型软包.pdf'
    write_json('PROVENANCE.json', dict(
        version=1, authored_date='2026-10-20', units='m_kg_s', up_axis='Z',
        reconstruction='Original procedural two-panel empty shell and original fictional-label texture; no model training',
        status='first_version_behavioral_asset_not_calibrated',
        sources=[
            dict(id='user_photos', file='../../../../典型软包.pdf', pages=[2], positions=['upper_left','upper_right'],
                 sha256=hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf.is_file() else None,
                 usage='cool-gray film appearance, mailing-bag structure and qualitative wrinkles; no exact material/size claim'),
            dict(id='representative_gray_pe_mailer', url=SOURCE_URL,
                 facts={'nominal_size_mm':[350,475], 'flap_mm':50, 'film_thickness_micrometre':50, 'material':'LDPE', 'color':'gray'},
                 usage='representative PE structure, gauge and closure; asset height deliberately changed to 450 mm; no copied image/texture'),
        ],
        assumptions=['Exact polymer formulation, physical dimensions and payloads in PDF are unknown',
                     'Density 920 kg/m3 and all bending/plastic/relaxation values are test assumptions',
                     'Planar welded two-panel stress-free reference is separate from rippled initial placement',
                     'No inflation pressure, internal support, pins, payload, adhesion, tearing or leak model',
                     'Low-stretch edges approximate film response and do not reproduce large-strain PE stretching',
                     'Shipping label is appearance only and contributes no additional stiffness or mass',
                     'Third-party product page is reference only; procedural files do not incorporate product imagery']))
    print(json.dumps(geometry, indent=2))


if __name__ == '__main__':
    main()
