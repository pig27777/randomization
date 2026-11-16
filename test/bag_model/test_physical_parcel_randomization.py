"""Behavior and physical consistency checks for randomized payloads."""
import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import replace
import mujoco
import physical_parcel_yam as adapter
import numpy as np
from bag_sim.physical_parcel import Config
from bag_sim.parcel_contents import mass_properties
from physical_parcel_checks import film_payload_geometry
from physical_parcel_randomization import sample_config, sample_surface, ARCHETYPES
from physical_parcel_yam import parcel_world


class RandomizationTests(unittest.TestCase):
    def test_density_gradient_inertia_matches_volume_integral(self):
        # Independent quadrature over a cuboid checks mass, COM and off-diagonal inertia.
        size=np.array([.12,.08,.03]);mass=.25;gradient=np.array([.15,-.1,.05])
        axes=[np.linspace(-s/2,s/2,60,endpoint=False)+s/120 for s in size]
        points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        weights=1+(points/(size/2))@gradient
        weights*=mass/weights.sum()
        center=np.average(points,weights=weights,axis=0)
        q=points-center
        inertia=np.eye(3)*np.sum(weights*np.sum(q*q,axis=1))-np.einsum('n,ni,nj->ij',weights,q,q)
        exact=mass_properties('box',size,mass,gradient)
        np.testing.assert_allclose(center,exact['com'],rtol=.0003)
        np.testing.assert_allclose(inertia,exact['inertia'],rtol=.0007,atol=1e-9)

    def test_each_shape_and_mode_initializes_without_geometric_penetration(self):
        rng=np.random.default_rng(4116)
        for shape,mode in ARCHETYPES:
            with self.subTest(shape=shape,mode=mode):
                cfg=sample_config(np.random.default_rng(4116),'pe',shape,mode)
                p,_=parcel_world(cfg)
                self.assertEqual(p.m.nflexvert,142)
                self.assertEqual(p.m.nflexelem,280)
                self.assertAlmostEqual(p.m.body_mass[p.payload_bodies].sum(),cfg.payload_mass)
                geom=film_payload_geometry(p.m,p.d,p.envelope['faces'])
                self.assertLessEqual(geom['geometric_depth_upper_m'],.001)
                self.assertEqual(geom['outside_corners'],0)
                for region in ['top','edge','corner','side']:
                    fields,actual=sample_surface(p,cfg,rng,region)
                    expected=np.array(actual['barycentric'])@(p.envelope['current'][p.envelope['faces'][actual['triangle']]]+[0,0,cfg.support_height])
                    np.testing.assert_allclose(fields['surface_point'],expected)
                    self.assertAlmostEqual(np.dot(fields['surface_normal'],fields['surface_closing']),0)

    def test_material_overrides_and_seed_are_applied(self):
        for profile in ['kraft','pe','bubble']:
            a=sample_config(np.random.default_rng(811),profile,'box','rigid')
            b=sample_config(np.random.default_rng(811),profile,'box','rigid')
            self.assertEqual(a,b)
            p,_=parcel_world(a)
            m=a.material
            self.assertAlmostEqual(p.cfg.membrane,m['young_pa']*m['thickness_m'])
            self.assertAlmostEqual(p.cfg.areal_mass,m['density_kg_m3']*m['thickness_m'])
            self.assertAlmostEqual(p.cfg.bending,m['young_pa']*m['thickness_m']**3/(12*(1-m['poisson']**2)))

    def test_invalid_shapes_and_density_are_rejected(self):
        for shape,size,g in [('sphere',[.1,.08,.1],[0,0,0]),('capsule',[.03,.04,.04],[0,0,0]),('box',[.1,.1,.1],[1,0,0])]:
            with self.assertRaises(ValueError):mass_properties(shape,size,.1,g)
        with self.assertRaises(ValueError):Config(mode='bendable',payload_shape='sphere',payload_size=(.03,.03,.03))
        with self.assertRaises(ValueError):Config(payload_shape='sphere',payload_size=(.03,.03,.03),payload_com=(.001,0,0))

    def test_wide_support_has_requested_geometry_and_signature(self):
        cfg=adapter.Config(geometry='geometry/kraft_edge_sealed_35.npz',dt=.0001,
                           support_height=.15,support_size=(.4,.3))
        parcel,xml=parcel_world(cfg)
        model=parcel.compile_model(xml);data=mujoco.MjData(model)
        adapter.initialize_state(parcel,model,data,cfg)
        geom=model.geom('table').id
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        corners=(signs*model.geom_size[geom])@data.geom_xmat[geom].reshape(3,3).T
        np.testing.assert_allclose(np.ptp(corners,axis=0),[.4,.3,.15],atol=1e-9)
        self.assertEqual(json.loads(adapter.state_signature(parcel,cfg))['support_size'],[.4,.3])
        fields,actual=sample_surface(parcel,cfg,np.random.default_rng(55),'top',points=data.flexvert_xpos)
        triangle=data.flexvert_xpos[parcel.envelope['faces'][actual['triangle']]]
        np.testing.assert_allclose(fields['surface_point'],np.asarray(actual['barycentric'])@triangle)

    def test_archived_prepared_box_defaults_load_but_changed_support_is_rejected(self):
        cfg=adapter.Config(geometry='geometry/kraft_edge_sealed_35.npz',dt=.0001)
        parcel,_=parcel_world(cfg)
        signature=json.loads(adapter.state_signature(parcel,cfg))
        for key in ['payload_shape','payload_euler','payload_offset','density_gradient','payload_friction']:
            del signature['model'][key]
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory)/'legacy.npz'
            np.savez_compressed(state,qpos=parcel.d.qpos,qvel=parcel.d.qvel,config_json=np.asarray(json.dumps(signature)))
            loaded=replace(cfg,initial_state=str(state))
            data=mujoco.MjData(parcel.m)
            adapter.initialize_state(parcel,parcel.m,data,loaded)
            np.testing.assert_allclose(data.qpos,parcel.d.qpos)
            with self.assertRaisesRegex(ValueError,'different parcel or support'):
                adapter.initialize_state(parcel,parcel.m,data,replace(loaded,support_size=(.4,.3)))

    def test_phase_snapshot_detects_approach_load_without_inventing_hold(self):
        from physical_parcel_yam_benchmark import refresh_saved_stages
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'trial').mkdir()
            (root/'summary.json').write_text(json.dumps(dict(initial_com=[0,0,.1])))
            trace=[dict(phase='approach',grip_N=0.,contact_magnitude_each_N=[0.,0.],support_N=1.,com=[0,0,.1]),
                   dict(phase='approach',grip_N=0.,contact_magnitude_each_N=[2.,0.],support_N=1.,com=[0,0,.1]),
                   dict(phase='close',grip_N=.1,contact_magnitude_each_N=[.1,.1],support_N=1.,com=[0,0,.1])]
            (root/'trial/trace.json').write_text(json.dumps(trace))
            result=refresh_saved_stages(root)
            self.assertEqual(result['stages']['approach']['index'],1)
            self.assertTrue(result['stages']['approach']['loaded'])
            self.assertFalse(result['stages']['approach']['gripping'])
            self.assertTrue(result['stages']['lift_hold']['missing'])


if __name__=='__main__':unittest.main()
