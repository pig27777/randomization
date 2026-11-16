"""Constitutive invariants and material-state tests for the six-face membrane."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import warp as wp
from contents.parcel import Batch
from contents.parcel.film import dissipative_forces,fracture,hinge_vertices,dihedral_gradients,hinge_rate
from film_checks import energy_checks


@wp.kernel
def angular_rate_error(points: wp.array2d(dtype=wp.vec3),velocities: wp.array2d(dtype=wp.vec3),
                       triangles: wp.array(dtype=wp.vec3i),adjacent: wp.array(dtype=wp.vec2i),
                       edges: wp.array(dtype=wp.vec2i),errors: wp.array2d(dtype=float)):
    e,h=wp.tid()
    ids=hinge_vertices(triangles,adjacent,edges,h)
    a=ids[0];b=ids[1];c=ids[2];d=ids[3]
    qa,qb,qc,qd=dihedral_gradients(points[e,a],points[e,b],points[e,c],points[e,d])
    rate=(wp.dot(qa,velocities[e,a])+wp.dot(qb,velocities[e,b])
          +wp.dot(qc,velocities[e,c])+wp.dot(qd,velocities[e,d]))
    errors[e,h]=rate-hinge_rate(points,velocities,triangles,adjacent,edges,e,h)


class FilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.b=Batch(4,prototype=3)
    def setUp(self):self.b.reset()

    def test_topology_mass_and_reset(self):
        b=self.b;s=b.surface;t=s.topo
        self.assertEqual((len(t.vertices),len(t.triangles),len(t.edges)),(56,108,162))
        self.assertEqual(len(t.vertices)-len(t.edges)+len(t.triangles),2)
        self.assertEqual(int(t.seam.sum()),36)
        self.assertTrue(all(int((t.face_id==i).sum())==18 for i in range(6)))
        p=s.parameters.numpy();x=s.rest.numpy();tri=t.triangles
        areas=np.linalg.norm(np.cross(x[:,tri[:,1]]-x[:,tri[:,0]],x[:,tri[:,2]]-x[:,tri[:,0]]),axis=2)/2
        np.testing.assert_allclose(s.masses.numpy().sum(1),areas.sum(1)*p['thickness']*p['density'],rtol=1e-6)
        self.assertTrue((s.masses.numpy()>0).all())
        arms=b.model.dof_armature.numpy()
        for adr in s.dofs.numpy():np.testing.assert_array_equal(arms[:,adr:adr+3],0)
        s.damage.fill_(1);b.observer.gathered.fill_(1);b.reset()
        np.testing.assert_array_equal(s.parameters.numpy(),p)
        np.testing.assert_array_equal(s.damage.numpy(),0)
        np.testing.assert_array_equal(b.observer.gathered.numpy(),0)
        b.reset(1);self.assertFalse(np.array_equal(s.parameters.numpy(),p))

    def test_energy_objectivity_and_conservation(self):
        result=energy_checks(self.b)
        for key,value in result.items():
            if key.endswith('_pass'):self.assertTrue(value,result)

    def test_damping_passivity_and_rigid_motion(self):
        s=self.b.surface;x=s.rest.numpy();rng=np.random.default_rng(9)
        s.points.assign(x);s.velocities.assign(rng.normal(0,.01,x.shape).astype('f'))
        args=[s.points,s.velocities,s.edges,s.seam,s.parameters,s.masses,s.damage,s.damping,s.power]
        s.damping.zero_();s.power.zero_();wp.launch(dissipative_forces,s.lengths.shape,args)
        f=s.damping.numpy();v=s.velocities.numpy()
        self.assertTrue((s.power.numpy()<=0).all())
        np.testing.assert_allclose((f*v).sum((1,2)),s.power.numpy(),atol=1e-7)
        np.testing.assert_allclose(f.sum(1),0,atol=1e-6)
        np.testing.assert_allclose(np.cross(x,f).sum(1),0,atol=1e-6)
        rigid=np.cross(np.array([.1,.2,.3]),x)+np.array([.5,-.2,.1])
        s.velocities.assign(rigid.astype('f'));s.damping.zero_();s.power.zero_()
        wp.launch(dissipative_forces,s.lengths.shape,args)
        self.assertLess(np.abs(s.power.numpy()).max(),1e-10)

    def test_bending_damping_is_objective_and_dissipative(self):
        s=self.b.surface;x=s.rest.numpy();rng=np.random.default_rng(12)
        x+=rng.normal(0,.001,x.shape).astype('f')
        s.points.assign(x);s.elastic_forces();elastic=s.points.grad.numpy().copy()
        s.velocities.assign(rng.normal(0,.01,x.shape).astype('f'))
        errors=wp.empty(s.lengths.shape,dtype=float)
        wp.launch(angular_rate_error,s.lengths.shape,[s.points,s.velocities,s.triangles,s.adjacent,s.edges,errors])
        self.assertLess(np.abs(errors.numpy()).max(),1.e-4)
        s.damping.zero_();s.power.zero_();s.bending_damping()
        f=s.damping.numpy();v=s.velocities.numpy()
        self.assertTrue((s.power.numpy()<0).all())
        np.testing.assert_allclose((f*v).sum((1,2)),s.power.numpy(),atol=1e-7,rtol=1e-5)
        np.testing.assert_allclose(f.sum(1),0,atol=1e-6)
        np.testing.assert_allclose(np.cross(x,f).sum(1),0,atol=1e-6)
        np.testing.assert_array_equal(s.points.grad.numpy(),elastic)
        rigid=np.cross(np.array([.1,.2,.3]),x)+np.array([.5,-.2,.1])
        s.velocities.assign(rigid.astype('f'));s.damping.zero_();s.power.zero_();s.bending_damping()
        self.assertLess(np.abs(s.power.numpy()).max(),1e-10)

    def test_cross_face_force_and_environment_isolation(self):
        s=self.b.surface;t=s.topo;x=s.rest.numpy()
        node=np.flatnonzero((t.vertices[:,2]==.5)&(np.abs(t.vertices[:,0])<.5)&(np.abs(t.vertices[:,1])<.5))[0]
        x[0,node,2]+=.002;s.points.assign(x);s.elastic_forces();f=-s.points.grad.numpy()
        np.testing.assert_allclose(f[1:],0,atol=1e-7)
        side_nodes=np.flatnonzero(np.abs(t.vertices[:,0])==.5)
        self.assertGreater(np.linalg.norm(f[0,side_nodes]),1e-5)

    def test_six_face_material_coordinates_and_budget(self):
        b=self.b;s=b.surface;o=b.observer
        n=b.nworld;capacity=o.capacity
        world=np.zeros(capacity,dtype=np.int32);positions=np.zeros((capacity,3),dtype=np.float32)
        sides=np.full(capacity,-1,dtype=np.int32);elements=np.zeros(capacity,dtype=np.int32)
        forces=np.zeros((capacity,3),dtype=np.float32)
        geom=b.data.geom_xpos.numpy();rest=s.rest.numpy();pads=o.pads.numpy()
        initial=s.available.numpy().copy()
        for face in range(6):
            o.reset()
            t=int(np.flatnonzero(s.topo.face_id==face)[len(np.flatnonzero(s.topo.face_id==face))//2])
            ids=s.topo.triangles[t]
            for e in range(n):
                a,bb,c=rest[e,ids];normal=np.cross(bb-a,c-a);normal/=np.linalg.norm(normal)
                position=(a+bb+c)/3
                geom[e,pads[0]]=position-normal*.005;geom[e,pads[1]]=position+normal*.005
                for side in (0,1):
                    k=2*e+side;world[k]=e;sides[k]=side;elements[k]=t;positions[k]=position
                    forces[k]=normal*(1 if side else -1)
            # First gather by tangential motion while both fingers touch the film.
            folded_geom=geom.copy();folded_forces=forces.copy()
            for e in range(n):
                a,bb,c=rest[e,ids];tangent=bb-a;tangent/=np.linalg.norm(tangent)
                normal=np.cross(bb-a,c-a);normal/=np.linalg.norm(normal)
                position=(a+bb+c)/3
                geom[e,pads[0]]=position-tangent*.005;geom[e,pads[1]]=position+tangent*.005
                forces[2*e]=normal;forces[2*e+1]=normal
            o.forces.assign(forces);o.sides.assign(sides);o.elements.assign(elements)
            nacon=wp.array([2*n],dtype=int);world_wp=wp.array(world,dtype=int)
            points_wp=wp.array(positions,dtype=wp.vec3);geom_wp=wp.array(geom,dtype=wp.vec3)
            state=o.states.numpy();state['previous_gap']=.011;o.states.assign(state)
            o._update(nacon,world_wp,points_wp,geom_wp,s.rest,0,0,face,b.dt)
            self.assertGreater(o.gathered.numpy().sum(),0)
            # The resolved fold then presents opposed normals to the closing fingers.
            o.forces.assign(folded_forces);geom_wp.assign(folded_geom)
            o._update(nacon,world_wp,points_wp,geom_wp,s.rest,1,0,face,b.dt)
            state=o.states.numpy()
            self.assertTrue((state['face_id']==face).all())
            self.assertTrue((state['mode']==1).all())
            self.assertTrue((state['barycentric']>=0).all())
            np.testing.assert_allclose(state['barycentric'].sum(1),1,atol=1e-6)
            self.assertTrue((o.gathered.numpy()<=initial+1e-7).all())
            saved=o.gathered.numpy().copy()
            for requested in range(6):o._update(nacon,world_wp,points_wp,geom_wp,s.rest,1,0,requested,b.dt)
            np.testing.assert_array_equal(o.gathered.numpy(),saved)
            self.assertTrue((o.states.numpy()['face_id']==face).all())
            o._update(nacon,world_wp,points_wp,geom_wp,s.rest,0,1,-1,b.dt)
            self.assertTrue((o.states.numpy()['captured_length']==0).all())
            np.testing.assert_array_equal(o.gathered.numpy(),saved)
        np.testing.assert_array_equal(s.available.numpy(),initial)

    def test_fracture_energy_accounting(self):
        s=self.b.surface;x=s.rest.numpy();x[0,0]+=[.003,.001,.002]
        p=s.parameters.numpy();p['breaking_force']=.001;s.parameters.assign(p)
        s.points.assign(x);s.elastic_forces();before=s.energy.numpy().copy()
        wp.launch(fracture,s.lengths.shape,[s.points,s.edges,s.seam,s.parameters,s.lengths,s.angles,
            s.triangles,s.adjacent,s.damage,s.fracture_loss])
        self.assertGreater(s.damage.numpy().sum(),0)
        s.elastic_forces()
        np.testing.assert_allclose(before,s.energy.numpy()+s.fracture_loss.numpy(),atol=1e-7,rtol=1e-4)


if __name__=='__main__':unittest.main()
