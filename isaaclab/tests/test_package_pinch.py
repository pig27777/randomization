"""Point reaction, free rotation, load filtering, and original-mesh topology."""
from pathlib import Path
import sys
import unittest

import numpy as np
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pinch_effect import closest_material_point, lowpass_force, point_connect, spring_force
from package_assets import coarsen, original_mesh, topology, wrapped, seam_pairs


class PinchTests(unittest.TestCase):
    def test_filter_releases_on_crossing_not_duration(self):
        dt=.0004
        # A single 20 N step stays below a 2 N limit after filtering.
        f=lowpass_force(0.,20.,dt)
        self.assertLess(f,2.)
        for _ in range(20):
            before=f;f=lowpass_force(f,3.,dt)
            if f>2.:
                self.assertLessEqual(before,2.)
                break
        else:self.fail('constant overload did not cross the capacity')
        self.assertLess(f,4.,'raising current grip to 4 N removes the overload')

    def test_spring_gathering_and_no_force_before_eligibility(self):
        self.assertEqual(spring_force(.04,.02,False),0.)
        self.assertEqual(spring_force(.04,.03),0.)
        self.assertAlmostEqual(spring_force(.04,.02975),10.)

    def test_material_point_and_nearest_edge(self):
        v=np.array([[0.,0,0],[1.,0,0],[0.,1,0]])
        f=np.array([[0,1,2]])
        _,w,p=closest_material_point(v,f,[.25,.25,.01])
        np.testing.assert_allclose(w,[.5,.25,.25]);np.testing.assert_allclose(p,[.25,.25,0.])
        _,w,p=closest_material_point(v,f,[-1.,.5,0.])
        np.testing.assert_allclose(p,[0,.5,0]);self.assertTrue(np.all(w>=0))

    def project(self,vertices,target,force=100.):
        q=wp.array(vertices,dtype=wp.vec3,device='cpu');qd=wp.zeros(3,dtype=wp.vec3,device='cpu')
        reaction=wp.zeros(1,dtype=wp.vec3,device='cpu');filtered=wp.zeros(1,device='cpu')
        active=wp.ones(1,dtype=int,device='cpu');broke=wp.zeros(1,dtype=int,device='cpu')
        wp.launch(point_connect,dim=1,inputs=[q,qd,wp.ones(3,device='cpu'),
            wp.array([[0,1,2]],dtype=int,device='cpu'),wp.array([[1/3]*3],dtype=wp.vec3,device='cpu'),
            wp.array([target],dtype=wp.vec3,device='cpu'),active,wp.ones(1,device='cpu'),
            wp.full(1,force,device='cpu'),filtered,reaction,broke,.001],device='cpu')
        return q.numpy(),reaction.numpy()[0],bool(active.numpy()[0]),bool(broke.numpy()[0])

    def test_point_projection_preserves_rotation_and_load(self):
        v=np.array([[-.1,-.1,0.],[.1,-.1,0.],[0.,.2,0.]])
        rotated=v@np.array([[1,0,0],[0,0,-1],[0,1,0]])
        after,reaction,active,broke=self.project(rotated,[0,0,0])
        np.testing.assert_allclose(after,rotated,atol=1e-7)
        np.testing.assert_allclose(reaction,0,atol=1e-6)
        fallen=v-[0,0,9.81e-6]
        after,reaction,active,broke=self.project(fallen,[0,0,0])
        np.testing.assert_allclose(after.mean(0),0.,atol=1e-7)
        np.testing.assert_allclose(reaction,[0,0,-3*9.81],rtol=1e-5)
        self.assertTrue(active);self.assertFalse(broke)
        _,_,active,broke=self.project(fallen,[0,0,0],force=1.)
        self.assertFalse(active);self.assertTrue(broke)

    def test_coarse_mesh_preserves_panels_and_topology(self):
        fine=original_mesh();coarse=coarsen(fine)
        self.assertEqual(len(coarse['vertices']),150)
        self.assertEqual(set(coarse['panel_ids']),set(fine['panel_ids']))
        self.assertEqual(topology(coarse['faces'])['euler'],topology(fine['faces'])['euler'])
        self.assertEqual(topology(coarse['faces'])['nonmanifold_edges'],0)
        self.assertAlmostEqual(coarse['masses_kg'].sum(),fine['masses_kg'].sum())

    def test_gripper_moves_in_native_solver_and_generates_contact(self):
        from package_scene import PackageScene
        scene=PackageScene(cloth=False,device='cpu')
        np.testing.assert_allclose(scene.rigid_solver.mj_model.geom_solref[:,0],.040,rtol=1e-6)
        target=scene.targets.numpy();target[:,0]=.123
        scene.targets.assign(target)
        for _ in range(2000):scene.step()
        native=scene.rigid_solver.mj_data.xpos[scene.pad_mj_ids.numpy()]
        np.testing.assert_allclose(native,target,atol=1e-5)
        self.assertAlmostEqual(float(scene.table_force.numpy()[0]),scene.mass*9.81,delta=.05)
        # Start the contact fixture at its opening instead of sweeping through the box.
        scene=PackageScene(cloth=False,device='cpu')
        target=scene.targets.numpy()
        target[:,0]=0.;target[:,1]=[-.024,.024];target[:,2]=.021
        scene.targets.assign(target)
        coords=scene.s0.joint_q.numpy()
        for i,j in enumerate(scene.pad_coordinate_starts):coords[j:j+3]=target[i]-scene.pad_origins.numpy()[i]
        scene.s0.joint_q.assign(coords)
        for _ in range(2000):scene.step()
        self.assertTrue(np.isfinite(scene.pad_forces.numpy()).all())
        self.assertGreater(np.abs(scene.pad_forces.numpy()[:,1]).mean(),.05)

    def test_seam_endpoints_are_solved_in_different_color_groups(self):
        from types import SimpleNamespace
        from package_scene import color_seams
        groups=[]
        builder=SimpleNamespace(particle_count=4,particle_color_groups=[[0,3],[1],[2]],
            tri_indices=[[0,1,2]],edge_indices=[],spring_indices=[0,3],set_coloring=groups.extend)
        color_seams(builder)
        labels={int(v):c for c,group in enumerate(groups) for v in group}
        self.assertNotEqual(labels[0],labels[3])
        self.assertEqual(len({labels[i] for i in (0,1,2)}),3)

    def test_loading_closes_seam_without_spring_preload(self):
        mesh=original_mesh();_,q,_,report=wrapped(mesh,np.array([.08,.045,.02]),.35132402626147194)
        a,b=seam_pairs(mesh).T
        np.testing.assert_allclose(np.linalg.norm(q[a]-q[b],axis=1),.0009,atol=1e-10)
        self.assertLessEqual(report['maximum_edge_strain'],.05)

    def test_point_reaction_reaches_gripper_without_a_rotational_weld(self):
        from package_grasp import effect_reaction
        positions=np.array([[0,-.022,.02],[0,.022,.02]],np.float32)
        q=wp.array([wp.transform(p,wp.quat_identity()) for p in positions],dtype=wp.transform,device='cpu')
        force=wp.zeros(2,dtype=wp.spatial_vector,device='cpu');known=wp.zeros(2,dtype=wp.vec3,device='cpu')
        zero=wp.zeros(1,dtype=int,device='cpu')
        load=np.array([1.,2.,-3.],np.float32)
        wp.launch(effect_reaction,dim=1,inputs=[q,wp.array([0,1],dtype=int,device='cpu'),
            wp.zeros(1,device='cpu'),zero,zero,wp.ones(1,dtype=int,device='cpu'),
            wp.array([load],dtype=wp.vec3,device='cpu'),wp.zeros(1,dtype=wp.vec3,device='cpu'),
            force,known],device='cpu')
        wrench=force.numpy()
        np.testing.assert_allclose(wrench[:,:3].sum(0),load)
        np.testing.assert_allclose((np.cross(positions,wrench[:,:3])+wrench[:,3:]).sum(0),0.,atol=1e-7)

    def test_failure_is_latched_between_control_reads(self):
        from package_scene import content_failure
        q=wp.array([wp.transform([0,0,0],wp.quat_identity()),wp.transform([.06,0,0],wp.quat_identity())],dtype=wp.transform,device='cpu')
        velocity=wp.zeros(2,dtype=wp.spatial_vector,device='cpu')
        empty=wp.zeros(0,dtype=wp.vec3,device='cpu');out=wp.zeros(4,device='cpu')
        inputs=[q,velocity,empty,empty,wp.array([0],dtype=int,device='cpu'),
            wp.array([1],dtype=int,device='cpu'),wp.array([[.05,0,0]],dtype=wp.vec3,device='cpu'),
            wp.array([.3],dtype=float,device='cpu'),wp.array([[1e30,.5,.5]],dtype=wp.vec3,device='cpu'),out]
        wp.launch(content_failure,dim=2,inputs=inputs,device='cpu')
        self.assertAlmostEqual(float(out.numpy()[0]),.4,places=6)
        self.assertGreater(float(out.numpy()[1]),1.2)
        q.assign([wp.transform([0,0,0],wp.quat_identity()),wp.transform([.1,0,0],wp.quat_identity())])
        wp.launch(content_failure,dim=2,inputs=inputs,device='cpu')
        self.assertGreater(float(out.numpy()[1]),1.2,'a recovered pose must retain the intervening limit breach')

    def test_obb_uses_package_axes_after_rotation(self):
        from package_grasp import oriented_bbox
        angle=.63;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        points=np.array([[-5,-2],[-5,2],[5,-2],[5,2],[0,0]])@R.T+[1,2]
        center,size,heading=oriented_bbox(points)
        np.testing.assert_allclose(center,[1,2],atol=1e-12)
        np.testing.assert_allclose(size,[10,4],atol=1e-12)
        self.assertAlmostEqual(abs(np.cos(heading-angle)),1.)

    def test_cached_rest_reproduces_generalized_state(self):
        import tempfile
        from package_scene import PackageScene
        from package_grasp import Evaluation
        results=Path(__file__).resolve().parents[1]/'results'
        with tempfile.TemporaryDirectory(dir=results,prefix='cache-check-') as directory:
            cache=Path(directory)/'rest.npz'
            first=Evaluation(PackageScene(cloth=False,device='cpu'),mode='side',cache=cache,capture=False)
            second=Evaluation(PackageScene(cloth=False,device='cpu'),mode='side',cache=cache,capture=False)
            self.assertTrue(second.cache_used)
            for name,expected in first.snapshot().items():np.testing.assert_array_equal(second.snapshot()[name],expected)

    def test_cloth_cached_frame_preserves_distances(self):
        import tempfile
        from package_scene import PackageScene
        from package_grasp import Evaluation
        scene=PackageScene(category='hard',scale=.35132402626147194,mass_scale=10,device='cpu')
        source=Evaluation.__new__(Evaluation);source.scene=scene
        original=scene.s0.particle_q.numpy().copy()
        results=Path(__file__).resolve().parents[1]/'results'
        with tempfile.TemporaryDirectory(dir=results,prefix='cloth-cache-check-') as directory:
            cache=Path(directory)/'rest.npz';source.store(cache)
            restored=Evaluation(scene,mode='top',cache=cache,capture=False)
            self.assertTrue(restored.cache_used)
            actual=scene.s0.particle_q.numpy()
            self.assertEqual(actual.dtype,original.dtype)
            np.testing.assert_allclose(np.linalg.norm(actual-actual.mean(0),axis=1),
                np.linalg.norm(original-original.mean(0),axis=1),atol=1e-7)


if __name__=='__main__':unittest.main()
