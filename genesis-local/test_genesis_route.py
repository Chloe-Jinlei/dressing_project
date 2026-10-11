"""Control checks and opt-in CPU contact integration; no network/model calls."""
import io
import json
import os
from pathlib import Path
import tomllib
import unittest
from unittest.mock import Mock

import numpy as np
from genesis_route import GenesisController


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,dtype=float)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value.copy()


class Finger:
    def __init__(self,idx,pos,quat):self.idx=idx;self.pos=pos;self.quat=quat
    def get_pos(self,**kwargs):return Tensor(self.pos)
    def get_quat(self,**kwargs):return Tensor(self.quat)


def controller_fixture():
    controller=GenesisController.__new__(GenesisController)
    controller.steps=0;controller.dt=.005;controller.attached={};controller.used=set()
    controller.contact_log=io.StringIO()
    controller.controls=np.r_[np.zeros(7),[0.,0.],np.zeros(7),[.04,.04]]
    controller.closing={"left":True,"right":False}
    controller.robots={side:Mock() for side in ("left","right")}
    for robot in controller.robots.values():
        robot.get_qpos.return_value=Tensor(np.r_[np.zeros(7),[.01,.01]])
    controller.fingers={
        "left":[Finger(10,[0,-.018,0],[1,0,0,0]),Finger(11,[0,.018,0],[0,0,0,1])],
        "right":[Finger(20,[1,-.018,0],[1,0,0,0]),Finger(21,[1,.018,0],[0,0,0,1])],
    }
    x=np.array([-.004,.004,0.,.006])
    points=np.concatenate([np.column_stack([x,np.full(4,-.0085),np.full(4,.0445)]),
                           np.column_stack([x,np.full(4,.0085),np.full(4,.0445)])])
    entity=Mock();entity.get_particles_pos.return_value=Tensor(points)
    controller.objects={"garment":entity};controller.surface_indices={"garment":np.arange(8)}
    return controller,entity,points


class ContactGateTests(unittest.TestCase):
    def test_remote_or_single_pad_contact_does_not_attach(self):
        controller,entity,points=controller_fixture()
        entity.get_particles_pos.return_value=Tensor(points+[0,.3,0])
        controller.grasp_contacts();entity.fix_particles_to_link.assert_not_called()
        points[4:]+=np.array([0,.3,0])
        entity.get_particles_pos.return_value=Tensor(points)
        controller.grasp_contacts();entity.fix_particles_to_link.assert_not_called()

    def test_open_command_does_not_attach_even_with_pad_contact(self):
        controller,entity,_=controller_fixture()
        values=controller.controls.copy();values[7:9]=.04
        controller.set_control(values);controller.grasp_contacts()
        entity.fix_particles_to_link.assert_not_called()

    def test_both_pad_contacts_attach_disjoint_local_nodes_and_open_releases(self):
        controller,entity,_=controller_fixture()
        controller.grasp_contacts()
        calls=entity.fix_particles_to_link.call_args_list
        self.assertEqual(len(calls),2)
        groups=[set(call.kwargs["particles_idx_local"]) for call in calls]
        self.assertTrue(all(0<len(group)<=4 for group in groups))
        self.assertFalse(groups[0]&groups[1])
        self.assertEqual(len(controller.used),8)
        # The contact aperture is held until an opening command is received.
        np.testing.assert_allclose(controller.robots["left"].control_dofs_position.call_args.args[0][7:],[.01,.01])
        values=controller.controls.copy();values[7:9]=.04
        controller.set_control(values)
        self.assertFalse(controller.attached);self.assertFalse(controller.used)
        self.assertEqual(entity.release_particle.call_count,2)
        self.assertIn('"event": "released"',controller.contact_log.getvalue())

    def test_ik_branch_jump_is_rejected_without_sending_controls(self):
        controller,entity,_=controller_fixture()
        controller.config=tomllib.loads((Path(__file__).resolve().parent.parent/"bunnylab"/"config.toml").read_text())
        controller.hands={side:object() for side in controller.robots}
        controller.observe=lambda:{"robot":{
            "left_position_m":[0,0,1],"right_position_m":[.1,0,1],
            "left_quaternion_wxyz":[1,0,0,0],"right_quaternion_wxyz":[1,0,0,0]}}
        q=np.r_[np.zeros(7),[.01,.01]];q[0]=.21
        controller.robots["left"].inverse_kinematics.return_value=(Tensor(q),Tensor(np.zeros(6)))
        action={"left_delta_m":[.01,0,0],"right_delta_m":[0,0,0],
            "left_rotation_rad":[0,0,0],"right_rotation_rad":[0,0,0],"left_gripper":1,"right_gripper":1}
        with self.assertRaisesRegex(ValueError,"joint displacement"):
            controller.plan(action)
        controller.robots["left"].control_dofs_position.assert_not_called()


@unittest.skipUnless(os.environ.get("GENESIS_CONTACT_TEST_RUN_DIR"), "opt-in actual CPU contact integration")
class RigidPBDFrictionTests(unittest.TestCase):
    def test_supported_particles_stop_slide_and_remain_free_in_flight(self):
        import genesis as gs
        import trimesh
        from genesis_mesh_adapter import existing_pbd_mesh, install_rigid_pbd_friction

        run = Path(os.environ["GENESIS_CONTACT_TEST_RUN_DIR"])
        run.mkdir(parents=True, exist_ok=True)
        gs.init(backend=gs.cpu, seed=0, precision="32", logging_level="warning")
        try:
            scene = gs.Scene(show_viewer=False,
                sim_options=gs.options.SimOptions(dt=.001, substeps=1),
                pbd_options=gs.options.PBDOptions(particle_size=.004,
                    lower_bound=(-1, -1, -1), upper_bound=(1, 1, 1)))
            contact = install_rigid_pbd_friction(scene)
            scene.add_entity(morph=gs.morphs.Plane(),
                material=gs.materials.Rigid(needs_coup=True, coup_friction=.7))
            cases = {"frictionless": (0., .1, .002), "static": (.6, .001, .002),
                     "sliding": (.6, .4, .002), "airborne": (.6, .1, .2)}
            faces = np.array([[a, b, c] for row in range(3) for col in range(3)
                for a, b, c in ((row * 4 + col, row * 4 + col + 1, (row + 1) * 4 + col + 1),
                                (row * 4 + col, (row + 1) * 4 + col + 1, (row + 1) * 4 + col))], dtype=np.int32)
            objects = {}
            for index, (name, (friction, speed, height)) in enumerate(cases.items()):
                points = np.array([[col * .01, row * .01, height] for row in range(4) for col in range(4)])
                points[:, 0] += -.3 + index * .15
                mesh = run / f"{name}.obj"
                trimesh.Trimesh(points, faces, process=False).export(mesh)
                with existing_pbd_mesh("garment", points, faces, .001):
                    objects[name] = scene.add_entity(
                        morph=gs.morphs.Mesh(file=str(mesh), file_meshes_are_zup=True,
                                             decimate=False, convexify=False),
                        material=gs.materials.PBD.Cloth(static_friction=friction,
                            kinetic_friction=friction, air_resistance=0))
            scene.build()
            initial = {}
            for name, entity in objects.items():
                initial[name] = entity.get_particles_pos().cpu().numpy().copy()
                entity.set_particles_vel(np.tile([cases[name][1], 0., 0.], (entity.n_particles, 1)))
            checkpoints = {}
            for step in range(100):
                scene.step()
                if step + 1 in (1, 20, 100):
                    checkpoints[str(step + 1)] = {name: {
                        "mean_velocity": entity.get_particles_vel().cpu().numpy().mean(0).tolist(),
                        "mean_displacement": (entity.get_particles_pos().cpu().numpy() - initial[name]).mean(0).tolist(),
                    } for name, entity in objects.items()}
            (run / "report.json").write_text(json.dumps({"backend": "cpu", "real_model_call": False,
                "contact": contact, "checkpoints": checkpoints}, indent=2))
            early, late = checkpoints["20"], checkpoints["100"]
            self.assertAlmostEqual(late["frictionless"]["mean_velocity"][0], .1, delta=.002)
            self.assertLess(abs(late["static"]["mean_displacement"][0]), 2e-6)
            self.assertGreater(early["sliding"]["mean_velocity"][0], .25)
            self.assertLess(early["sliding"]["mean_velocity"][0], .33)
            self.assertLess(abs(late["sliding"]["mean_velocity"][0]), .002)
            self.assertGreater(late["sliding"]["mean_displacement"][0], .01)
            self.assertAlmostEqual(late["airborne"]["mean_velocity"][0], .1, delta=.002)
            self.assertLess(late["airborne"]["mean_displacement"][2], -.04)
        finally:
            gs.destroy()


class CottonMaterialTests(unittest.TestCase):
    def test_si_mass_and_geometric_scaling(self):
        from genesis_mesh_adapter import cotton_discretization
        settings = tomllib.loads((Path(__file__).resolve().parent.parent/"bunnylab/config.toml").read_text())["scene"]["genesis"]["garment_material"]
        p = np.array([[0.,0,0],[.1,0,0],[0,.1,0],[.1,.1,0]])
        f = np.array([[0,1,2],[1,3,2]])
        a = cotton_discretization(p,f,settings)
        b = cotton_discretization(p*2,f,settings)
        self.assertAlmostEqual(a["mass"].sum(), .01*.211)
        np.testing.assert_allclose(b["mass"],a["mass"]*4)
        np.testing.assert_allclose(b["stretch_compliance"],a["stretch_compliance"])
        np.testing.assert_allclose(b["bending_compliance"],a["bending_compliance"])
        with self.assertRaises(ValueError):
            cotton_discretization(p,f,dict(settings,bending_rigidity_nm=-1))


@unittest.skipUnless(os.environ.get("GENESIS_MATERIAL_TEST_RUN_DIR"), "opt-in actual material integration")
class DeformableMaterialIntegrationTests(unittest.TestCase):
    def test_volume_step_matches_energy_gradient(self):
        import genesis as gs
        from dressing_scene import write_obj, tetra_boundary
        from genesis_mesh_adapter import existing_pbd_mesh, install_cotton_solver
        run=Path(os.environ["GENESIS_MATERIAL_TEST_RUN_DIR"]+"-volume")
        run.mkdir(parents=True,exist_ok=False)
        points=np.array([[0.,0,.5],[.02,0,.5],[0,.02,.5],[0,0,.52]])
        tet=np.array([[0,1,2,3]])
        _,faces=tetra_boundary(points,tet)
        write_obj(run/"tetra.obj",points,faces)
        gs.init(backend=gs.cuda,precision="32",logging_level="warning")
        try:
            scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.01,gravity=(0,0,0)),
                vis_options=gs.options.VisOptions(n_support_neighbors=4),
                pbd_options=gs.options.PBDOptions(particle_size=.00001,max_stretch_solver_iterations=1,max_volume_solver_iterations=1))
            install_cotton_solver(scene)
            with existing_pbd_mesh("sloth",points,tet,.001):
                body=scene.add_entity(morph=gs.morphs.Mesh(file=str(run/"tetra.obj"),file_meshes_are_zup=True),
                    material=gs.materials.PBD.Elastic(stretch_compliance=1e20,volume_compliance=1e-12,volume_relaxation=.05))
            scene.build()
            body.fix_particles([0,1,2])
            deformed=body.get_particles_pos().detach().cpu().numpy().reshape(-1,3).copy()
            deformed[3,2]+=.005
            body.set_particles_pos(deformed)
            volume=lambda x: np.linalg.det(x[1:]-x[0])/6
            gradient=np.zeros(3)
            for axis in range(3):
                plus=deformed.astype(float);minus=deformed.astype(float)
                plus[3,axis]+=1e-6;minus[3,axis]-=1e-6
                gradient[axis]=(volume(plus)-volume(minus))/2e-6
            rest_volume=body.solver.elems_info.vol_rest.to_numpy()[0]
            inverse_mass=1/body.solver.particles_info.mass.to_numpy()[3]
            expected=-inverse_mass*(volume(deformed.astype(float))-rest_volume)*gradient/(inverse_mass*(gradient@gradient)+1e-8)*.05
            scene.step()
            actual=body.get_particles_pos().detach().cpu().numpy().reshape(-1,3)[3]-deformed[3]
            np.testing.assert_allclose(actual,expected,rtol=.002,atol=5e-8)
            self.assertGreater(np.linalg.norm(actual),1e-4)
            (run/"report.json").write_text(json.dumps({"expected_displacement_m":expected.tolist(),
                "actual_displacement_m":actual.tolist()},indent=2))
        finally:
            gs.destroy()

    def test_bending_step_matches_energy_gradient(self):
        import genesis as gs
        from dressing_scene import write_obj
        from genesis_mesh_adapter import existing_pbd_mesh, install_cotton_solver, apply_cotton_parameters
        run=Path(os.environ["GENESIS_MATERIAL_TEST_RUN_DIR"]+"-hinge")
        run.mkdir(parents=True,exist_ok=False)
        settings=tomllib.loads((Path(__file__).resolve().parent.parent/"bunnylab/config.toml").read_text())["scene"]["genesis"]["garment_material"]
        points=np.array([[0.,0,.5],[.02,0,.5],[0,.02,.5],[0,-.015,.515]])
        faces=np.array([[0,1,2],[1,0,3]])
        write_obj(run/"hinge.obj",points,faces)
        gs.init(backend=gs.cuda,precision="32",logging_level="warning")
        try:
            scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.01,gravity=(0,0,0)),
                vis_options=gs.options.VisOptions(n_support_neighbors=4),
                pbd_options=gs.options.PBDOptions(particle_size=.00001,max_stretch_solver_iterations=1,max_bending_solver_iterations=1))
            install_cotton_solver(scene)
            with existing_pbd_mesh("garment",points,faces,.001):
                cloth=scene.add_entity(morph=gs.morphs.Mesh(file=str(run/"hinge.obj"),file_meshes_are_zup=True),
                    material=gs.materials.PBD.Cloth(bending_relaxation=1.,air_resistance=0))
            scene.build()
            parameters=apply_cotton_parameters(cloth,settings)
            cloth.solver.edges_info.stretch_compliance.fill(1e20)
            cloth.fix_particles([0,1,2])
            initial=cloth.get_particles_pos().detach().cpu().numpy().reshape(-1,3).astype(float)
            hinge=cloth._inner_edges[0]
            def constraint(p):
                a,b,c=p[hinge[1:]]-p[hinge[0]]
                n=np.cross(a,b);m=np.cross(a,c)
                return np.arccos(np.clip(n@m/np.linalg.norm(n)/np.linalg.norm(m),-1,1))-np.pi
            grad=np.zeros(3)
            for axis in range(3):
                plus=initial.copy();minus=initial.copy();plus[3,axis]+=1e-6;minus[3,axis]-=1e-6
                grad[axis]=(constraint(plus)-constraint(minus))/2e-6
            inverse_mass=1/cloth.solver.particles_info.mass.to_numpy()[3]
            alpha=cloth.solver.inner_edges_info.bending_compliance.to_numpy()[0]/.01**2
            expected=-inverse_mass*constraint(initial)*grad/(inverse_mass*(grad@grad)+alpha)
            scene.step()
            actual=cloth.get_particles_pos().detach().cpu().numpy().reshape(-1,3)[3]-initial[3]
            np.testing.assert_allclose(actual,expected,rtol=.02,atol=5e-8)
            self.assertGreater(np.linalg.norm(actual),1e-6)
            (run/"report.json").write_text(json.dumps({"expected_displacement_m":expected.tolist(),
                "actual_displacement_m":actual.tolist(),"relative_error":float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                "cotton_material":parameters},indent=2))
        finally:
            gs.destroy()

    def test_original_garment_and_sloth_deform_under_gravity(self):
        import genesis as gs
        from PIL import Image
        from dressing_scene import prepare_assets
        from genesis_mesh_adapter import existing_pbd_mesh, install_cotton_solver, apply_cotton_parameters
        run = Path(os.environ["GENESIS_MATERIAL_TEST_RUN_DIR"])
        run.mkdir(parents=True, exist_ok=False)
        config, paths, prepared, report = prepare_assets(Path(__file__).resolve().parent.parent/"bunnylab", run)
        material = config["scene"]["genesis"]["garment_material"]
        gs.init(backend=gs.cuda, seed=0, precision="32", logging_level="warning")
        try:
            scene = gs.Scene(show_viewer=False,
                sim_options=gs.options.SimOptions(dt=.01,substeps=10),
                pbd_options=gs.options.PBDOptions(particle_size=.004,
                    max_stretch_solver_iterations=8,max_bending_solver_iterations=4,max_volume_solver_iterations=32,
                    lower_bound=(-1,-1,0),upper_bound=(1,1,2)))
            install_cotton_solver(scene)
            with np.load(prepared/"physics_meshes.npz") as data:
                rest={name:data[name+"_points"].copy() for name in ("sloth","garment")}
                connectivity={"sloth":data["sloth_tetra"].copy(),"garment":data["garment_faces"].copy()}
            entities={}
            for name in rest:
                mass=config["scene"][name]["mass"]
                mat=(gs.materials.PBD.Elastic(**config["scene"]["genesis"]["sloth_material"])
                     if name=="sloth" else gs.materials.PBD.Cloth(rho=material["areal_density_kg_m2"],
                        stretch_relaxation=material["stretch_relaxation"],bending_relaxation=material["bending_relaxation"]))
                surface=(gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(paths["sloth_texture"])),double_sided=True)
                         if name=="sloth" else gs.surfaces.Default(color=(.12,.48,.26,1),double_sided=True))
                with existing_pbd_mesh(name,rest[name],connectivity[name],mass):
                    entities[name]=scene.add_entity(material=mat,
                        morph=gs.morphs.Mesh(file=str(prepared/f"{name}.obj"),file_meshes_are_zup=True,decimate=False,convexify=False),
                        surface=surface)
            camera=scene.add_camera(res=(960,640),GUI=False,pos=(.65,-1.05,1.15),lookat=(0,-.08,.65),fov=43)
            scene.build()
            report["cotton_material"]=apply_cotton_parameters(entities["garment"],material)
            # Lab fixture only: small fixed patch, all remaining original vertices
            # evolve under gravity. No prescribed deformation or per-frame resets.
            pins={}
            for name,entity in entities.items():
                p=rest[name]
                ids=(np.flatnonzero(p[:,0]<p[:,0].min()+.015) if name=="garment"
                     else np.argsort(np.linalg.norm(p-p[p[:,2].argmax()],axis=1))[:4])
                pins[name]=ids.tolist()
                entity.fix_particles(ids)
            camera.start_recording(save_to_filename=str(run/"deformation.mp4"),fps=20)
            samples=[]
            for step in range(301):
                if step:
                    scene.step()
                if step%5==0:
                    camera.render()
                if step in (0,50,150,300):
                    rgb,*_=camera.render()
                    Image.fromarray(rgb).save(run/f"deformation-{step:03d}.png")
                    sample={"time_s":step*.01}
                    for name,entity in entities.items():
                        p=entity.get_particles_pos().detach().cpu().numpy().reshape(-1,3)
                        self.assertTrue(np.isfinite(p).all())
                        x=rest[name]-rest[name].mean(0);y=p-p.mean(0)
                        u,_,vt=np.linalg.svd(x.T@y)
                        rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
                        rms=float(np.sqrt(np.mean(np.sum((x@rotation-y)**2,axis=1))))
                        sample[name]={"nonrigid_rms_m":rms,"max_displacement_m":float(np.linalg.norm(p-rest[name],axis=1).max()),
                                      "minimum_z_m":float(p[:,2].min())}
                        edges=entity._edges
                        ratios=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)/entity._edges_len_rest
                        sample[name]["edge_stretch_p99"]=float(np.quantile(ratios,.99))
                        if name=="sloth":
                            tet=connectivity[name]
                            volume=lambda x: np.linalg.det(x[tet[:,1:]]-x[tet[:,:1]])
                            initial_volumes=volume(rest[name]);current_volumes=volume(p)
                            sample[name]["volume_ratio"]=float(abs(current_volumes).sum()/abs(initial_volumes).sum())
                            sample[name]["inverted_tetrahedra"]=int(np.count_nonzero(current_volumes*initial_volumes<=0))
                    samples.append(sample)
            camera.stop_recording()
            report.update(samples=samples,pinned_vertex_indices=pins,
                fixture="Suspended by a small fixed patch; gravity only; not a robot grasp or dressing test")
            (run/"report.json").write_text(json.dumps(report,indent=2))
            self.assertGreater(samples[-1]["garment"]["nonrigid_rms_m"],.01)
            self.assertGreater(samples[-1]["sloth"]["nonrigid_rms_m"],.002)
            self.assertGreater(samples[-1]["garment"]["minimum_z_m"],.1)
            self.assertTrue(all(.8<s["sloth"]["volume_ratio"]<1.2 for s in samples))
            self.assertTrue(all(s["sloth"]["inverted_tetrahedra"]==0 for s in samples))
            self.assertLess(samples[-1]["garment"]["edge_stretch_p99"],1.15)
        finally:
            gs.destroy()


@unittest.skipUnless(os.environ.get("GENESIS_SURFACE_TEST_RUN_DIR"), "opt-in GPU surface collision regression")
class SurfaceCollisionIntegrationTests(unittest.TestCase):
    def test_cloth_cannot_pass_through_triangle_interior(self):
        import genesis as gs
        from dressing_scene import write_obj,tetra_boundary
        from genesis_mesh_adapter import existing_pbd_mesh,install_cotton_solver,install_surface_contacts
        run=Path(os.environ["GENESIS_SURFACE_TEST_RUN_DIR"])/"triangle-interior"
        run.mkdir(parents=True,exist_ok=False)
        body=np.array([[-.1,-.1,.5],[.1,-.1,.5],[0,.1,.5],[0,0,.35]])
        tet=np.array([[0,1,2,3]])
        used,bound=tetra_boundary(body,tet);bf=used[bound]
        cloth=np.array([[-.02,-.02,.52],[.02,-.02,.52],[-.02,.02,.52],[.02,.02,.52]])
        cf=np.array([[0,1,2],[1,3,2]])
        write_obj(run/"body.obj",body,bf);write_obj(run/"cloth.obj",cloth,cf)
        gs.init(backend=gs.cuda,precision="32",logging_level="warning")
        try:
            scene=gs.Scene(show_viewer=False,sim_options=gs.options.SimOptions(dt=.001,gravity=(0,0,0)),
                vis_options=gs.options.VisOptions(n_support_neighbors=4),
                pbd_options=gs.options.PBDOptions(particle_size=.004,max_stretch_solver_iterations=8,max_bending_solver_iterations=4))
            install_cotton_solver(scene)
            objects={}
            for name,points,elems,mesh,mat in (
                ("sloth",body,tet,"body",gs.materials.PBD.Elastic()),
                ("garment",cloth,cf,"cloth",gs.materials.PBD.Cloth())):
                with existing_pbd_mesh(name,points,elems,.01):
                    objects[name]=scene.add_entity(morph=gs.morphs.Mesh(file=str(run/f"{mesh}.obj"),file_meshes_are_zup=True),material=mat)
            scene.build()
            settings=dict(clearance_m=.001,iterations=12,max_constraints=32768,max_correction_m=.002)
            report=install_surface_contacts(objects,bf,settings)
            objects["sloth"].fix_particles()
            objects["garment"].set_particles_vel([0,0,-5.])
            minimum=10.
            for _ in range(100):
                scene.step()
                minimum=min(minimum,float(objects["garment"].get_particles_pos().detach().cpu().numpy()[:,2].min()))
            report.update(minimum_cloth_z_m=minimum,body_surface_z_m=.5,initial_speed_m_s=5.,
                peak_constraints=objects["sloth"].solver.surface_contacts.peak_constraints)
            (run/"report.json").write_text(json.dumps(report,indent=2))
            self.assertGreater(minimum,.5004)
            self.assertGreater(report["peak_constraints"],0)
        finally:gs.destroy()

    def test_original_garment_drapes_over_original_sloth(self):
        import genesis as gs
        import trimesh
        from PIL import Image
        from types import SimpleNamespace
        from dressing_scene import prepare_assets,build_scene,write_obj,tetra_boundary,positions_stats
        run=Path(os.environ["GENESIS_SURFACE_TEST_RUN_DIR"])/"original-assets"
        run.mkdir(parents=True,exist_ok=False)
        config,paths,prepared,report=prepare_assets(Path(__file__).resolve().parent.parent/"bunnylab",run)
        with np.load(prepared/"physics_meshes.npz") as data:physics={key:data[key].copy() for key in data.files}
        # Initial placement for a gravity-driven contact test, not animation.
        physics["garment_points"]+=np.array([-.36,0,.13])
        np.savez(prepared/"physics_meshes.npz",**physics)
        write_obj(prepared/"garment.obj",physics["garment_points"],physics["garment_faces"])
        args=SimpleNamespace(backend="cuda",substeps=10,particle_size=.004,width=640,height=480,headless=True,task="settle",rpc=True)
        try:
            scene,robots,objects,cameras=build_scene(config,paths,prepared,report,args)
            camera=cameras["detail"]
            camera.start_recording(save_to_filename=str(run/"cloth-on-sloth.mp4"),fps=20)
            used,bf=tetra_boundary(physics["sloth_points"],physics["sloth_tetra"])
            samples=[]
            def intersection_count(vertices, faces, other_mesh):
                edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
                starts=vertices[edges[:,0]]
                directions=vertices[edges[:,1]]-starts
                lengths=np.linalg.norm(directions,axis=1)
                directions/=np.maximum(lengths[:,None],1e-12)
                locations,rays,_=other_mesh.ray.intersects_location(starts,directions,multiple_hits=True)
                distances=np.linalg.norm(locations-starts[rays],axis=1)
                return int(((distances>1e-6)&(distances<lengths[rays]-1e-6)).sum())
            for step in range(151):
                if step:scene.step()
                if step%5==0:
                    rgb,*_=camera.render()
                if step%10==0:
                    body=objects["sloth"].get_particles_pos().detach().cpu().numpy()
                    cloth=objects["garment"].get_particles_pos().detach().cpu().numpy()
                    mesh=trimesh.Trimesh(body[used],bf,process=False)
                    cf=physics["garment_faces"]
                    points=np.concatenate([cloth,cloth[cf].mean(axis=1)])
                    signed=trimesh.proximity.signed_distance(mesh,points)
                    cloth_mesh=trimesh.Trimesh(cloth,cf,process=False)
                    intersections=(intersection_count(cloth,cf,mesh)
                                   +intersection_count(body[used],bf,cloth_mesh))
                    sample={"edge_triangle_intersections":intersections,"time_s":step*.01,"largest_sample_inside_depth_m":float(max(0,signed.max())),
                            "objects":positions_stats(objects,config["scene"]["table_height"])}
                    samples.append(sample)
                    if step in (0,50,100,150):
                        Image.fromarray(rgb).save(run/f"contact-{step:03d}.png")
            camera.stop_recording()
            report.update(contact_samples=samples,peak_constraints=scene.sim.pbd_solver.surface_contacts.peak_constraints,
                fixture="Original garment dropped onto original sloth under gravity; no model call or scripted dressing")
            (run/"report.json").write_text(json.dumps(report,indent=2))
            self.assertGreater(report["peak_constraints"],0)
            self.assertLess(max(s["largest_sample_inside_depth_m"] for s in samples),.0002)
            self.assertEqual(sum(s["edge_triangle_intersections"] for s in samples),0)
        finally:gs.destroy()


if __name__=="__main__":unittest.main()
