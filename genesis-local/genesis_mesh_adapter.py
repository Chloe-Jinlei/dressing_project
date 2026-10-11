"""Small, version-checked adapter for pre-existing PBD triangle/tetra meshes.

Genesis 1.4.3's Mesh importer always remeshes and retetrahedralizes PBD assets.
The bunnylab tetra mesh has a non-manifold exterior edge, so that path cannot
preserve it. The import context reuses Genesis's PBD constraints and state APIs,
replacing only pre-build sampling with the supplied vertices/connectivity.
No installed Genesis files or global disk caches are modified.
"""
from contextlib import contextmanager
from importlib.metadata import version

import numpy as np
import trimesh


def simulation_connectivity(entity, kind):
    """Connectivity for the adapter's import-fidelity check (private 1.4.3 field)."""
    return entity._elems if kind == "sloth" else entity.mesh.faces


def cotton_discretization(points, faces, settings):
    """SI thin-sheet approximation on the actual mesh, not raw solver knobs.

    Each triangle contributes one third of its area to its vertices and edges.
    Edge springs approximate an isotropic membrane (exact only on regular grids).
    Hinge energy is B*l/(2*d)*theta², d=(h1+h2)/3. B is in N m.
    This does not reproduce weave anisotropy, hysteresis or measured tensile curves.
    """
    for key in ("areal_density_kg_m2", "bending_rigidity_nm", "membrane_stiffness_n_m",
                "air_density_kg_m3", "drag_coefficient"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid cotton parameter: {key}")
    tri = points[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)/2
    if np.any(areas < 1e-12):
        raise ValueError("Degenerate cloth triangle")
    vertex_area = np.bincount(faces.ravel(), weights=np.repeat(areas/3, 3), minlength=len(points))
    edges, inverse = np.unique(np.sort(np.concatenate([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_inverse=True)
    edge_area = np.bincount(inverse, weights=np.tile(areas, 3))
    length2 = np.sum((points[edges[:, 0]]-points[edges[:, 1]])**2, axis=1)
    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    adjacency, shared = trimesh.graph.face_adjacency(mesh=mesh, return_edges=True)
    hinge_length2 = np.sum((points[shared[:, 0]]-points[shared[:, 1]])**2, axis=1)
    return {
        "mass": vertex_area * settings["areal_density_kg_m2"],
        "drag": .5 * settings["air_density_kg_m3"] * settings["drag_coefficient"] * vertex_area,
        "stretch_compliance": 3*length2/(4*settings["membrane_stiffness_n_m"]*edge_area),
        "bending_compliance": 2*areas[adjacency].sum(axis=1)/(3*settings["bending_rigidity_nm"]*hinge_length2),
    }


def apply_cotton_parameters(entity, settings):
    """Populate mesh-dependent coefficients after Genesis allocates its fields."""
    values = cotton_discretization(entity._particles, entity.mesh.faces, settings)
    solver = entity.solver
    for field, key, start in (
        (solver.particles_info.mass, "mass", entity.particle_start),
        (solver.particles_info.air_resistance, "drag", entity.particle_start),
        (solver.edges_info.stretch_compliance, "stretch_compliance", entity._edge_start),
        (solver.inner_edges_info.bending_compliance, "bending_compliance", entity._inner_edge_start),
    ):
        data = field.to_numpy()
        data[start:start+len(values[key])] = values[key]
        field.from_numpy(data)
    return {"parameters": settings.copy(), "mass_kg": float(values["mass"].sum()),
            "stretch_compliance_range_m_per_n": [float(values["stretch_compliance"].min()), float(values["stretch_compliance"].max())],
            "hinge_compliance_range_per_nm": [float(values["bending_compliance"].min()), float(values["bending_compliance"].max())],
            "measured_inputs": ["areal_density_kg_m2", "bending_rigidity_nm"],
            "tensile_and_contact_calibrated": False,
            "model": "isotropic edge membrane + XPBD dihedral thin-sheet bending"}


def install_cotton_solver(scene):
    """Scene-local 1.4.3 adapter: consistent angular gradients and XPBD lambdas.

    Native bending uses the gradient of cos(theta), but adds compliance for theta
    without the sin² factor. Native iterations also omit accumulated lambda.
    Correct the native volume update's extra squared-gradient factor as well.
    Keep native sloth stretch and collision/robot behavior.
    """
    import quadrants as qd
    import genesis as gs
    from genesis.engine.solvers.pbd_solver import PBDSolver
    import genesis.utils.sdf as sdf
    from genesis.utils import array_class

    original = scene.sim.pbd_solver
    if version("genesis-world") != "1.4.3" or scene.is_built or original.n_particles:
        raise RuntimeError("Install the reviewed cotton adapter before adding PBD entities")

    @qd.data_oriented
    class CottonSolver(PBDSolver):
        def init_edge_fields(self):
            super().init_edge_fields()
            self.stretch_lambda = qd.field(gs.qd_float, shape=(max(1, self._n_edges), self._B))
            self.bend_lambda = qd.field(gs.qd_float, shape=(max(1, self._n_inner_edges), self._B))

        def init_elem_fields(self):
            super().init_elem_fields()
            self.volume_lambda = qd.field(gs.qd_float, shape=(max(1, self._n_elems), self._B))

        def substep_post_coupling(self, f):
            super().substep_post_coupling(f)
            if hasattr(self, "surface_contacts"):
                self.surface_contacts.solve()
            rigid=self._sim.rigid_solver
            if self._n_elems and rigid.is_active:
                self._guard_contact_volume(rigid.dyn_state.geoms,rigid.dyn_info.geoms,
                    rigid.collider._sdf._sdf_info,rigid.rigid_info,rigid.collider.collider_config)

        @qd.func
        def _contact_gradient(self,gradient:gs.qd_vec3,position:gs.qd_vec3,b:qd.i32,
                              geoms_state:array_class.GeomsState,geoms_info:array_class.GeomsInfo,
                              sdf_info:array_class.SDFInfo,rigid_info:array_class.RigidInfo,collider_config:qd.template()):
            # Active unilateral contact: a volume correction may leave the
            # surface or slide along it, but cannot push farther into it.
            g=gradient
            for geom in range(self._sim.rigid_solver.n_geoms):
                if geoms_info.needs_coup[geom]:
                    distance=sdf.sdf_func_world(geom,b,position,geoms_state,geoms_info,sdf_info)
                    if distance<self.particle_size/2+1e-6:
                        normal=sdf.sdf_func_normal_world(geom,b,position,geoms_state,geoms_info,
                            rigid_info,sdf_info,collider_config)
                        g-=qd.min(0.,g.dot(normal))*normal
            return g

        @qd.kernel
        def _guard_contact_volume(self,geoms_state:array_class.GeomsState,geoms_info:array_class.GeomsInfo,
                                  sdf_info:array_class.SDFInfo,rigid_info:array_class.RigidInfo,collider_config:qd.template()):
            # Table/robot projection happens after elastic constraints and can
            # flatten thin surface tets. Enforce a unilateral 10%-rest-volume
            # barrier with contact-aware gradients, not a reset to a rest pose.
            for _ in qd.static(range(16)):
                for e,b in qd.ndrange(self._n_elems,self._B):
                    ids=qd.Vector([self.elems_info[e].v1,self.elems_info[e].v2,self.elems_info[e].v3,self.elems_info[e].v4])
                    p1,p2=self.particles[ids[0],b].pos,self.particles[ids[1],b].pos
                    p3,p4=self.particles[ids[2],b].pos,self.particles[ids[3],b].pos
                    volume=(p2-p1).cross(p3-p1).dot(p4-p1)/6.
                    residual=volume-.1*self.elems_info[e].vol_rest
                    if residual<0:
                        gradients=qd.Matrix.rows([(p4-p2).cross(p3-p2)/6.,(p3-p1).cross(p4-p1)/6.,
                                                  (p4-p1).cross(p2-p1)/6.,(p2-p1).cross(p3-p1)/6.])
                        weights=qd.Vector.zero(gs.qd_float,4)
                        denominator=0.
                        for j in qd.static(range(4)):
                            gradients[j,:]=self._contact_gradient(gradients[j,:],self.particles[ids[j],b].pos,b,
                                geoms_state,geoms_info,sdf_info,rigid_info,collider_config)
                            weights[j]=self.particles[ids[j],b].free/self.particles_info[ids[j]].mass
                            denominator+=weights[j]*gradients[j,:].norm_sqr()
                        if denominator>1e-20:
                            for j in qd.static(range(4)):
                                self.particles[ids[j],b].dpos+=-.2*residual/denominator*weights[j]*gradients[j,:]
                for p,b in qd.ndrange(self._n_particles,self._B):
                    if self.particles[p,b].free:
                        self.particles[p,b].pos+=self.particles[p,b].dpos
                        self.particles[p,b].vel+=self.particles[p,b].dpos/self._substep_dt
                    self.particles[p,b].dpos.fill(0)

        @qd.kernel
        def _kernel_solve_stretch(self, f: qd.i32):
            for e, b in qd.ndrange(self._n_edges, self._B):
                self.stretch_lambda[e, b] = 0
            for iteration in qd.static(range(max(32, self._max_stretch_solver_iterations))):
                for e, b in qd.ndrange(self._n_edges, self._B):
                    v1, v2 = self.edges_info[e].v1, self.edges_info[e].v2
                    w1 = self.particles[v1, b].free / self.particles_info[v1].mass
                    w2 = self.particles[v2, b].free / self.particles_info[v2].mass
                    n = self.particles[v1, b].pos - self.particles[v2, b].pos
                    length = n.norm()
                    alpha = self.edges_info[e].stretch_compliance / self._substep_dt**2
                    residual = length - self.edges_info[e].len_rest
                    cloth = self.particles_info[v1].material_type == self.MATERIAL.CLOTH
                    if cloth:
                        residual += alpha * self.stretch_lambda[e, b]
                    elif iteration >= self._max_stretch_solver_iterations:
                        residual = 0.
                    if w1+w2 > 0 and length > 1e-10:
                        dl = -residual/(w1+w2+alpha)*self.edges_info[e].stretch_relaxation
                        if cloth:
                            self.stretch_lambda[e, b] += dl
                        self.particles[v1, b].dpos += w1*dl*n/length
                        self.particles[v2, b].dpos -= w2*dl*n/length
                for p, b in qd.ndrange(self._n_particles, self._B):
                    if self.particles[p, b].free and self.particles_info[p].material_type != self.MATERIAL.PARTICLE:
                        self.particles[p, b].pos += self.particles[p, b].dpos
                    self.particles[p, b].dpos.fill(0)

        @qd.kernel
        def _kernel_solve_bending(self, f: qd.i32):
            for e, b in qd.ndrange(self._n_inner_edges, self._B):
                self.bend_lambda[e, b] = 0
            for _ in qd.static(range(self._max_bending_solver_iterations)):
                for e, b in qd.ndrange(self._n_inner_edges, self._B):
                    v1, v2 = self.inner_edges_info[e].v1, self.inner_edges_info[e].v2
                    v3, v4 = self.inner_edges_info[e].v3, self.inner_edges_info[e].v4
                    w1 = self.particles[v1,b].free/self.particles_info[v1].mass
                    w2 = self.particles[v2,b].free/self.particles_info[v2].mass
                    w3 = self.particles[v3,b].free/self.particles_info[v3].mass
                    w4 = self.particles[v4,b].free/self.particles_info[v4].mass
                    p2 = self.particles[v2,b].pos-self.particles[v1,b].pos
                    p3 = self.particles[v3,b].pos-self.particles[v1,b].pos
                    p4 = self.particles[v4,b].pos-self.particles[v1,b].pos
                    a, c = p2.cross(p3).norm(), p2.cross(p4).norm()
                    if a > 1e-12 and c > 1e-12 and w1+w2+w3+w4 > 0:
                        n1, n2 = p2.cross(p3)/a, p2.cross(p4)/c
                        d = qd.math.clamp(n1.dot(n2), -1., 1.)
                        sine = qd.sqrt(qd.max(0., 1-d*d))
                        if sine > 1e-6:
                            # q = -grad(cos theta); grad(theta) = q/sin(theta).
                            q3 = (p2.cross(n2)+n1.cross(p2)*d)/a/sine
                            q4 = (p2.cross(n1)+n2.cross(p2)*d)/c/sine
                            q2 = (-(p3.cross(n2)+n1.cross(p3)*d)/a-(p4.cross(n1)+n2.cross(p4)*d)/c)/sine
                            q1 = -q2-q3-q4
                            denom = w1*q1.norm_sqr()+w2*q2.norm_sqr()+w3*q3.norm_sqr()+w4*q4.norm_sqr()
                            alpha = self.inner_edges_info[e].bending_compliance/self._substep_dt**2
                            residual = qd.acos(d)-qd.acos(-1.)+alpha*self.bend_lambda[e,b]
                            dl = -residual/(denom+alpha)*self.inner_edges_info[e].bending_relaxation
                            self.bend_lambda[e,b] += dl
                            self.particles[v1,b].dpos += w1*dl*q1
                            self.particles[v2,b].dpos += w2*dl*q2
                            self.particles[v3,b].dpos += w3*dl*q3
                            self.particles[v4,b].dpos += w4*dl*q4
                for p, b in qd.ndrange(self._n_particles, self._B):
                    if self.particles[p,b].free and self.particles_info[p].material_type != self.MATERIAL.PARTICLE:
                        self.particles[p,b].pos += self.particles[p,b].dpos
                    self.particles[p,b].dpos.fill(0)

        @qd.kernel
        def _kernel_solve_volume(self, f: qd.i32):
            for e,b in qd.ndrange(self._n_elems,self._B):
                self.volume_lambda[e,b]=0
            for _ in qd.static(range(self._max_volume_solver_iterations)):
                for e,b in qd.ndrange(self._n_elems,self._B):
                    v1,v2=self.elems_info[e].v1,self.elems_info[e].v2
                    v3,v4=self.elems_info[e].v3,self.elems_info[e].v4
                    p1,p2=self.particles[v1,b].pos,self.particles[v2,b].pos
                    p3,p4=self.particles[v3,b].pos,self.particles[v4,b].pos
                    g1=(p4-p2).cross(p3-p2)/6.
                    g2=(p3-p1).cross(p4-p1)/6.
                    g3=(p4-p1).cross(p2-p1)/6.
                    g4=(p2-p1).cross(p3-p1)/6.
                    w1=self.particles[v1,b].free/self.particles_info[v1].mass
                    w2=self.particles[v2,b].free/self.particles_info[v2].mass
                    w3=self.particles[v3,b].free/self.particles_info[v3].mass
                    w4=self.particles[v4,b].free/self.particles_info[v4].mass
                    denom=w1*g1.norm_sqr()+w2*g2.norm_sqr()+w3*g3.norm_sqr()+w4*g4.norm_sqr()
                    if denom>1e-20:
                        volume=(p2-p1).cross(p3-p1).dot(p4-p1)/6.
                        alpha=self.elems_info[e].volume_compliance/self._substep_dt**2
                        dl=-(volume-self.elems_info[e].vol_rest+alpha*self.volume_lambda[e,b])/(denom+alpha)*self.elems_info[e].volume_relaxation
                        self.volume_lambda[e,b]+=dl
                        self.particles[v1,b].dpos+=w1*dl*g1
                        self.particles[v2,b].dpos+=w2*dl*g2
                        self.particles[v3,b].dpos+=w3*dl*g3
                        self.particles[v4,b].dpos+=w4*dl*g4
                for p,b in qd.ndrange(self._n_particles,self._B):
                    if self.particles[p,b].free and self.particles_info[p].material_type != self.MATERIAL.PARTICLE:
                        self.particles[p,b].pos+=self.particles[p,b].dpos
                    self.particles[p,b].dpos.fill(0)

    solver = CottonSolver(scene, scene.sim, original._options)
    scene.sim._solvers[scene.sim._solvers.index(original)] = solver
    scene.sim.pbd_solver = solver
    scene.sim.coupler.pbd_solver = solver


def install_rigid_pbd_friction(scene):
    """Enable Coulomb contact friction on this scene, before kernel compilation.

    Genesis 1.4.3's legacy contact performs only normal SDF projection. Use
    position-level friction (Macklin et al. 2014, section 6.1), consistent with
    Genesis's PBD particle contacts. The normal projection is the friction
    budget; there is no attraction, world-position pinning, or velocity reset.
    Both coefficients use the smaller of the particle and rigid coefficients.
    Attached particles retain the existing contact/attachment behavior.
    """
    import quadrants as qd
    import genesis as gs
    import genesis.utils.sdf as sdf
    from genesis.engine.couplers.legacy_coupler import LegacyCoupler
    from genesis.utils import array_class

    if version("genesis-world") != "1.4.3":
        raise RuntimeError("Rigid/PBD friction requires the reviewed Genesis 1.4.3 contact interface")
    original = scene.sim.coupler
    if type(original) is not LegacyCoupler or scene.is_built:
        raise RuntimeError("Install rigid/PBD friction once, on an unbuilt legacy-coupler scene")

    @qd.data_oriented
    class FrictionCoupler(LegacyCoupler):
        @qd.func
        def _func_pbd_collide_with_rigid_geom(
            self, i, pos_world, vel, mass, normal_prev, geom_idx, batch_idx,
            geoms_state: array_class.GeomsState,
            geoms_info: array_class.GeomsInfo,
            links_state: array_class.LinksState,
            sdf_info: array_class.SDFInfo,
            rigid_info: array_class.RigidInfo,
            collider_static_config: qd.template(),
        ):
            distance = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
            normal = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info,
                rigid_info, sdf_info, collider_static_config)
            new_pos, new_vel = pos_world, vel
            depth = self.pbd_solver.particle_size / 2 - distance
            if depth > 0:
                new_pos = pos_world + normal * depth
                previous = self.pbd_solver.particles_reordered[i, batch_idx].ipos
                if self.pbd_solver.particles_reordered[i, batch_idx].free:
                    rigid_velocity = self.rigid_solver._func_vel_at_point(
                        pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx],
                        i_b=batch_idx, links_state=links_state)
                    relative_delta = new_pos - previous - rigid_velocity * self.pbd_solver._substep_dt
                    tangent = relative_delta - normal * relative_delta.dot(normal)
                    tangent_length = tangent.norm()
                    mu_s = qd.min(self.pbd_solver.particles_info_reordered[i, batch_idx].mu_s,
                                  geoms_info.coup_friction[geom_idx])
                    mu_k = qd.min(self.pbd_solver.particles_info_reordered[i, batch_idx].mu_k,
                                  geoms_info.coup_friction[geom_idx])
                    if tangent_length > 1e-12:
                        fraction = 1.0
                        if tangent_length > mu_s * depth:
                            fraction = qd.min(1.0, mu_k * depth / tangent_length)
                        new_pos -= fraction * tangent
                new_vel = (new_pos - previous) / self.pbd_solver._substep_dt
                # Preserve the original equal-and-opposite rigid coupling force,
                # including the tangential impulse introduced by friction.
                force = -mass * (new_vel - vel) / self.rigid_solver._substep_dt
                self.rigid_solver._func_apply_coupling_force(
                    geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state)
            return new_pos, new_vel, normal

    scene.sim._coupler = FrictionCoupler(scene.sim, original.options)
    return {"adapter": "scene-local Genesis 1.4.3 legacy Coulomb friction",
            "method": "position-level static/kinetic friction limited by normal SDF projection",
            "coefficient_combination": "minimum(particle coefficient, rigid coup_friction)",
            "applies_to": "free PBD particles contacting coupled rigid geometry",
            "normal_projection_unchanged": True, "reciprocal_rigid_force": True,
            "world_position_pinning": False, "installed_genesis_modified": False}


@contextmanager
def existing_pbd_mesh(kind, points, elements, mass):
    import genesis as gs
    import genesis.engine.solvers.pbd_solver as solver_module

    if version("genesis-world") != "1.4.3":
        raise RuntimeError("The existing-mesh adapter was verified with Genesis 1.4.3; review it before using another version.")
    class_name = "PBD3DEntity" if kind == "sloth" else "PBD2DEntity"
    original_class = getattr(solver_module, class_name)

    def sample(self):
        # Files have already been transformed into world coordinates.
        if tuple(self.morph.pos) != (0., 0., 0.) or tuple(self.morph.quat) != (1., 0., 0., 0.):
            raise ValueError("Existing PBD mesh must use an identity morph transform")
        self._sampled_pos = np.zeros(3, dtype=gs.np_float)
        self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
        self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_int)
        self._mesh = self._vmesh.copy()
        self._particles = np.asarray(points, dtype=gs.np_float)
        self._init_particles_offset = gs.tensor(self._particles)
        self._mass = float(mass)
        self._n_particles = len(self._particles)
        self._particle_mass = self._mass / self._n_particles
        elems = np.asarray(elements, dtype=gs.np_int)
        if kind == "sloth":
            self._elems = elems.copy()
            volumes = np.linalg.det(self._particles[elems[:, 1:]] - self._particles[elems[:, :1]]) / 6
            if np.any(np.abs(volumes) < 1e-14):
                raise ValueError("Source sloth mesh contains degenerate tetrahedra")
            # Normalize orientation; vertex sets and tetra connectivity stay identical.
            negative = volumes < 0
            self._elems[negative] = self._elems[negative][:, [0, 2, 1, 3]]
            self._elems_vol_rest = np.abs(volumes)
            self._edges = np.unique(np.sort(np.concatenate([
                self._elems[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), axis=1), axis=0)
        else:
            self._mesh = gs.Mesh.from_trimesh(trimesh.Trimesh(
                vertices=self._particles, faces=elems, process=False), surface=self.surface)
            self._edges = np.unique(np.sort(np.concatenate([
                elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]]), axis=1), axis=0)
            adjacency, shared = trimesh.graph.face_adjacency(mesh=self._mesh.trimesh, return_edges=True)
            opposite1 = elems[adjacency[:, 0]].sum(axis=1) - shared.sum(axis=1)
            opposite2 = elems[adjacency[:, 1]].sum(axis=1) - shared.sum(axis=1)
            self._inner_edges = np.column_stack([shared, opposite1, opposite2]).astype(gs.np_int)
            self._inner_edges_len_rest = np.linalg.norm(
                self._particles[opposite1] - self._particles[opposite2], axis=1)
        self._edges = self._edges.astype(gs.np_int, copy=False)
        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1)
    # Keep the original entity class: solver counts use isinstance(), so
    # temporarily replacing that class would miss previously imported meshes.
    original_sample = original_class.sample
    original_class.sample = sample
    try:
        yield
    finally:
        original_class.sample = original_sample


def install_surface_contacts(objects, body_faces, settings):
    """Triangle-level cloth/body contact shared by Route A and scene demos.

    Both vertex/face directions and edge/edge pairs are tested every substep.
    Swept AABBs and previous-side tests catch crossings within a small substep;
    this is not a certified general-purpose continuous collision detector.
    """
    import quadrants as qd
    import genesis as gs
    from scipy.spatial import cKDTree

    body, cloth = objects["sloth"], objects["garment"]
    solver = body.solver
    if solver._B != 1:
        raise ValueError("Surface contacts currently require one simulation environment")
    margin=float(settings["clearance_m"])
    capacity=int(settings["max_constraints"])
    iterations=int(settings["iterations"])
    cap=float(settings["max_correction_m"])
    if not (0 < margin <= .004 and 1 <= iterations <= 64 and capacity >= 1024 and 0 < cap <= .005):
        raise ValueError("Invalid surface contact settings")
    bf=np.asarray(body_faces,dtype=np.int32)+body.particle_start
    cf=np.asarray(cloth.mesh.faces,dtype=np.int32)+cloth.particle_start
    faces=np.concatenate([bf,cf])
    body_vertices=np.unique(bf)
    vertices=np.concatenate([body_vertices,np.unique(cf)])
    def edges(faces):
        return np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    be,ce=edges(bf),edges(cf)
    # These assets already have a physical vertex at every visual vertex.
    # Exact binding prevents skinning across front/back sheets at small gaps.
    for entity in objects.values():
        distance,indices=cKDTree(entity._particles).query(entity._vverts)
        if distance.max()>1e-6:
            raise ValueError("Visual mesh does not correspond to source physical vertices")
        idx=solver.vverts_info.support_idxs.to_numpy()
        weights=solver.vverts_info.support_weights.to_numpy()
        start,end=entity.vvert_start,entity.vvert_end
        idx[start:end,:]=indices[:,None]+entity.particle_start
        weights[start:end,:]=0
        weights[start:end,0]=1
        solver.vverts_info.support_idxs.from_numpy(idx)
        solver.vverts_info.support_weights.from_numpy(weights)

    @qd.data_oriented
    class SurfaceContacts:
        def __init__(self):
            self.solver=solver
            self.nv=len(vertices);self.nf=len(faces);self.nb=len(body_vertices);self.nbf=len(bf)
            self.neb=len(be);self.nec=len(ce)
            self.vertices=qd.field(qd.i32,shape=self.nv);self.vertices.from_numpy(vertices.astype(np.int32))
            self.faces=qd.Vector.field(3,qd.i32,shape=self.nf);self.faces.from_numpy(faces)
            self.be=qd.Vector.field(2,qd.i32,shape=self.neb);self.be.from_numpy(be)
            self.ce=qd.Vector.field(2,qd.i32,shape=self.nec);self.ce.from_numpy(ce)
            self.lo=qd.Vector.field(3,gs.qd_float,shape=self.nf)
            self.hi=qd.Vector.field(3,gs.qd_float,shape=self.nf)
            self.ids=qd.Vector.field(4,qd.i32,shape=capacity)
            self.coeff=qd.Vector.field(4,gs.qd_float,shape=capacity)
            self.normal=qd.Vector.field(3,gs.qd_float,shape=capacity)
            self.count=qd.field(qd.i32,shape=())
            self.degree=qd.field(qd.i32,shape=solver.n_particles)
            self.residual=qd.field(gs.qd_float,shape=())
            self.peak_constraints=0

        @qd.func
        def pos(self,i):return self.solver.particles[i,0].pos

        @qd.func
        def old(self,i):return self.solver.particles[i,0].ipos

        @qd.func
        def closest(self,p,a,b,c):
            ab,ac=b-a,c-a
            aa,bb,cc=ab.dot(ab),ab.dot(ac),ac.dot(ac)
            ap=p-a
            denominator=aa*cc-bb*bb
            bary=qd.Vector([1.,0.,0.])
            projected=qd.Vector([-1.,-1.,-1.])
            if denominator>1e-20:
                v=(cc*ap.dot(ab)-bb*ap.dot(ac))/denominator
                w=(aa*ap.dot(ac)-bb*ap.dot(ab))/denominator
                projected=qd.Vector([1-v-w,v,w])
            if projected.min()>=0:
                bary=projected
            else:
                best=gs.qd_float(1e20)
                for j in qd.static(range(3)):
                    x,y=a,b
                    if qd.static(j==1):x,y=b,c
                    if qd.static(j==2):x,y=c,a
                    t=qd.math.clamp((p-x).dot(y-x)/qd.max((y-x).norm_sqr(),1e-20),0.,1.)
                    d=(p-(x+t*(y-x))).norm_sqr()
                    if d<best:
                        best=d;bary=qd.Vector.zero(gs.qd_float,3)
                        bary[j]=1-t;bary[(j+1)%3]=t
            return bary,projected.min()>=-1e-5

        @qd.func
        def segments(self,a,b,c,d):
            u,v,w=b-a,d-c,a-c
            aa,bb,cc=u.dot(u),u.dot(v),v.dot(v)
            dd,ee=u.dot(w),v.dot(w)
            denominator=aa*cc-bb*bb
            s=gs.qd_float(0.)
            if denominator>1e-20:s=qd.math.clamp((bb*ee-cc*dd)/denominator,0.,1.)
            t=(bb*s+ee)/qd.max(cc,1e-20)
            if t<0:t=0.;s=qd.math.clamp(-dd/qd.max(aa,1e-20),0.,1.)
            elif t>1:t=1.;s=qd.math.clamp((bb-dd)/qd.max(aa,1e-20),0.,1.)
            return s,t

        @qd.func
        def add(self,ids,coeff,normal):
            free=gs.qd_float(0.)
            for j in qd.static(range(4)):
                free+=self.solver.particles[ids[j],0].free*coeff[j]**2
            if free>1e-12:
                k=qd.atomic_add(self.count[None],1)
                if k<capacity:
                    self.ids[k]=ids;self.coeff[k]=coeff;self.normal[k]=normal
                    for j in qd.static(range(4)):
                        if qd.abs(coeff[j])>1e-6:qd.atomic_add(self.degree[ids[j]],1)

        @qd.kernel
        def bounds(self):
            for f in range(self.nf):
                lo=qd.Vector([1e10,1e10,1e10]);hi=-lo
                for j in qd.static(range(3)):
                    i=self.faces[f][j]
                    lo=qd.min(lo,qd.min(self.pos(i),self.old(i)))
                    hi=qd.max(hi,qd.max(self.pos(i),self.old(i)))
                self.lo[f]=lo-margin;self.hi[f]=hi+margin

        @qd.kernel
        def vertex_faces(self):
            for v,f in qd.ndrange(self.nv,self.nf):
                if (v<self.nb)!=(f<self.nbf):
                    i=self.vertices[v];tri=self.faces[f]
                    p,p0=self.pos(i),self.old(i)
                    if (qd.max(p,p0)>=self.lo[f]).all() and (qd.min(p,p0)<=self.hi[f]).all():
                        a,b,c=self.pos(tri[0]),self.pos(tri[1]),self.pos(tri[2])
                        a0,b0,c0=self.old(tri[0]),self.old(tri[1]),self.old(tri[2])
                        bary,inside=self.closest(p,a,b,c)
                        delta=p-(bary[0]*a+bary[1]*b+bary[2]*c)
                        distance=delta.norm()
                        n=(b-a).cross(c-a).normalized(1e-12)
                        n0=(b0-a0).cross(c0-a0).normalized(1e-12)
                        before=(p0-a0).dot(n0);after=(p-a).dot(n)
                        crossing=inside and before*after<0
                        if distance<margin or crossing:
                            normal=delta.normalized(1e-12)
                            if crossing or distance<1e-8:
                                normal=n
                                if before<0:normal=-n
                            self.add(qd.Vector([i,tri[0],tri[1],tri[2]]),
                                     qd.Vector([1.,-bary[0],-bary[1],-bary[2]]),normal)

        @qd.kernel
        def edge_edges(self):
            for e,f in qd.ndrange(self.neb,self.nec):
                x,y=self.be[e],self.ce[f]
                a,b,c,d=self.pos(x[0]),self.pos(x[1]),self.pos(y[0]),self.pos(y[1])
                a0,b0,c0,d0=self.old(x[0]),self.old(x[1]),self.old(y[0]),self.old(y[1])
                lo1=qd.min(qd.min(a,b),qd.min(a0,b0));hi1=qd.max(qd.max(a,b),qd.max(a0,b0))
                lo2=qd.min(qd.min(c,d),qd.min(c0,d0));hi2=qd.max(qd.max(c,d),qd.max(c0,d0))
                if (hi1+margin>=lo2).all() and (hi2+margin>=lo1).all():
                    s,t=self.segments(a,b,c,d)
                    delta=(1-s)*a+s*b-(1-t)*c-t*d
                    before=(1-s)*a0+s*b0-(1-t)*c0-t*d0
                    normal=(b-a).cross(d-c).normalized(1e-12)
                    crossing=(before.dot(normal)*delta.dot(normal)<0 and s>1e-5 and s<1-1e-5 and t>1e-5 and t<1-1e-5)
                    distance=delta.norm()
                    if distance<margin or crossing:
                        n=delta.normalized(1e-12)
                        if crossing or distance<1e-8:
                            n=normal
                            if before.dot(n)<0:n=-n
                        if n.norm_sqr()>.5:
                            self.add(qd.Vector([x[0],x[1],y[0],y[1]]),qd.Vector([1-s,s,t-1,-t]),n)

        @qd.kernel
        def project(self):
            for k in range(self.count[None]):
                ids,w,n=self.ids[k],self.coeff[k],self.normal[k]
                difference=qd.Vector.zero(gs.qd_float,3)
                inverse=qd.Vector.zero(gs.qd_float,4)
                denominator=gs.qd_float(0.);degree=1;maximum=gs.qd_float(0.)
                for j in qd.static(range(4)):
                    difference+=w[j]*self.pos(ids[j])
                    inverse[j]=self.solver.particles[ids[j],0].free/self.solver.particles_info[ids[j]].mass
                    denominator+=inverse[j]*w[j]**2
                    if qd.abs(w[j])>1e-6:
                        degree=qd.max(degree,self.degree[ids[j]])
                        maximum=qd.max(maximum,inverse[j]*qd.abs(w[j]))
                violation=margin-difference.dot(n)
                if violation>0 and denominator>1e-15:
                    impulse=qd.min(violation/denominator,cap/qd.max(maximum,1e-15))*.9/degree
                    for j in qd.static(range(4)):
                        self.solver.particles[ids[j],0].dpos+=inverse[j]*w[j]*impulse*n
            for p in range(self.solver.n_particles):
                delta=self.solver.particles[p,0].dpos
                if self.solver.particles[p,0].free:
                    self.solver.particles[p,0].pos+=delta
                    self.solver.particles[p,0].vel+=delta/self.solver._substep_dt
                self.solver.particles[p,0].dpos.fill(0)

        @qd.kernel
        def check(self):
            for k in range(self.count[None]):
                d=qd.Vector.zero(gs.qd_float,3)
                for j in qd.static(range(4)):d+=self.coeff[k][j]*self.pos(self.ids[k][j])
                qd.atomic_max(self.residual[None],margin-d.dot(self.normal[k]))

        def solve(self):
            self.count.fill(0);self.degree.fill(0)
            self.bounds();self.vertex_faces();self.edge_edges()
            count=int(self.count[None]);self.peak_constraints=max(self.peak_constraints,count)
            if count>=capacity:raise RuntimeError("Surface contact buffer full; stopping rather than dropping collisions")
            if not count:return
            residual=0.
            for batch in range(4):
                for _ in range(iterations):self.project()
                self.residual.fill(0);self.check();residual=float(self.residual[None])
                if residual<margin*.5:break
            if residual>margin*.95:
                raise RuntimeError(f"Unresolved cloth/body surface contact ({residual:.6g} m); simulation stopped")

    contact=SurfaceContacts()
    solver.surface_contacts=contact
    return {"method":"bidirectional vertex-triangle and edge-edge, swept AABB and previous-side crossing checks",
            "scope":"cloth versus sloth, every physics substep; shared by Route A and scene demos",
            "clearance_m":margin,"iterations_per_batch":iterations,"maximum_batches":4,
            "max_constraints":capacity,"visual_binding":"exact original physical vertex",
            "cloth_self_collision":"native particle contacts; not certified triangle self-contact",
            "general_continuous_collision_guarantee":False}
